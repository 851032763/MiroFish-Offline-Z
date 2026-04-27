"""
Neo4jStorage — GraphStorage 的 Neo4j 社区版实现。

将所有 Zep Cloud API 调用替换为本地 Neo4j Cypher 查询。
包括：增删改查、基于 NER/RE 的文本摄入、混合搜索、重试逻辑。
"""

import json
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

from neo4j import GraphDatabase, Session as Neo4jSession
from neo4j.exceptions import (
    TransientError,
    ServiceUnavailable,
    SessionExpired,
)

from ..config import Config
from .graph_storage import GraphStorage
from .embedding_service import EmbeddingService
from .ner_extractor import NERExtractor
from .search_service import SearchService
from . import neo4j_schema

logger = logging.getLogger('mirofish.neo4j_storage')


class Neo4jStorage(GraphStorage):
    """GraphStorage 接口的 Neo4j 社区版实现。"""

    MAX_RETRIES = 3
    RETRY_DELAY_BASE = 1  # 秒

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        ner_extractor: Optional[NERExtractor] = None,
    ):
        self._uri = uri or Config.NEO4J_URI
        self._user = user or Config.NEO4J_USER
        self._password = password or Config.NEO4J_PASSWORD

        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )
        self._embedding = embedding_service or EmbeddingService()
        self._ner = ner_extractor or NERExtractor()
        self._search = SearchService(self._embedding)

        # 初始化模式（索引、约束）
        self._ensure_schema()

    def close(self):
        """关闭 Neo4j 驱动连接。"""
        self._driver.close()

    def _ensure_schema(self):
        """如果索引和约束不存在，则创建它们。"""
        with self._driver.session() as session:
            for query in neo4j_schema.ALL_SCHEMA_QUERIES:
                try:
                    session.run(query)
                except Exception as e:
                    logger.warning(f"模式查询警告（可能已存在）：{e}")

    # ----------------------------------------------------------------
    # 重试包装器
    # ----------------------------------------------------------------

    def _call_with_retry(self, func, *args, **kwargs):
        """
        在 Neo4j 临时错误时重试执行函数。

        替换 Zep 代码库中的 3 种不同重试模式。
        """
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (TransientError, ServiceUnavailable, SessionExpired) as e:
                last_error = e
                wait = self.RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    f"Neo4j 临时错误（尝试 {attempt + 1}/{self.MAX_RETRIES}），"
                    f"{wait} 秒后重试：{e}"
                )
                time.sleep(wait)
            except Exception:
                raise

        raise last_error  # type: ignore

    # ----------------------------------------------------------------
    # 图生命周期
    # ----------------------------------------------------------------

    def create_graph(self, name: str, description: str = "") -> str:
        graph_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        def _create(tx):
            tx.run(
                """
                CREATE (g:Graph {
                    graph_id: $graph_id,
                    name: $name,
                    description: $description,
                    ontology_json: '{}',
                    created_at: $created_at
                })
                """,
                graph_id=graph_id,
                name=name,
                description=description,
                created_at=now,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _create)

        logger.info(f"已创建图 '{name}'，ID 为 {graph_id}")
        return graph_id

    def delete_graph(self, graph_id: str) -> None:
        def _delete(tx):
            # 删除所有实体及其关系
            tx.run(
                "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                gid=graph_id,
            )
            # 删除图节点
            tx.run(
                "MATCH (g:Graph {graph_id: $gid}) DELETE g",
                gid=graph_id,
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _delete)
        logger.info(f"已删除图 {graph_id}")

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        def _set(tx):
            tx.run(
                """
                MATCH (g:Graph {graph_id: $gid})
                SET g.ontology_json = $ontology_json
                """,
                gid=graph_id,
                ontology_json=json.dumps(ontology, ensure_ascii=False),
            )

        with self._driver.session() as session:
            self._call_with_retry(session.execute_write, _set)

    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (g:Graph {graph_id: $gid}) RETURN g.ontology_json AS oj",
                gid=graph_id,
            )
            record = result.single()
            if record and record["oj"]:
                return json.loads(record["oj"])
            return {}

    # ----------------------------------------------------------------
    # 添加数据（NER → 节点/边）
    # ----------------------------------------------------------------

    def add_text(self, graph_id: str, text: str) -> str:
        """处理文本：NER/RE → 批量嵌入 → 创建节点/边 → 返回 episode_id。"""
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # 获取用于 NER 指导的本体
        ontology = self.get_ontology(graph_id)

        # 提取实体和关系
        logger.info(f"[add_text] 开始对文本块进行 NER 提取（{len(text)} 字符）...")
        extraction = self._ner.extract(text, ontology)
        entities = extraction.get("entities", [])
        relations = extraction.get("relations", [])

        logger.info(
            f"[add_text] NER 完成：{len(entities)} 个实体，{len(relations)} 个关系"
        )

        # --- 一次性批量嵌入所有文本 ---
        entity_summaries = [f"{e['name']} ({e['type']})" for e in entities]
        fact_texts = [r.get("fact", f"{r['source']} {r['type']} {r['target']}") for r in relations]
        all_texts_to_embed = entity_summaries + fact_texts

        all_embeddings: list = []
        if all_texts_to_embed:
            logger.info(f"[add_text] 正在批量嵌入 {len(all_texts_to_embed)} 个文本...")
            try:
                all_embeddings = self._embedding.embed_batch(all_texts_to_embed)
            except Exception as e:
                logger.warning(f"[add_text] 批量嵌入失败，回退到空列表：{e}")
                all_embeddings = [[] for _ in all_texts_to_embed]

        entity_embeddings = all_embeddings[:len(entities)]
        relation_embeddings = all_embeddings[len(entities):]
        logger.info(f"[add_text] 嵌入完成，正在写入 Neo4j...")

        with self._driver.session() as session:
            # 创建 episode 节点
            def _create_episode(tx):
                tx.run(
                    """
                    CREATE (ep:Episode {
                        uuid: $uuid,
                        graph_id: $graph_id,
                        data: $data,
                        processed: true,
                        created_at: $created_at
                    })
                    """,
                    uuid=episode_id,
                    graph_id=graph_id,
                    data=text,
                    created_at=now,
                )

            self._call_with_retry(session.execute_write, _create_episode)

            # MERGE 实体（按 graph_id + name + 主标签进行 upsert）
            entity_uuid_map: Dict[str, str] = {}  # name_lower -> uuid
            for idx, entity in enumerate(entities):
                ename = entity["name"]
                etype = entity["type"]
                attrs = entity.get("attributes", {})
                summary_text = entity_summaries[idx]
                embedding = entity_embeddings[idx] if idx < len(entity_embeddings) else []

                e_uuid = str(uuid.uuid4())
                entity_uuid_map[ename.lower()] = e_uuid

                def _merge_entity(tx, _uuid=e_uuid, _name=ename, _type=etype,
                                  _attrs=attrs, _embedding=embedding,
                                  _summary=summary_text, _now=now):
                    # 按 graph_id + 小写名称 MERGE 以去重
                    result = tx.run(
                        """
                        MERGE (n:Entity {graph_id: $gid, name_lower: $name_lower})
                        ON CREATE SET
                            n.uuid = $uuid,
                            n.name = $name,
                            n.summary = $summary,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding,
                            n.created_at = $now
                        ON MATCH SET
                            n.summary = CASE WHEN n.summary = '' OR n.summary IS NULL
                                THEN $summary ELSE n.summary END,
                            n.attributes_json = $attrs_json,
                            n.embedding = $embedding
                        RETURN n.uuid AS uuid
                        """,
                        gid=graph_id,
                        name_lower=_name.lower(),
                        uuid=_uuid,
                        name=_name,
                        summary=_summary,
                        attrs_json=json.dumps(_attrs, ensure_ascii=False),
                        embedding=_embedding,
                        now=_now,
                    )
                    record = result.single()
                    return record["uuid"] if record else _uuid

                actual_uuid = self._call_with_retry(session.execute_write, _merge_entity)
                entity_uuid_map[ename.lower()] = actual_uuid

                # 添加实体类型标签
                if etype and etype != "Entity":
                    try:
                        def _add_label(tx, _name_lower=ename.lower()):
                            tx.run(
                                f"MATCH (n:Entity {{graph_id: $gid, name_lower: $nl}}) SET n:`{etype}`",
                                gid=graph_id,
                                nl=_name_lower,
                            )
                        self._call_with_retry(session.execute_write, _add_label)
                    except Exception as e:
                        logger.warning(f"无法为 '{ename}' 添加标签 '{etype}'：{e}")

            # 创建关系
            for idx, relation in enumerate(relations):
                source_name = relation["source"]
                target_name = relation["target"]
                rtype = relation["type"]
                fact = relation["fact"]

                source_uuid = entity_uuid_map.get(source_name.lower())
                target_uuid = entity_uuid_map.get(target_name.lower())

                if not source_uuid or not target_uuid:
                    logger.warning(
                        f"跳过关系 {source_name}->{target_name}："
                        f"在提取结果中未找到实体"
                    )
                    continue

                fact_embedding = relation_embeddings[idx] if idx < len(relation_embeddings) else []
                r_uuid = str(uuid.uuid4())

                def _create_relation(tx, _r_uuid=r_uuid, _source_uuid=source_uuid,
                                     _target_uuid=target_uuid, _rtype=rtype,
                                     _fact=fact, _fact_emb=fact_embedding,
                                     _episode_id=episode_id, _now=now):
                    tx.run(
                        """
                        MATCH (src:Entity {uuid: $src_uuid})
                        MATCH (tgt:Entity {uuid: $tgt_uuid})
                        CREATE (src)-[r:RELATION {
                            uuid: $uuid,
                            graph_id: $gid,
                            name: $name,
                            fact: $fact,
                            fact_embedding: $fact_embedding,
                            attributes_json: '{}',
                            episode_ids: [$episode_id],
                            created_at: $now,
                            valid_at: null,
                            invalid_at: null,
                            expired_at: null
                        }]->(tgt)
                        """,
                        src_uuid=_source_uuid,
                        tgt_uuid=_target_uuid,
                        uuid=_r_uuid,
                        gid=graph_id,
                        name=_rtype,
                        fact=_fact,
                        fact_embedding=_fact_emb,
                        episode_id=_episode_id,
                        now=_now,
                    )

                self._call_with_retry(session.execute_write, _create_relation)

        logger.info(f"[add_text] 文本块处理完成：episode={episode_id}")
        return episode_id

    def add_text_batch(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """批量添加文本块，并报告进度。"""
        episode_ids = []
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            if not chunk or not chunk.strip():
                continue
            episode_id = self.add_text(graph_id, chunk)
            episode_ids.append(episode_id)

            if progress_callback:
                progress = (i + 1) / total
                progress_callback(progress)

            logger.info(f"已处理文本块 {i + 1}/{total}")

        return episode_ids

    def wait_for_processing(
        self,
        episode_ids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ) -> None:
        """无操作 —— 在 Neo4j 中处理是同步的。"""
        if progress_callback:
            progress_callback(1.0)

    # ----------------------------------------------------------------
    # 读取节点
    # ----------------------------------------------------------------

    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                ORDER BY n.created_at DESC
                LIMIT $limit
                """,
                gid=graph_id,
                limit=limit,
            )
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                "MATCH (n:Entity {uuid: $uuid}) RETURN n, labels(n) AS labels",
                uuid=uuid,
            )
            record = result.single()
            if record:
                return self._node_to_dict(record["n"], record["labels"])
            return None

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """O(1) Cypher —— 不像旧的 Zep 代码那样进行全表扫描 + 过滤。"""
        def _read(tx):
            result = tx.run(
                """
                MATCH (n:Entity {uuid: $uuid})-[r:RELATION]-(m:Entity)
                RETURN r, startNode(r).uuid AS src_uuid, endNode(r).uuid AS tgt_uuid
                """,
                uuid=node_uuid,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        def _read(tx):
            # 查询中的动态标签（安全 —— 标签来自本体，而非用户输入）
            query = f"""
                MATCH (n:Entity:`{label}` {{graph_id: $gid}})
                RETURN n, labels(n) AS labels
            """
            result = tx.run(query, gid=graph_id)
            return [self._node_to_dict(record["n"], record["labels"]) for record in result]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # 读取边
    # ----------------------------------------------------------------

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        def _read(tx):
            result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid
                ORDER BY r.created_at DESC
                """,
                gid=graph_id,
            )
            return [
                self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                for record in result
            ]

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # 搜索
    # ----------------------------------------------------------------

    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ):
        """
        混合搜索 —— 返回匹配范围的结果。

        返回包含 'edges' 和/或 'nodes' 列表的字典
       （调用者如 zep_tools 会将其包装为 SearchResult）。
        """
        result = {"edges": [], "nodes": [], "query": query}

        with self._driver.session() as session:
            if scope in ("edges", "both"):
                result["edges"] = self._search.search_edges(
                    session, graph_id, query, limit
                )

            if scope in ("nodes", "both"):
                result["nodes"] = self._search.search_nodes(
                    session, graph_id, query, limit
                )

        return result

    # ----------------------------------------------------------------
    # 图信息
    # ----------------------------------------------------------------

    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        def _read(tx):
            # 统计节点数
            node_result = tx.run(
                "MATCH (n:Entity {graph_id: $gid}) RETURN count(n) AS cnt",
                gid=graph_id,
            )
            node_count = node_result.single()["cnt"]

            # 统计边数
            edge_result = tx.run(
                "MATCH ()-[r:RELATION {graph_id: $gid}]->() RETURN count(r) AS cnt",
                gid=graph_id,
            )
            edge_count = edge_result.single()["cnt"]

            # 不同的实体类型
            label_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                UNWIND labels(n) AS lbl
                WITH lbl WHERE lbl <> 'Entity'
                RETURN DISTINCT lbl
                """,
                gid=graph_id,
            )
            entity_types = [record["lbl"] for record in label_result]

            return {
                "graph_id": graph_id,
                "node_count": node_count,
                "edge_count": edge_count,
                "entity_types": entity_types,
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        完整图转储，包含增强的边格式（用于前端）。
        包含派生字段：fact_type、source_node_name、target_node_name。
        """
        def _read(tx):
            # 获取所有节点
            node_result = tx.run(
                """
                MATCH (n:Entity {graph_id: $gid})
                RETURN n, labels(n) AS labels
                """,
                gid=graph_id,
            )
            nodes = []
            node_map: Dict[str, str] = {}  # uuid -> name
            for record in node_result:
                nd = self._node_to_dict(record["n"], record["labels"])
                nodes.append(nd)
                node_map[nd["uuid"]] = nd["name"]

            # 获取所有边及其源/目标节点名称（JOIN）
            edge_result = tx.run(
                """
                MATCH (src:Entity)-[r:RELATION {graph_id: $gid}]->(tgt:Entity)
                RETURN r, src.uuid AS src_uuid, tgt.uuid AS tgt_uuid,
                       src.name AS src_name, tgt.name AS tgt_name
                """,
                gid=graph_id,
            )
            edges = []
            for record in edge_result:
                ed = self._edge_to_dict(record["r"], record["src_uuid"], record["tgt_uuid"])
                # 前端用的增强字段
                ed["fact_type"] = ed["name"]
                ed["source_node_name"] = record["src_name"] or ""
                ed["target_node_name"] = record["tgt_name"] or ""
                # 旧版本别名
                ed["episodes"] = ed.get("episode_ids", [])
                edges.append(ed)

            return {
                "graph_id": graph_id,
                "nodes": nodes,
                "edges": edges,
                "node_count": len(nodes),
                "edge_count": len(edges),
            }

        with self._driver.session() as session:
            return self._call_with_retry(session.execute_read, _read)

    # ----------------------------------------------------------------
    # 字典转换辅助方法
    # ----------------------------------------------------------------

    @staticmethod
    def _node_to_dict(node, labels: List[str]) -> Dict[str, Any]:
        """将 Neo4j 节点转换为标准节点字典格式。"""
        props = dict(node)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # 从字典中移除内部字段
        props.pop("embedding", None)
        props.pop("name_lower", None)

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "labels": [l for l in labels if l != "Entity"] if labels else [],
            "summary": props.get("summary", ""),
            "attributes": attributes,
            "created_at": props.get("created_at"),
        }

    @staticmethod
    def _edge_to_dict(rel, source_uuid: str, target_uuid: str) -> Dict[str, Any]:
        """将 Neo4j 关系转换为标准边字典格式。"""
        props = dict(rel)
        attrs_json = props.pop("attributes_json", "{}")
        try:
            attributes = json.loads(attrs_json) if attrs_json else {}
        except (json.JSONDecodeError, TypeError):
            attributes = {}

        # 移除内部字段
        props.pop("fact_embedding", None)

        episode_ids = props.get("episode_ids", [])
        if episode_ids and not isinstance(episode_ids, list):
            episode_ids = [str(episode_ids)]

        return {
            "uuid": props.get("uuid", ""),
            "name": props.get("name", ""),
            "fact": props.get("fact", ""),
            "source_node_uuid": source_uuid,
            "target_node_uuid": target_uuid,
            "attributes": attributes,
            "created_at": props.get("created_at"),
            "valid_at": props.get("valid_at"),
            "invalid_at": props.get("invalid_at"),
            "expired_at": props.get("expired_at"),
            "episode_ids": episode_ids,
        }
