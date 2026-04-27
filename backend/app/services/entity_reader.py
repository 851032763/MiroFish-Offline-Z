"""
实体读取和过滤服务。
从 Neo4j 图谱中读取节点，过滤出有意义的实体类型节点。

替代 zep_entity_reader.py — 所有 Zep Cloud 调用已替换为 GraphStorage。
"""

from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field

from ..utils.logger import get_logger
from ..storage import GraphStorage

logger = get_logger('mirofish.entity_reader')


@dataclass
class EntityNode:
    """实体节点数据结构"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # 相关的边
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # 相关的其他节点
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """获取实体类型（排除默认的 Entity 标签）"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """过滤后的实体集合"""
    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


class EntityReader:
    """
    实体读取和过滤服务（通过 GraphStorage / Neo4j）

    主要功能：
    1. 从图谱中读取所有节点
    2. 过滤出有意义的实体类型节点（标签不仅仅是 "Entity" 的节点）
    3. 获取每个实体的相关边和链接节点信息
    """

    def __init__(self, storage: GraphStorage):
        self.storage = storage

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        从图谱中获取所有节点。

        参数:
            graph_id: 图谱 ID

        返回:
            节点列表。
        """
        logger.info(f"正在获取图谱 {graph_id} 中的所有节点...")
        nodes = self.storage.get_all_nodes(graph_id)
        logger.info(f"共获取 {len(nodes)} 个节点")
        return nodes

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        从图谱中获取所有边。

        参数:
            graph_id: 图谱 ID

        返回:
            边列表。
        """
        logger.info(f"正在获取图谱 {graph_id} 中的所有边...")
        edges = self.storage.get_all_edges(graph_id)
        logger.info(f"共获取 {len(edges)} 条边")
        return edges

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        获取指定节点的所有相关边。

        参数:
            node_uuid: 节点 UUID

        返回:
            边列表。
        """
        try:
            return self.storage.get_node_edges(node_uuid)
        except Exception as e:
            logger.warning(f"获取节点 {node_uuid} 的边失败: {str(e)}")
            return []

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        过滤并提取具有有意义实体类型的节点。

        过滤逻辑：
        - 如果节点的标签仅包含 "Entity"，则它没有有意义的类型，将被跳过。
        - 如果节点的标签包含除了 "Entity" 和 "Node" 之外的其他标签，则它具有有意义的类型，将被保留。

        参数:
            graph_id: 图谱 ID
            defined_entity_types: 要过滤的实体类型列表（可选）。如果提供，则只保留匹配这些类型的实体。
            enrich_with_edges: 是否获取每个实体的相关边信息。

        返回:
            FilteredEntities: 过滤后的实体集合。
        """
        logger.info(f"开始在图谱 {graph_id} 中过滤实体...")

        # 获取所有节点
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        # 获取所有边（用于后续关联查询）
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []

        # 构建从节点 UUID 到节点数据的映射
        node_map = {n["uuid"]: n for n in all_nodes}

        # 过滤符合条件的实体
        filtered_entities = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])

            # 过滤逻辑：标签必须包含除了 "Entity" 和 "Node" 之外的标签
            custom_labels = [la for la in labels if la not in ["Entity", "Node"]]

            if not custom_labels:
                # 只有默认标签，跳过
                continue

            # 如果指定了预定义类型，检查是否匹配
            if defined_entity_types:
                matching_labels = [la for la in custom_labels if la in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            # 创建实体节点对象
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
            )

            # 获取相关边和节点
            if enrich_with_edges:
                related_edges = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge.get("fact", ""),
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges

                # 获取相关的链接节点及其信息
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node.get("labels", []),
                            "summary": related_node.get("summary", ""),
                        })

                entity.related_nodes = related_nodes

            filtered_entities.append(entity)

        logger.info(f"过滤完成: 总节点数 {total_count}, 匹配数 {len(filtered_entities)}, ")
                     f"entity types: {entity_types_found}")

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        获取单个实体及其完整上下文（边和相关节点）。

        优化：使用 get_node() + get_node_edges() 而不是加载所有节点。
        只在需要时单独获取相关节点。

        参数:
            graph_id: 图谱 ID
            entity_uuid: 实体 UUID

        返回:
            EntityNode 或 None。
        """
        try:
            # 直接通过 UUID 获取节点（O(1) 查找）
            node = self.storage.get_node(entity_uuid)
            if not node:
                return None

            # 获取该节点的边（通过 Cypher 为 O(degree)）
            edges = self.storage.get_node_edges(entity_uuid)

            # 处理相关边并收集相关节点 UUID
            related_edges = []
            related_node_uuids: Set[str] = set()

            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge.get("fact", ""),
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            # 单独获取相关节点（避免加载所有节点）
            related_nodes = []
            for related_uuid in related_node_uuids:
                related_node = self.storage.get_node(related_uuid)
                if related_node:
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node.get("labels", []),
                        "summary": related_node.get("summary", ""),
                    })

            return EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=node.get("labels", []),
                summary=node.get("summary", ""),
                attributes=node.get("attributes", {}),
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"获取实体 {entity_uuid} 失败: {str(e)}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        获取指定类型的所有实体。

        参数:
            graph_id: 图谱 ID
            entity_type: 实体类型（例如 "Student", "PublicFigure" 等）
            enrich_with_edges: 是否获取每个实体的相关边信息。

        返回:
            指定类型的实体列表。
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities
