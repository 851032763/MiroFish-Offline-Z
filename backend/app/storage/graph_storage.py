"""
GraphStorage — 图存储后端的抽象接口。

所有 Zep Cloud 调用都由该抽象替代。
当前实现: Neo4jStorage (neo4j_storage.py)。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable


class GraphStorage(ABC):
    """图存储后端的抽象接口。"""

    # --- 图生命周期 ---

    @abstractmethod
    def create_graph(self, name: str, description: str = "") -> str:
        """创建一个新图。返回 graph_id。"""

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        """删除图及其所有节点/边。"""

    @abstractmethod
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]) -> None:
        """为图存储本体 (实体类型 + 关系类型)。"""

    @abstractmethod
    def get_ontology(self, graph_id: str) -> Dict[str, Any]:
        """检索图的本体。"""

    # --- 添加数据 ---

    @abstractmethod
    def add_text(self, graph_id: str, text: str) -> str:
        """
        处理文本: NER/RE → 创建节点/边 → 返回 episode_id。
        这是同步的 (不像 Zep Cloud 的异步 episodes)。
        """

    @abstractmethod
    def add_text_batch(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """批量添加文本块。返回 episode_id 列表。"""

    @abstractmethod
    def wait_for_processing(
        self,
        episode_ids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ) -> None:
        """
        等待 episodes 处理完成。
        对于 Neo4j: 无操作 (同步处理)。
        为与 Zep 时代调用者的 API 兼容性而保留。
        """

    # --- 读取节点 ---

    @abstractmethod
    def get_all_nodes(self, graph_id: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """获取图中的所有节点 (可选限制)。"""

    @abstractmethod
    def get_node(self, uuid: str) -> Optional[Dict[str, Any]]:
        """通过 UUID 获取单个节点。"""

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """获取连接到节点的所有边 (通过 Cypher O(1) 复杂度, 非全扫描)。"""

    @abstractmethod
    def get_nodes_by_label(self, graph_id: str, label: str) -> List[Dict[str, Any]]:
        """按实体类型标签筛选节点。"""

    # --- 读取边 ---

    @abstractmethod
    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """获取图中的所有边。"""

    # --- 搜索 ---

    @abstractmethod
    def search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ):
        """
        图数据的混合搜索 (向量 + 关键词)。

        Args:
            graph_id: 要搜索的图
            query: 搜索查询文本
            limit: 最大结果数
            scope: "edges"、"nodes" 或 "both"

        Returns:
            包含 'edges' 和/或 'nodes' 列表的字典 (由 GraphToolsService 包装成 SearchResult)
        """

    # --- 图信息 ---

    @abstractmethod
    def get_graph_info(self, graph_id: str) -> Dict[str, Any]:
        """获取图元数据 (节点数、边数、实体类型)。"""

    @abstractmethod
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        获取完整图数据 (为前端提供的增强格式)。

        返回包含以下内容的字典:
            graph_id, nodes, edges, node_count, edge_count
        边字典包含派生字段: fact_type, source_node_name, target_node_name
        """
