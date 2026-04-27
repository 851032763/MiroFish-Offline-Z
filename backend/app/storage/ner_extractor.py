"""
NER/RE Extractor — 通过本地 LLM 进行实体和关系提取

替换 Zep Cloud 内置的 NER/RE 流水线。
使用 LLMClient.chat_json() 和结构化提示词从文本块中
提取实体和关系，由图的本体指导。
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient

logger = logging.getLogger('mirofish.ner_extractor')

# System prompt template for NER/RE extraction
_SYSTEM_PROMPT = """你是一个命名实体识别和关系抽取系统。
根据给定的文本和本体定义（实体类型+关系类型），提取所有实体和关系。

本体定义:
{ontology_description}

规则:
1. 只抽取本体中定义的实体类型和关系类型。
2. 标准化实体名称：去除空格，使用规范形式（例如："张三" 而非 "三张"）。
3. 每个实体必须包含：名称、类型（来自本体）、可选属性。
4. 每个关系必须包含：源实体名称、目标实体名称、类型（来自本体）、描述该关系的事实语句。
5. 如果没有找到实体或关系，返回空列表。
6. 保持精确——只抽取文本中明确陈述或强烈暗示的内容。

只返回以下精确格式的有效JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "fact": "..."}}
  ]
}}"""

_USER_PROMPT = """从以下文本中提取实体和关系:

{text}"""


class NERExtractor:
    """使用本地 LLM 从文本中提取实体和关系。"""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        self.llm = llm_client or LLMClient()
        self.max_retries = max_retries

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        在本体的指导下从文本中提取实体和关系。

        Args:
            text: 输入文本块
            ontology: 来自图的字典，包含 'entity_types' 和 'relation_types'

        Returns:
            包含 'entities' 和 'relations' 列表的字典：
            {
                "entities": [{"name": str, "type": str, "attributes": dict}],
                "relations": [{"source": str, "target": str, "type": str, "fact": str}]
            }
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        system_msg = _SYSTEM_PROMPT.format(ontology_description=ontology_desc)
        user_msg = _USER_PROMPT.format(text=text.strip())

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,  # 低温度以保证提取精度
                    max_tokens=4096,
                )
                return self._validate_and_clean(result, ontology)

            except ValueError as e:
                last_error = e
                logger.warning(
                    f"NER 提取失败（尝试 {attempt + 1}）：无效的 JSON —— {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"NER 提取错误：{e}")
                if attempt >= self.max_retries:
                    break

        logger.error(
            f"NER 提取在 {self.max_retries + 1} 次尝试后失败：{last_error}"
        )
        return {"entities": [], "relations": []}

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """将本体字典格式化为 LLM 提示词可读的文本。"""
        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            parts.append("实体类型：")
            for et in entity_types:
                if isinstance(et, dict):
                    name = et.get("name", str(et))
                    desc = et.get("description", "")
                    attrs = et.get("attributes", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if attrs:
                        attr_names = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in attrs]
                        line += f" (attributes: {', '.join(attr_names)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {et}")

        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        if relation_types:
            parts.append("\n关系类型：")
            for rt in relation_types:
                if isinstance(rt, dict):
                    name = rt.get("name", str(rt))
                    desc = rt.get("description", "")
                    source_targets = rt.get("source_targets", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if source_targets:
                        st_strs = [f"{st.get('source', '?')} → {st.get('target', '?')}" for st in source_targets]
                        line += f" ({', '.join(st_strs)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {rt}")

        if not parts:
            parts.append("未定义特定的本体。提取你发现的所有实体和关系。")

        return "\n".join(parts)

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证和规范化 LLM 输出。"""
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # 从本体中获取有效的类型名称
        valid_entity_types = set()
        for et in ontology.get("entity_types", []):
            if isinstance(et, dict):
                valid_entity_types.add(et.get("name", "").strip())
            else:
                valid_entity_types.add(str(et).strip())

        valid_relation_types = set()
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(rt, dict):
                valid_relation_types.add(rt.get("name", "").strip())
            else:
                valid_relation_types.add(str(rt).strip())

        # 清理实体
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", "Entity")).strip()
            if not name:
                continue

            # 按规范化名称去重
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # 如果本体有类型定义，发出警告但保留未知类型的实体
            if valid_entity_types and etype not in valid_entity_types:
                logger.debug(f"实体 '{name}' 的类型 '{etype}' 不在本体中，仍然保留")

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": entity.get("attributes", {}),
            })

        # 清理关系
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            rtype = str(relation.get("type", "RELATED_TO")).strip()
            fact = str(relation.get("fact", "")).strip()

            if not source or not target:
                continue

            # 确保源和目标实体存在
           # （如果 LLM 幻觉出关系但没有实体，它们可能不存在）
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
