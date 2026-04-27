"""
Ontology generation service
Interface 1: Analyze text content and generate entity and relationship type definitions suitable for social simulation
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# System prompt for ontology generation
ONTOLOGY_SYSTEM_PROMPT = """你是一位专业的知识图谱本体设计专家。你的任务是分析给定的文本内容和模拟需求，设计适合**社交媒体舆论模拟**的实体类型和关系类型。

**重要：你必须输出有效的JSON格式数据，不要输出其他内容。**

## 核心任务背景

我们正在构建一个**社交媒体舆论模拟系统**。在这个系统中：
- 每个实体都是一个可以在社交媒体上发声、互动和传播信息的"账号"或"主体"
- 实体之间相互影响、转发、评论和回应
- 我们需要模拟舆论事件中各方的反应和信息传播路径

因此，**实体必须是能够在社交媒体上发声和互动的现实世界实体**：

**可以是**：
- 特定个人（公众人物、利益相关者、意见领袖、专家、普通人）
- 企业和公司（包括其官方账号）
- 组织机构（大学、协会、非政府组织、工会等）
- 政府部门和监管机构
- 媒体机构（报社、电视台、自媒体、网站）
- 社交媒体平台本身
- 特定群体代表（如校友会粉丝群、维权群体等）

**不能是**：
- 抽象概念（如"舆论"、"情绪"、"趋势"）
- 话题/主题（如"学术诚信"、"教育改革"）
- 观点/态度（如"支持者"、"反对者"）

## 输出格式

请按以下结构输出JSON格式：

```json
{
    "entity_types": [
        {
            "name": "实体类型名称（英文，PascalCase）",
            "description": "简短描述（英文，不超过100个字符）",
            "attributes": [
                {
                    "name": "属性名称（英文，snake_case）",
                    "type": "text",
                    "description": "属性描述"
                }
            ],
            "examples": ["示例实体1", "示例实体2"]
        }
    ],
    "edge_types": [
        {
            "name": "关系类型名称（英文，UPPER_SNAKE_CASE）",
            "description": "简短描述（英文，不超过100个字符）",
            "source_targets": [
                {"source": "源实体类型", "target": "目标实体类型"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "对文本内容的简要分析说明"
}
```

## 设计指南（极其重要！）

### 1. 实体类型设计 - 必须严格遵循

**数量要求：必须有恰好10个实体类型**

**层次结构要求（必须包含具体类型和回退类型）**：

你的10个实体类型必须包含以下层次：

A. **回退类型（必须包含，放在列表最后2个）**：
   - `Person`：任何自然人的回退类型。当一个人不适合其他更具体的个人类型时，使用这个。
   - `Organization`：任何组织的回退类型。当一个组织不适合其他更具体的组织类型时，使用这个。

B. **具体类型（8个，根据文本内容设计）**：
   - 为文本中出现的主要角色设计更具体的类型
   - 示例：如果文本涉及学术事件，可以有`Student`、`Professor`、`University`
   - 示例：如果文本涉及商业事件，可以有`Company`、`CEO`、`Employee`

**为什么需要回退类型**：
- 文本中会出现各种人，如"中小学教师"、"随机网民"、"某网友"
- 如果没有更具体的类型匹配，应归类为`Person`
- 类似地，小型组织和临时团体应归类为`Organization`

**具体类型设计原则**：
- 从文本中识别高频或关键角色类型
- 每个具体类型应有清晰的边界，避免重叠
- 描述必须清楚说明此类型与回退类型的区别

### 2. 关系类型设计

- 数量：6-10个
- 关系应反映社交媒体互动中的真实联系
- 确保关系source_targets覆盖你定义的所有实体类型

### 3. 属性设计

- 每个实体类型1-3个关键属性
- **注意**：属性名不能使用`name`、`uuid`、`group_id`、`created_at`、`summary`（这些是系统保留字）
- 推荐使用：`full_name`、`title`、`role`、`position`、`location`、`description`等

## 实体类型参考

**个人类型（具体）**：
- Student：学生
- Professor：教授/学者
- Journalist：记者
- Celebrity：名人/网红
- Executive：企业高管
- Official：政府官员
- Lawyer：律师
- Doctor：医生

**个人类型（回退）**：
- Person：任何自然人（当不适合其他具体类型时使用）

**组织类型（具体）**：
- University：大学
- Company：公司/企业
- GovernmentAgency：政府机构
- MediaOutlet：媒体机构
- Hospital：医院
- School：中小学
- NGO：非政府组织

**组织类型（回退）**：
- Organization：任何组织（当不适合其他具体类型时使用）

## 关系类型参考

- WORKS_FOR：为...工作
- STUDIES_AT：在...学习
- AFFILIATED_WITH：隶属于
- REPRESENTS：代表
- REGULATES：监管
- REPORTS_ON：报道
- COMMENTS_ON：评论
- RESPONDS_TO：回应
- SUPPORTS：支持
- OPPOSES：反对
- COLLABORATES_WITH：与...合作
- COMPETES_WITH：与...竞争
"""


class OntologyGenerator:
    """
    Ontology generator
    Analyze text content and generate entity and relationship type definitions
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate ontology definition

        Args:
            document_texts: List of document texts
            simulation_requirement: Description of simulation requirements
            additional_context: Additional context

        Returns:
            Ontology definition (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Call LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Validate and post-process
        result = self._validate_and_process(result)

        return result

    # Maximum text length for LLM (50,000 characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Build user message"""

        # Combine texts
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # If text exceeds 50,000 characters, truncate (only affects LLM input, not graph construction)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...（原始文本共{original_length}个字符，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}个字符用于本体分析）..."

        message = f"""## 模拟需求

{simulation_requirement}

## 文档内容

{combined_text}
"""

        if additional_context:
            message += f"""
## 补充说明

{additional_context}
"""

        message += """
基于以上内容，设计适合社会舆论模拟的实体类型和关系类型。

**必须遵循的规则**：
1. 必须输出恰好10个实体类型
2. 最后2个必须是回退类型：Person（个人回退）和 Organization（组织回退）
3. 前8个是基于文本内容设计的具体类型
4. 所有实体类型必须是能够发声的真实世界主体，不能是抽象概念
5. 属性名不能使用保留字如name、uuid、group_id，应使用full_name、org_name等代替
"""

        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and post-process result"""

        # Ensure necessary fields exist
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Validate entity types
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Ensure description doesn't exceed 100 characters
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Validate relationship types
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Zep API limit: maximum 10 custom entity types, maximum 10 custom edge types
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Fallback type definitions
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }

        # Check if fallback types already exist
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Fallback types to add
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # If adding would exceed 10, need to remove some existing types
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculate how many to remove
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Remove from end (keep more important specific types in front)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Add fallback types
            result["entity_types"].extend(fallbacks_to_add)

        # Final check to ensure limits not exceeded (defensive programming)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        [DEPRECATED] Convert ontology definition to Zep-format Pydantic code.
        Not used in MiroFish-Offline (ontology stored as JSON in Neo4j).
        Kept for reference only.
        """
        code_lines = [
            '"""',
            'Custom entity type definitions',
            'Auto-generated by MiroFish for social opinion simulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entity Type Definitions ==============',
            '',
        ]

        # Generate entity types
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== Relationship Type Definitions ==============')
        code_lines.append('')

        # Generate relationship types
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        # Generate type dictionaries
        code_lines.append('# ============== Type Configuration ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        # Generate source_targets mapping for edges
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)

