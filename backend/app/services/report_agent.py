"""
Report Agent Service
Generate simulated reports using ReACT pattern (via GraphStorage / Neo4j)

Features:
1. Generate reports based on simulation requirements and graph information
2. First plan the outline structure, then generate section by section
3. Each section uses ReACT multi-round thinking and reflection pattern
4. Support conversations with users, autonomously call retrieval tools during conversations
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .graph_tools import (
    GraphToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent Detailed Logger

    Generates agent_log.jsonl file in the report folder, recording detailed actions at each step.
    Each line is a complete JSON object containing timestamp, action type, details, etc.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize the logger

        Args:
            report_id: Report ID, used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Ensure the log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Get elapsed time from start to now (in seconds)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Log an entry

        Args:
            action: Action type, e.g. 'start', 'tool_call', 'llm_response', 'section_complete', etc
            stage: Current stage, e.g. 'planning', 'generating', 'completed'
            details: Details dictionary, not truncated
            section_title: Current section title (optional)
            section_index: Current section index (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Append to JSONL file
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Log report generation start"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Report generation task started"
            }
        )
    
    def log_planning_start(self):
        """Log outline planning start"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Started planning report outline"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Log context information acquired during planning"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Acquired simulation context information",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Log outline planning completion"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Outline planning completed",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Log section generation start"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Started generating section: {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Log ReACT thinking process"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"ReACT round {iteration} thought"
            }
        )
    
    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Log tool call"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Called tool: {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Log tool call result (full content, not truncated)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Full result, not truncated
                "result_length": len(result),
                "message": f"Tool {tool_name} returned result"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Log LLM response (full content, not truncated)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Full response, not truncated
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"LLM response (tool calls: {has_tool_calls}, final answer: {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Log section content generation completion (records content only, not the whole section completion)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Full content, not truncated
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Section {section_title} content generation completed"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Log section generation completion

        Frontend should listen to this log to determine if a section is truly complete and get full content
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Section {section_title} generation completed"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Log report generation completion"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Report generation completed"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Log error"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Error occurred: {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Report Agent Console Logger

    Writes console-style logs (INFO, WARNING, etc.) to console_log.txt file in the report folder.
    These logs are different from agent_log.jsonl and are plain text console output.
    """
    
    def __init__(self, report_id: str):
        """
        Initialize console logger

        Args:
            report_id: Report ID, used to determine the log file path
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Ensure the log file directory exists"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Set up file handler to write logs to file"""
        import logging

        # Create file handler
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)

        # Use the same concise format as console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)

        # Add to report_agent related loggers
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.graph_tools',
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Avoid duplicate additions
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Close file handler and remove it from logger"""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.graph_tools',
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Ensure file handler is closed during destructor"""
        self.close()


class ReportStatus(str, Enum):
    """Report status"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Report section"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to Markdown format"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Report outline"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convert to Markdown format"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Complete report"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt Template Constants
# ═══════════════════════════════════════════════════════════════

# ── Tool Descriptions ──

TOOL_DESC_INSIGHT_FORGE = """\
[深度洞察检索 - 强大的检索工具]
这是一个强大的检索功能，专为深度分析而设计。它能够:
1. 将您的问题自动分解为多个子问题
2. 从多个维度从模拟知识图谱中检索信息
3. 整合语义搜索、实体分析和关系链追踪的结果
4. 返回最全面、最深入的检索内容

[适用场景]
- 需要深入分析某个主题
- 需要了解事件的多个方面
- 需要获取丰富的材料来支持报告章节

[返回内容]
- 原文中的相关事实（可直接引用）
- 核心实体洞察
- 关系链分析"""

TOOL_DESC_PANORAMA_SEARCH = """\
[广度搜索 - 获取完整概览]
此工具用于获取模拟结果的完整全景视图，特别适合理解事件的演化过程。它将:
1. 检索所有相关的节点和关系
2. 区分当前有效事实和历史/过期事实
3. 帮助您理解事件是如何演化的

[适用场景]
- 需要了解事件的完整发展轨迹
- 需要比较不同阶段的舆情变化
- 需要获取全面的实体和关系信息

[返回内容]
- 当前有效事实（最新模拟结果）
- 历史/过期事实（演化记录）
- 所有涉及的实体"""

TOOL_DESC_QUICK_SEARCH = """\
[简单搜索 - 快速检索]
一个轻量级的快速检索工具，适合简单直接的信息查询。

[适用场景]
- 需要快速查找特定信息
- 需要验证某个事实
- 简单信息检索

[返回内容]
- 与查询最相关的事实列表"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[深度访谈 - 真实Agent访谈（双平台）]
调用OASIS模拟环境的访谈API，对正在运行的模拟Agent进行真实访谈！
这不是LLM模拟，而是调用真实的访谈接口来获取模拟Agent的原初回应。
默认情况下，会同时在Twitter和Reddit两个平台进行访谈，以获取更全面的视角。

功能流程:
1. 自动读取人设文件，了解所有模拟Agent
2. 智能选择与访谈主题最相关的Agent（如学生、媒体、官方等）
3. 自动生成访谈问题
4. 调用/api/simulation/interview/batch接口在双平台进行真实访谈
5. 整合所有访谈结果，提供多视角分析

[适用场景]
- 需要从不同角色角度了解事件观点（学生怎么看？媒体怎么说？官方如何回应？）
- 需要收集多样的意见和立场
- 需要获取模拟Agent的真实回应（来自OASIS模拟环境）
- 希望报告更加生动，包含"访谈记录"

[返回内容]
- 被访谈Agent的身份信息
- 各Agent在Twitter和Reddit平台上的访谈回应
- 关键引述（可直接引用）
- 访谈摘要与视角对比

[重要] 此功能需要OASIS模拟环境正在运行！"""

# ── Outline Planning Prompt ──

PLAN_SYSTEM_PROMPT = """\
你是一位撰写"未来预测报告"的专家，拥有模拟世界的"上帝视角"——你可以洞察模拟中每个Agent的行为、言论和互动。

[核心概念]
我们构建了一个模拟世界，并将特定的"模拟需求"作为变量注入其中。模拟世界的演化结果是对未来可能发生事件的预测。你所观察的不是"实验数据"，而是"未来的预演"。

[你的任务]
撰写一份"未来预测报告"，回答以下问题:
1. 在我们设定的条件下，未来的情况是什么样的？
2. 各Agent（群体）如何反应和行动？
3. 这个模拟揭示了哪些值得关注的未来趋势和风险？

[报告定位]
- ✅ 这是一份基于模拟的未来预测报告，揭示"如果这样会发生，未来的走向"
- ✅ 聚焦于预测结果：事件轨迹、群体反应、涌现现象、潜在风险
- ✅ 模拟世界中Agent的言论和行为是对未来人类行为的预测
- ❌ 不是对现实世界当前状态的分析
- ❌ 不是泛泛的舆情概述

[章节数量限制]
- 最少2个章节，最多5个章节
- 不需要子章节，每个章节直接撰写完整内容
- 内容应简洁，聚焦于核心预测发现
- 章节结构根据预测结果独立设计

请按以下JSON格式输出报告大纲:
{
    "title": "报告标题",
    "summary": "报告摘要（一句话概括核心预测发现）",
    "sections": [
        {
            "title": "章节标题",
            "description": "章节内容描述"
        }
    ]
}

注意：sections数组必须有至少2个、最多5个元素！
重要提示：整个报告大纲（标题、摘要、章节标题和描述）必须使用中文。"""

PLAN_USER_PROMPT_TEMPLATE = """\
[预测场景设定]
注入模拟世界的变量（模拟需求）: {simulation_requirement}

[模拟世界规模]
- 参与模拟的实体数量: {total_nodes}
- 实体之间生成的关系数量: {total_edges}
- 实体类型分布: {entity_types}
- 活跃Agent数量: {total_entities}

[模拟预测的未来事实样例]
{related_facts_json}

请从"上帝视角"审视这场未来预演:
1. 在我们设定的条件下，未来呈现出什么状态？
2. 各群体（Agent）如何反应和行动？
3. 这个模拟揭示了哪些值得关注的未来趋势？

根据预测结果，设计最合适的报告章节结构。

[提醒] 报告章节数量：最少2个，最多5个，内容应简洁，聚焦于核心预测发现。"""

# ── Section Generation Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
你是一位撰写"未来预测报告"的专家，正在撰写报告的一个章节。

报告标题: {report_title}
报告摘要: {report_summary}
预测场景（模拟需求）: {simulation_requirement}

当前待撰写章节: {section_title}

═══════════════════════════════════════════════════════════════
[核心概念]
═══════════════════════════════════════════════════════════════

模拟世界是对未来的预演。我们将特定条件（模拟需求）注入模拟世界。
模拟中Agent的行为和互动是对未来人类行为的预测。

你的任务是:
- 揭示在设定条件下未来会发生什么
- 预测各群体（Agent）如何反应和行动
- 发现值得关注的未来趋势、风险和机遇

❌ 不要将其写成对现实世界当前状态的分析
✅ 聚焦于"未来将如何展开"——模拟结果即预测的未来

═══════════════════════════════════════════════════════════════
[最重要的规则 - 必须遵守]
═══════════════════════════════════════════════════════════════

1. [必须调用工具来观察模拟世界]
   - 你正在从"上帝视角"观察未来的预演
   - 所有内容必须来自模拟世界中的事件和Agent言论/行为
   - 禁止使用自己的知识来撰写报告内容
   - 每个章节必须至少调用3次（最多5次）工具来观察代表未来的模拟世界

2. [必须引用原始Agent言论和行为]
   - Agent的言论和行为是对未来人类行为的预测
   - 在报告中使用引用格式来展示这些预测，例如:
     > "某些群体会说：原文内容..."
   - 这些引述是模拟预测的核心证据

3. [语言一致性 - 必须始终使用中文]
   - 整个报告必须使用中文撰写，不论原材料语言
   - 工具返回的内容可能包含中文、中英混合或其他语言
   - 在引用工具返回的非中文内容时，必须先将其翻译成流畅的中文后再写入报告
   - 翻译时保持原意，确保表达自然
   - 此规则适用于正文和引述内容（>格式）
   - 报告中不要中途切换成其他语言

4. [忠实呈现预测结果]
   - 报告内容必须反映模拟世界中代表未来的模拟结果
   - 不要添加模拟中不存在的信息
   - 如果某些方面的信息不足，如实说明

═══════════════════════════════════════════════════════════════
[⚠️ 格式规范 - 极其重要！]
═══════════════════════════════════════════════════════════════

[一个章节 = 最小内容单元]
- 每个章节是报告的最小内容单元
- ❌ 禁止在章节内使用任何Markdown标题（#、##、###、####等）
- ❌ 禁止在内容开头添加章节标题
- ✅ 章节标题由系统自动添加，只需撰写纯正文
- ✅ 使用**粗体**、段落分隔、引用和列表来组织内容，但不要使用标题

[正确示例]
```
本节分析监管转变如何重塑企业战略。通过深入分析模拟数据，我们发现...

**行业初步反应**

主要科技公司迅速重新评估其合规姿态：

> "OpenAI和Anthropic正在努力满足新的透明度要求..."

**新兴战略分化**

在拥抱监管和抵制监管的公司之间出现了明显分歧：

- 将主动合规作为竞争优势
- 开展游说以软化执行力度
```

[错误示例]
```
## 执行摘要          ← 错误！不要添加任何标题
### 1. 初始阶段         ← 错误！不要使用###作为子节
#### 1.1 详细分析   ← 错误！不要使用####作为细分

本节分析...
```

═══════════════════════════════════════════════════════════════
[可用检索工具]（每个章节调用3-5次）
═══════════════════════════════════════════════════════════════

{tools_description}

[工具使用建议 - 请混合使用不同工具，不要只使用一种]
- insight_forge: 深度洞察分析，自动分解问题，从多维度检索事实和关系
- panorama_search: 广角全景搜索，了解事件的完整视图、时间线和演化过程
- quick_search: 快速验证特定信息点
- interview_agents: 访谈模拟Agent，获取不同角色的第一人称视角和真实反应

═══════════════════════════════════════════════════════════════
[工作流程]
═══════════════════════════════════════════════════════════════

你的每次回复只能做以下两件事之一（不能同时做）:

选项A - 调用工具:
先输出你的思考，然后使用以下格式调用工具:
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名称": "参数值"}}}}
</tool_call>
系统将执行工具并将结果返回给你。你不需要也不能自己编写工具返回结果。

选项B - 输出最终内容:
当你通过工具收集到足够信息后，以"最终答案:"开头输出章节内容。

⚠️ 严格禁止:
- 禁止在一次回复中同时包含工具调用和最终答案
- 禁止捏造工具返回结果（观察结果），所有工具结果由系统注入
- 每次回复最多一个工具调用

═══════════════════════════════════════════════════════════════
[章节内容要求]
═══════════════════════════════════════════════════════════════

1. 内容必须基于工具检索的模拟数据
2. 大量引用原文来展示模拟效果
3. 使用Markdown格式（但禁止使用标题）:
   - 使用**粗体文字**标记重点（代替子节标题）
   - 使用列表（- 或 1.2.3.）来组织要点
   - 使用空行分隔段落
   - ❌ 禁止使用任何标题语法如 #、##、###、####
4. [引述格式规范 - 必须单独成段]
   引述必须是独立段落，前后有空行，不能混入段落中:

   ✅ 正确格式:
   ```
   学校方面的回应被认为缺乏实质性内容。

   > "学校方面的回应模式在快速变化的社交媒体环境中显得僵化和迟缓。"

   这一评估反映了公众的普遍不满。
   ```

   ❌ 错误格式:
   ```
   学校方面的回应被认为缺乏实质性内容。> "学校方面的回应模式..." 这一评估反映了...
   ```
5. 与其他章节保持逻辑连贯
6. [避免重复] 仔细阅读下面已完成的章节内容，不要重复描述相同信息
7. [再次强调] 不要添加任何标题！使用**粗体**代替章节子标题"""

SECTION_USER_PROMPT_TEMPLATE = """\
已完成章节内容（请仔细阅读以避免重复）:
{previous_content}

═══════════════════════════════════════════════════════════════
[当前任务] 撰写章节: {section_title}
═══════════════════════════════════════════════════════════════

[重要提醒]
1. 仔细阅读上面已完成的章节，避免重复描述相同内容！
2. 在开始之前必须调用工具获取模拟数据
3. 请混合使用不同工具，不要只使用一种
4. 报告内容必须来自检索结果，不要使用自己的知识

[⚠️ 格式警告 - 必须遵守]
- ❌ 不要写任何标题（#、##、###、####都不允许）
- ❌ 不要以"{section_title}"作为开头
- ✅ 章节标题由系统自动添加
- ✅ 直接撰写正文，使用**粗体**代替子节标题

请开始:
1. 先思考（Thought）本章节需要什么信息
2. 然后调用工具（Action）获取模拟数据
3. 收集足够信息后，输出最终答案（纯正文，无标题）"""

# ── ReACT Loop Message Templates ──

REACT_OBSERVATION_TEMPLATE = """\
观察结果（检索结果）:

═══ 工具 {tool_name} 返回 ═══
{result}

═══════════════════════════════════════════════════════════════
已调用工具 {tool_calls_count}/{max_tool_calls} 次（已使用: {used_tools_str}）{unused_hint}
- 如果信息足够：以"最终答案:"开头输出章节内容（必须引用上述原文）
- 如果需要更多信息：调用工具继续检索
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[通知] 你只调用了 {tool_calls_count} 次工具，需要至少 {min_tool_calls} 次。 "
    "请再次调用工具获取更多模拟数据，然后输出最终答案。{unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "目前已调用 {tool_calls_count} 次工具，需要至少 {min_tool_calls} 次。 "
    "请调用工具获取模拟数据。{unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "工具调用次数已达上限（{tool_calls_count}/{max_tool_calls}），不能再调用工具了。 "
    '请立即以"最终答案:"开头，基于已获取的信息输出章节内容。'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 你还没有使用: {unused_list}，建议尝试不同的工具获取多视角信息"

REACT_FORCE_FINAL_MSG = "工具调用已达上限，请直接输出最终答案:并生成章节内容。"

# ── Chat Prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
你是一个简洁高效的模拟预测助手。

[背景]
预测条件: {simulation_requirement}

[生成的预测分析报告]
{report_content}

[规则]
1. 优先基于上述报告内容回答问题
2. 直接回答问题，避免冗长的讨论
3. 只有当报告内容不足以回答时，才调用工具检索更多数据
4. 回答应简洁、清晰、有条理

[可用工具]（仅在需要时使用，最多调用1-2次）
{tools_description}

[工具调用格式]
<tool_call>
{{"name": "工具名称", "parameters": {{"参数名称": "参数值"}}}}
</tool_call>

[回答风格]
- 简洁直接，不要写长篇大论
- 使用 > 格式引用关键内容
- 先给出结论，再解释原因
- 必须使用中文回答，不论原材料或报告内容使用何种语言"""

CHAT_OBSERVATION_SUFFIX = "\n\n请简洁回答问题。"


# ═══════════════════════════════════════════════════════════════
# ReportAgent Main Class
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent - Simulation Report Generation Agent

    Uses ReACT (Reasoning + Acting) pattern:
    1. Planning Phase: Analyze simulation requirements, plan report outline structure
    2. Generation Phase: Generate content section by section, each section can call tools multiple times to get information
    3. Reflection Phase: Check content completeness and accuracy
    """
    
    # Maximum tool call count (per section)
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Maximum reflection rounds
    MAX_REFLECTION_ROUNDS = 3

    # Maximum tool call count in conversation
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        graph_tools: Optional[GraphToolsService] = None
    ):
        """
        Initialize Report Agent

        Args:
            graph_id: Graph ID
            simulation_id: Simulation ID
            simulation_requirement: Simulation requirement description
            llm_client: LLM client (optional)
            graph_tools: Graph tools service (optional, requires external GraphStorage injection)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        if graph_tools is None:
            raise ValueError(
                "graph_tools (GraphToolsService) is required. "
                "Create it via GraphToolsService(storage=...) and pass it in."
            )
        self.graph_tools = graph_tools
        
        # Tool definitions
        self.tools = self._define_tools()

        # Logger (initialized in generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Console logger (initialized in generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(f"ReportAgent initialization complete: graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Define available tools"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "你想要深入分析的问题或主题",
                    "report_context": "当前报告章节的上下文（可选，有助于生成更准确的子问题）"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "搜索查询，用于相关性排序",
                    "include_expired": "是否包含过期/历史内容（默认True）"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "搜索查询字符串",
                    "limit": "返回结果数量（可选，默认10）"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "访谈主题或需求描述（例如：'了解学生对宿舍甲醛事件的看法'）",
                    "max_agents": "最大访谈Agent数量（可选，默认5，最大10）"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Execute tool call

        Args:
            tool_name: Tool name
            parameters: Tool parameters
            report_context: Report context (for InsightForge)

        Returns:
            Tool execution result (text format)
        """
        logger.info(f"Executing tool: {tool_name}, parameters: {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.graph_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Breadth search - get complete panorama
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.graph_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Simple search - quick retrieval
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.graph_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Deep interview - call real OASIS interview API to get simulated agent responses (dual platform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.graph_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Backward Compatibility: Old Tools (Internal Redirect to New Tools) ==========

            elif tool_name == "search_graph":
                # Redirect to quick_search
                logger.info("search_graph has been redirected to quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.graph_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.graph_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirect to insight_forge because it's more powerful
                logger.info("get_simulation_context has been redirected to insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.graph_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unknown tool: {tool_name}. Please use one of the following tools: insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {str(e)}")
            return f"Tool execution failed: {str(e)}"
    
    # Valid tool names set, used for validation when parsing raw JSON fallback
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse tool calls from LLM response

        Supported formats (in priority order):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Raw JSON (the entire response or a single line is a tool call JSON)
        """
        tool_calls = []

        # Format 1: XML-style (standard format)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback - LLM directly outputs raw JSON (not wrapped in <tool_call> tags)
        # Only try if format 1 didn't match to avoid mismatching JSON in body text
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Response may contain thinking text + raw JSON, try to extract the last JSON object
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Validate if the parsed JSON is a valid tool call"""
        # Support both {"name": ..., "parameters": ...} and {"tool": ..., "params": ...} key names
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normalize key names to name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Generate tool description text"""
        desc_parts = ["Available Tools:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameters: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self,
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Plan report outline

        Use LLM to analyze simulation requirements and plan the report structure

        Args:
            progress_callback: Progress callback function

        Returns:
            ReportOutline: Report outline
        """
        logger.info("Starting to plan report outline...")

        if progress_callback:
            progress_callback("planning", 0, "Analyzing simulation requirements...")

        # First get simulation context
        context = self.graph_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("planning", 30, "Generating report outline...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Parsing outline structure...")

            # Parse outline
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulation Analysis Report"),
                summary=response.get("summary", ""),
                sections=sections
            )

            if progress_callback:
                progress_callback("planning", 100, "Outline planning completed")

            logger.info(f"Outline planning completed: {len(sections)} sections")
            return outline

        except Exception as e:
            logger.error(f"Outline planning failed: {str(e)}")
            # Return default outline (3 sections as fallback)
            return ReportOutline(
                title="Future Prediction Report",
                summary="Future trends and risk analysis based on simulation predictions",
                sections=[
                    ReportSection(title="Prediction Scenario and Core Findings"),
                    ReportSection(title="Crowd Behavior Prediction Analysis"),
                    ReportSection(title="Trend Outlook and Risk Warning")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Generate individual section content using ReACT pattern

        ReACT loop:
        1. Thought - Analyze what information is needed
        2. Action - Call tool to get information
        3. Observation - Analyze tool return results
        4. Repeat until information is sufficient or maximum iterations reached
        5. Final Answer - Generate section content

        Args:
            section: Section to generate
            outline: Complete outline
            previous_sections: Content of previous sections (for maintaining coherence)
            progress_callback: Progress callback
            section_index: Section index (for logging)

        Returns:
            Section content (Markdown format)
        """
        logger.info(f"ReACT generating section: {section.title}")
        
        # Log section start
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Build user prompt - pass maximum 4000 characters for each completed section
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Maximum 4000 characters per section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(This is the first section)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT loop
        tool_calls_count = 0
        max_iterations = 5  # Maximum iterations
        min_tool_calls = 3  # Minimum tool calls
        conflict_retries = 0  # Consecutive conflicts where tool calls and Final Answer appear simultaneously
        used_tools = set()  # Record tool names already called
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Report context for InsightForge sub-question generation
        report_context = f"Section Title: {section.title}\nSimulation Requirement: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Deep retrieval and writing in progress ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Call LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Check if LLM return is None (API exception or empty content)
            if response is None:
                logger.warning(f"Section {section.title} round {iteration + 1} iteration: LLM returned None")
                # If there are more iterations, add message and retry
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Response empty)"})
                    messages.append({"role": "user", "content": "请继续生成内容。"})
                    continue
                # Last iteration also returned None, exit loop and enter forced conclusion
                break

            logger.debug(f"LLM response: {response[:200]}...")

            # Parse once, reuse result
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "最终答案:" in response

            # ── Conflict handling: LLM simultaneously output tool calls and Final Answer ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} round {iteration+1} : "
                    f"LLM simultaneously output tool calls and Final Answer (round {conflict_retries} conflicts)"
                )

                if conflict_retries <= 2:
                    # First two times: discard this response and request LLM to reply again
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[格式错误] 你不能在一次回复中同时包含工具调用和最终答案。\n"
                            "每次回复只能做以下其中一件事：\n"
                            "- 调用工具（输出一个<tool_call>块，不要写最终答案）\n"
                            "- 输出最终内容（以'最终答案:'开头，不要包含<tool_call>）\n"
                            "请重新回复，只做其中一件事。"
                        ),
                    })
                    continue
                else:
                    # Third time: downgrade, truncate to first tool call, force execution
                    logger.warning(
                        f"Section {section.title}: consecutive {conflict_retries} conflicts，"
                        "downgraded to truncate and execute first tool call"
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Log LLM response
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Case 1: LLM output Final Answer ──
            if has_final_answer:
                # Insufficient tool calls, reject and request to continue calling tools
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(These tools have not been used, recommend using them: {', '.join(unused_tools)}）" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Normal completion
                final_answer = response.split("最终答案:")[-1].strip()
                logger.info(f"Section {section.title} generation completed (tool calls: {tool_calls_count}times)")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Case 2: LLM attempts to call tools ──
            if has_tool_calls:
                # Tool quota exhausted → inform clearly, request output Final Answer
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Only execute the first tool call
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"LLM attempted to call {len(tool_calls)} tools, only execute the first: {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Build unused tools hint
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Case 3: NeitherTool call，nor Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Tool callcount insufficient，recommend unused tools
                unused_tools = all_tools - used_tools
                unused_hint = f"(These tools have not been used, recommend using them: {', '.join(unused_tools)}）" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # 直接采用此内容作为最终答案，不再等待
            logger.info(f"章节 {section.title} 未检测到'最终答案:'前缀，直接采用LLM输出作为最终内容（工具调用: {tool_calls_count}次）")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # 已达到最大迭代次数，强制生成内容
        logger.warning(f"章节 {section.title} 已达到最大迭代次数，强制生成")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # 检查强制结论时LLM返回是否为None
        if response is None:
            final_answer = f"（本章节生成失败：LLM返回空响应，请稍后重试）"
        elif "最终答案:" in response:
            final_answer = response.split("最终答案:")[-1].strip()
        else:
            final_answer = response
        
        # 记录章节内容生成完成日志
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        生成完整报告（每章节实时输出）
        
        文件结构:
        reports/{report_id}/
            outline.json    - 报告大纲
            progress.json   - 生成进度
            section_01.md   - 第1章
            section_02.md   - 第2章
            ...
            full_report.md  - 完整报告
        
        Args:
            report_id: 报告ID（可选，如不提供则自动生成）
            
        Returns:
            Report: 完整报告
        """
        import uuid
        
        # 如未提供report_id，则自动生成
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # 已完成章节标题列表（用于进度跟踪）
        completed_section_titles = []
        
        try:
            # 初始化：创建报告文件夹并保存初始状态
            ReportManager._ensure_report_folder(report_id)
            
            # 初始化日志记录器（结构化日志 agent_log.jsonl）
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # 初始化控制台日志记录器（console_log.txt）
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "初始化报告...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # 阶段1：规划大纲
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "开始规划报告大纲...",
                completed_sections=[]
            )
            
            # Log outline planning start
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Start planning report outline...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # 记录大纲规划完成日志
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # 保存大纲到文件
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"大纲规划完成，共{len(outline.sections)}章节",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"大纲已保存到文件: {report_id}/outline.json")
            
            # 阶段2：逐章生成（每章保存）
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # 保存内容以供上下文使用
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # 更新进度
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"正在生成章节: {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"正在生成章节: {section.title} ({section_num}/{total_sections})"
                    )
                
                # 生成主章节内容
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # 保存章节
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # 记录章节完成日志
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"章节已保存: {report_id}/section_{section_num:02d}.md")
                
                # 更新进度
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"章节 {section.title} 完成",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # 阶段3：组装完整报告
            if progress_callback:
                progress_callback("generating", 95, "正在组装完整报告...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "正在组装完整报告...",
                completed_sections=completed_section_titles
            )
            
            # 使用ReportManager组装完整报告
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # 计算总耗时
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # 记录报告完成日志
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # 保存最终报告
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "报告生成完成",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "报告生成完成")
            
            logger.info(f"报告生成完成: {report_id}")
            
            # 关闭控制台日志记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # 记录错误日志
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # 保存失败状态
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"报告生成失败: {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # 忽略保存失败错误
            
            # 关闭控制台日志记录器
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        与Report Agent聊天
        
        在聊天中Agent可以自主调用检索工具来回答问题
        
        Args:
            message: 用户消息
            chat_history: 聊天历史
            
        Returns:
            {
                "response": "Agent响应",
                "tool_calls": [调用工具列表],
                "sources": [信息来源]
            }
        """
        logger.info(f"Report Agent聊天: {message[:50]}...")
        
        chat_history = chat_history or []
        
        # 获取已生成的报告内容
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # 限制报告长度，避免过长的上下文
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [报告内容已截断] ..."
        except Exception as e:
            logger.warning(f"获取报告内容失败: {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "（nonereport）",
            tools_description=self._get_tools_description(),
        )

        # 构建消息
        messages = [{"role": "system", "content": system_prompt}]
        
        # 添加历史聊天
        for h in chat_history[-10:]:  # 限制历史长度
            messages.append(h)
        
        # 添加用户消息
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT循环（简化版）
        tool_calls_made = []
        max_iterations = 2  # 减少迭代次数
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # parseTool call
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # noTool call，directlyReturnresponse
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Execute toolcall（limitcount）
            tool_results = []
            for call in tool_calls[:1]:  # at mostExecute1 time tool call
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # limitresultlength
                })
                tool_calls_made.append(call)
            
            # convertresultadd to message
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']}result]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Reachedmaximum iteration，Getfinalresponse
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # cleanresponse
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    ReportManagemanager
    
    responsible forReportpersistence storage and retrieval
    
    filestructure（perSectionoutput）：
    reports/
      {report_id}/
        meta.json          - Reportmetainformationand status
        outline.json       - Reportoutline
        progress.json      - generateProgress
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Complete report
    """
    
    # Reportstorage directory
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """ensurereportroot directory exists"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """getreportfolderpath"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """ensurereportfolderexists andreturnedpath"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """getreportmetainformationfile path"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """getcompletereportMarkdownfile path"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """getoutlinefile path"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """getprogressfile path"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """getSectionMarkdownfile path"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """get Agent logsfile path"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """getconsolelogsfile path"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Getconsolelogcontent
        
        This isReportgenerateduring processconsoleoutputlog（INFO、WARNINGetc），
        and agent_log.jsonl structured logsdifferent。
        
        Args:
            report_id: ReportID
            from_line: from which rowrowStartRead（for incrementalGet，0 means from the beginningStart）
            
        Returns:
            {
                "logs": [logrowlist],
                "total_lines": totalrownumber,
                "from_line": startrownumber,
                "has_more": whether there are morelog
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # keeporiginallogrow，remove trailingrowcharacter
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # alreadyReadto the end
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        GetCompleteconsolelog（one-timeGetall）
        
        Args:
            report_id: ReportID
            
        Returns:
            logrowlist
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Get Agent logcontent
        
        Args:
            report_id: ReportID
            from_line: from which rowrowStartRead（for incrementalGet，0 means from the beginningStart）
            
        Returns:
            {
                "logs": [logentrylist],
                "total_lines": totalrownumber,
                "from_line": startrownumber,
                "has_more": whether there are morelog
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # skip parsingfailedrow
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # alreadyReadto the end
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        GetComplete Agent log（for one-timeGetall）
        
        Args:
            report_id: ReportID
            
        Returns:
            logentrylist
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        saveReportoutline
        
        in planningphasecompleteimmediately aftercall
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"outlinesaved: {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        savesinglesections

        inEach sectiongeneration completed afterimmediatelycall，implementperSectionoutput

        Args:
            report_id: ReportID
            section_index: Sectionindex（from1Start）
            section: Sectionobject

        Returns:
            savefile path
        """
        cls._ensure_report_folder(report_id)

        # BuildSectionMarkdowncontent - clean possibleduplicatetitle
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # savefile
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Sectionsaved: {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        cleanSectioncontent
        
        1. removecontentbeginningandSection TitleduplicateMarkdowntitlerow
        2. convertall ### and below levelstitleconvert toboldtext
        
        Args:
            content: originalcontent
            section_title: Section Title
            
        Returns:
            after cleaningcontent
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Checkwhether isMarkdowntitlerow
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Checkwhether isandSection Titleduplicatetitle（skip first5rowwithinduplicate）
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # convertallleveltitle（#, ##, ###, ####etc）convert tobold
                # becauseSection Titleadded by system，contentshould not have anytitle
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # addempty line
                continue
            
            # if previousrowwas skippedtitle，and currentrowempty，also skip
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # removebeginningempty line
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # removebeginningseparatorline
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # meanwhileremoveseparatorline afterempty line
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        UpdateReportgenerateProgress
        
        frontend can getReadprogress.jsonGetrealtimeProgress
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """getreportgenerateprogress"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        GetalreadygenerateSectionlist
        
        ReturnallalreadysaveSectionfileinformation
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # fromfilename parsingSectionindex
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        assembleComplete report
        
        fromsaveSectionfileassembleComplete report，and processrowtitleclean
        """
        folder = cls._get_report_folder(report_id)
        
        # BuildReportheader
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # sequentiallyReadallSectionfile
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # post-processing：clean entireReporttitlequestion
        md_content = cls._post_process_report(md_content, outline)
        
        # saveComplete report
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"completereporthasassemble: {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        post-processingReportcontent
        
        1. removeduplicatetitle
        2. keepReportmain title(#)andSection Title(##)，removeother levelstitle(###, ####etc)
        3. clean redundantempty lineandseparatorline
        
        Args:
            content: originalReportcontent
            outline: Reportoutline
            
        Returns:
            after processingcontent
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # collectoutlineinallSection Title
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Checkwhether istitlerow
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Checkwhether isduplicatetitle（inconsecutive5rowappear the same withincontenttitle）
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # skipduplicatetitleand subsequentempty line
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # titlelevel handling：
                # - # (level=1) onlykeepReportmain title
                # - ## (level=2) keepSection Title
                # - ### and below (level>=3) convert toboldtext
                
                if level == 1:
                    if title == outline.title:
                        # keepReportmain title
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Section Titleerrorusing#，corrected to##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # other first-leveltitleconvert tobold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # keepSection Title
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # nonSectionsecond-leveltitleconvert tobold
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### and below levelstitleconvert toboldtext
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # skiptitlefollowed immediately byseparatorline
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # titleafter onlykeeponeempty line
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # cleanconsecutivemultipleempty line（keepat most2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """SavereportmetainformationandcompleteReport"""
        cls._ensure_report_folder(report.report_id)
        
        # savemetainformationJSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # saveoutline
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # saveCompleteMarkdownReport
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"reportsaved: {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """getreport"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # backward compatibleformat：Checkdirectlystored inreportsunder directoryfile
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # rebuildReportobject
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # ifmarkdown_contentempty，attempt tofromfull_report.mdRead
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """based onsimulationIDgetreport"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # newformat：filefolder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # backward compatibleformat：JSONfile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """columnappearreport"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # newformat：filefolder
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # backward compatibleformat：JSONfile
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # sorted by creation time descending
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Deletereport（entirefolder）"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # newformat：Deleteentirefilefolder
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"reportfolderhasDelete: {report_id}")
            return True
        
        # backward compatibleformat：Deleteseparatefile
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
