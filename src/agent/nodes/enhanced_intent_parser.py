"""
Enhanced Intent Parser Node - 增强的意图解析节点

集成语义层进行数据类型解析，构建包含语义上下文的增强提示。

功能：
- 使用语义层进行数据类型语义匹配
- 构建包含候选数据类型的增强 LLM 提示
- 在解析结果中包含匹配置信度分数
- 向量检索无结果时回退到精确匹配

需求引用：
- 需求 3.1: 使用向量检索找到最相关的数据类型
- 需求 3.2: 将向量检索结果作为上下文提供给 LLM
- 需求 3.3: 将 top-k 结果（k≤5）提供给 LLM
- 需求 3.4: 在解析结果中包含匹配置信度分数
- 需求 3.5: 向量检索无结果时回退到精确匹配
"""

import json
import re
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.types import GraphState, NODE_INTENT_PARSER

if TYPE_CHECKING:
    from src.semantic_layer.semantic_layer import SemanticLayer


logger = logging.getLogger(__name__)


class EnhancedIntentParserNode:
    """
    增强的意图解析节点 - 集成语义层进行数据类型解析
    
    从用户自然语言查询中提取：
    - target: 查询目标（设备名/项目名）
    - time_start: 开始时间 (YYYY-MM-DD)
    - time_end: 结束时间 (YYYY-MM-DD)
    - data_type: 数据类型 (ep/i/u_line/p/qf/t 等)
    - confidence: 数据类型匹配置信度 (0.0-1.0)
    
    与基础 IntentParserNode 的区别：
    - 使用语义层进行数据类型的语义匹配
    - 在 LLM 提示中包含候选数据类型上下文
    - 解析结果包含置信度分数
    
    Attributes:
        llm: LangChain LLM 实例
        semantic_layer: 语义层实例（可选）
        default_days: 默认时间范围天数（默认 7 天）
    """
    
    # 基础系统提示（不含语义上下文）
    BASE_SYSTEM_PROMPT = """你是一个意图解析器。从用户查询中提取以下信息：

1. target: 查询目标（设备名、项目名、区域名等）
   - 如果是对比查询，保留完整表达，如 "第九味道和火锅店"、"A跟B"
   - 绝对不要因为有多个目标就返回错误！
2. time_start: 开始时间 (YYYY-MM-DD 格式)
3. time_end: 结束时间 (YYYY-MM-DD 格式)
4. data_type: 数据类型标识符
5. is_comparison: 是否为对比查询 (true/false)

当前时间: {current_time}

对比查询识别（设置 is_comparison=true）：
- "对比"、"比较"、"VS"、"vs"、"PK"
- "哪个更"、"谁更"、"哪个高"、"哪个低"
- "A和B"、"A与B"、"A跟B" 配合数据类型查询

示例：
- "第九味道和火锅店的用电量对比" → target="第九味道和火锅店", is_comparison=true
- "办公楼跟干燥室的温度哪个更高" → target="办公楼跟干燥室", is_comparison=true
- "对比一下A和B的电量" → target="A和B", is_comparison=true

【重要】无论查询包含多少个目标，都必须正常解析并返回结果，不要返回错误！

时间解析规则：
- "最近一周"/"上周" → 过去 7 天
- "今天" → 当天
- "昨天" → 前一天
- "本月" → 当月 1 日到今天
- 未指定时间 → 默认最近 7 天

{data_type_context}

输出 JSON（不要其他内容）：
{{
    "target": "目标名称（保留原始表达）",
    "time_start": "YYYY-MM-DD",
    "time_end": "YYYY-MM-DD",
    "data_type": "数据类型标识符",
    "is_comparison": true或false
}}"""

    # 默认数据类型上下文（无语义层时使用）
    DEFAULT_DATA_TYPE_CONTEXT = """数据类型可选值包括：
- ep: 电量/电能/用电量
- i: 电流/三相电流
- u_line: 电压/线电压/三相电压
- p: 功率/有功功率
- qf: 功率因数
- t: 温度
- sd: 湿度
- f: 频率
- soc: 电池容量/荷电状态
- gffddl: 光伏发电量
- loadrate: 负载率"""

    # 语义增强数据类型上下文模板
    SEMANTIC_DATA_TYPE_CONTEXT = """根据语义分析，以下是与用户查询最相关的数据类型候选：

{candidates}

请从上述候选中选择最匹配用户查询意图的数据类型。
如果候选中没有合适的选项，也可以选择其他数据类型：
- ep: 电量/电能
- i: 电流
- u_line: 电压
- p: 功率
- qf: 功率因数
- t: 温度
- sd: 湿度
- f: 频率
- soc: 电池容量
- gffddl: 光伏发电量
- loadrate: 负载率"""

    def __init__(
        self,
        llm: BaseChatModel,
        semantic_layer: Optional["SemanticLayer"] = None,
        default_days: int = 7,
    ):
        """
        初始化增强意图解析节点
        
        Args:
            llm: LangChain LLM 实例
            semantic_layer: 语义层实例（可选，为 None 时回退到基础模式）
            default_days: 默认时间范围天数（默认 7 天）
        """
        self.llm = llm
        self.semantic_layer = semantic_layer
        self.default_days = default_days
    
    def _get_default_time_range(self) -> tuple[str, str]:
        """
        获取默认时间范围（最近 N 天）
        
        Returns:
            (time_start, time_end) 元组，格式为 YYYY-MM-DD
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.default_days)
        return (
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
    
    def _validate_date_format(self, date_str: str) -> bool:
        """
        验证日期格式是否为 YYYY-MM-DD
        
        Args:
            date_str: 日期字符串
        
        Returns:
            是否为有效格式
        """
        if not date_str:
            return False
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def _validate_intent(self, intent: Dict[str, Any]) -> tuple[bool, str]:
        """
        验证解析后的意图是否有效
        
        Args:
            intent: 解析后的意图字典
        
        Returns:
            (is_valid, error_message) 元组
        """
        # 检查是否有错误字段
        if "error" in intent:
            return False, intent["error"]
        
        # 检查必需字段
        required_fields = ["target", "time_start", "time_end", "data_type"]
        for field in required_fields:
            if field not in intent:
                return False, f"缺少必需字段: {field}"
        
        # 验证 target 不为空
        if not intent.get("target") or not str(intent["target"]).strip():
            return False, "查询目标不能为空"
        
        # 验证日期格式
        if not self._validate_date_format(intent.get("time_start", "")):
            return False, f"开始时间格式无效: {intent.get('time_start')}"
        
        if not self._validate_date_format(intent.get("time_end", "")):
            return False, f"结束时间格式无效: {intent.get('time_end')}"
        
        # 验证时间范围
        try:
            start = datetime.strptime(intent["time_start"], "%Y-%m-%d")
            end = datetime.strptime(intent["time_end"], "%Y-%m-%d")
            if start > end:
                return False, f"开始时间 ({intent['time_start']}) 不能晚于结束时间 ({intent['time_end']})"
        except ValueError as e:
            return False, f"日期解析错误: {e}"
        
        return True, ""
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的 JSON 响应
        
        Args:
            response_text: LLM 返回的文本
        
        Returns:
            解析后的字典，如果解析失败返回包含 error 的字典
        """
        # 尝试直接解析
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 块
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # 尝试提取 markdown 代码块中的 JSON
        code_block_pattern = r'```(?:json)?\s*(\{[^`]*\})\s*```'
        code_matches = re.findall(code_block_pattern, response_text, re.DOTALL)
        
        for match in code_matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return {"error": "无法解析 LLM 响应为 JSON"}
    
    def _apply_defaults(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        为意图应用默认值
        
        Args:
            intent: 原始意图字典
        
        Returns:
            应用默认值后的意图字典
        """
        result = dict(intent)
        
        # 应用默认时间范围
        default_start, default_end = self._get_default_time_range()
        
        if not result.get("time_start") or not self._validate_date_format(result.get("time_start", "")):
            result["time_start"] = default_start
        
        if not result.get("time_end") or not self._validate_date_format(result.get("time_end", "")):
            result["time_end"] = default_end
        
        # 应用默认数据类型
        if not result.get("data_type"):
            result["data_type"] = "ep"
        
        return result
    
    def _get_semantic_context(self, query: str) -> Dict[str, Any]:
        """
        获取语义上下文信息
        
        Args:
            query: 用户查询
        
        Returns:
            语义上下文字典，包含 candidates、confidence 等
        """
        if not self.semantic_layer or not self.semantic_layer.is_initialized:
            return {
                "candidates": [],
                "confidence": 0.0,
                "fallback_used": True,
                "best_match": None,
            }
        
        try:
            return self.semantic_layer.get_enhanced_context(query)
        except Exception as e:
            logger.warning(f"获取语义上下文失败: {e}")
            return {
                "candidates": [],
                "confidence": 0.0,
                "fallback_used": True,
                "best_match": None,
            }
    
    def _build_enhanced_prompt(
        self,
        query: str,
        semantic_context: Dict[str, Any],
    ) -> str:
        """
        构建包含语义上下文的增强提示
        
        Args:
            query: 用户查询
            semantic_context: 语义上下文信息
        
        Returns:
            增强的系统提示字符串
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        candidates = semantic_context.get("candidates", [])
        
        if candidates:
            # 构建候选数据类型列表（限制为 top-k，k≤5）
            # Requirements 3.3: top-k 结果（k≤5）
            top_candidates = candidates[:5]
            
            candidate_lines = []
            for i, c in enumerate(top_candidates, 1):
                line = f"{i}. {c['type_id']}: {c['name']}"
                if c.get('synonym'):
                    line += f" (匹配词: {c['synonym']})"
                if c.get('score'):
                    line += f" [相关度: {c['score']:.2f}]"
                if c.get('description'):
                    line += f"\n   描述: {c['description']}"
                if c.get('unit'):
                    line += f"\n   单位: {c['unit']}"
                candidate_lines.append(line)
            
            candidates_text = "\n".join(candidate_lines)
            data_type_context = self.SEMANTIC_DATA_TYPE_CONTEXT.format(
                candidates=candidates_text
            )
        else:
            # 无语义候选时使用默认上下文
            data_type_context = self.DEFAULT_DATA_TYPE_CONTEXT
        
        return self.BASE_SYSTEM_PROMPT.format(
            current_time=current_time,
            data_type_context=data_type_context,
        )
    
    def __call__(self, state: GraphState) -> GraphState:
        """
        执行增强的意图解析
        
        Args:
            state: 当前图状态
        
        Returns:
            更新后的图状态，包含：
            - intent: 解析的意图（含 confidence 字段）
            - error: 错误信息（如有）
            - error_node: 错误节点（如有）
            - history: 更新的执行历史
        """
        user_query = state.get("user_query", "")
        history = list(state.get("history", []))
        
        if not user_query or not user_query.strip():
            return {
                **state,
                "error": "用户查询为空",
                "error_node": NODE_INTENT_PARSER,
                "history": history + [{
                    "node": NODE_INTENT_PARSER,
                    "result": "错误: 用户查询为空"
                }]
            }
        
        try:
            # 获取语义上下文
            semantic_context = self._get_semantic_context(user_query)
            
            logger.debug(f"语义上下文: {semantic_context}")
            
            # 构建增强提示
            system_prompt = self._build_enhanced_prompt(user_query, semantic_context)
            
            # 调用 LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_query)
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.debug(f"LLM 响应: {response_text}")
            
            # 解析响应
            intent = self._parse_llm_response(response_text)
            
            # 检查 LLM 是否返回错误
            if "error" in intent:
                return {
                    **state,
                    "error": intent["error"],
                    "error_node": NODE_INTENT_PARSER,
                    "history": history + [{
                        "node": NODE_INTENT_PARSER,
                        "result": f"解析失败: {intent['error']}"
                    }]
                }
            
            # 应用默认值
            intent = self._apply_defaults(intent)
            
            # 添加置信度分数 (Requirements 3.4)
            # 如果语义层返回了置信度，使用它；否则使用默认值
            if "confidence" not in intent:
                intent["confidence"] = semantic_context.get("confidence", 1.0)
            
            # 验证意图
            is_valid, error_msg = self._validate_intent(intent)
            if not is_valid:
                return {
                    **state,
                    "error": error_msg,
                    "error_node": NODE_INTENT_PARSER,
                    "history": history + [{
                        "node": NODE_INTENT_PARSER,
                        "result": f"验证失败: {error_msg}"
                    }]
                }
            
            # 记录语义层使用情况
            semantic_info = ""
            if semantic_context.get("candidates"):
                semantic_info = f", 语义匹配置信度={intent['confidence']:.2f}"
            elif semantic_context.get("fallback_used"):
                semantic_info = ", 使用回退匹配"
            
            # 成功解析
            return {
                **state,
                "intent": intent,
                "history": history + [{
                    "node": NODE_INTENT_PARSER,
                    "result": f"解析成功: 目标={intent['target']}, 时间={intent['time_start']}~{intent['time_end']}, 类型={intent['data_type']}{semantic_info}"
                }]
            }
            
        except Exception as e:
            logger.exception(f"意图解析异常: {e}")
            return {
                **state,
                "error": f"意图解析异常: {str(e)}",
                "error_node": NODE_INTENT_PARSER,
                "history": history + [{
                    "node": NODE_INTENT_PARSER,
                    "result": f"异常: {str(e)}"
                }]
            }


__all__ = ["EnhancedIntentParserNode"]
