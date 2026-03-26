"""
Intent Parser Node - 意图解析节点

使用 LLM 从用户自然语言查询中提取结构化意图。

功能：
- 提取查询目标（设备名、项目名）
- 解析时间窗口（开始/结束时间）
- 识别数据类型（电量、电流、电压等）
- 默认时间范围为最近 7 天
- 可选的语义层集成（向后兼容）

需求引用：
- 需求 2.1: 使用 LLM 提取实体和时间窗口
- 需求 2.2: 输出结构化 JSON
- 需求 2.3: 默认使用最近 7 天
- 需求 2.5: 解析失败时设置 error 字段
- 需求 5.3: 提供与现有接口兼容的语义层支持
- 需求 5.5: 不影响现有 DAG 节点的执行顺序和数据流
"""

import json
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from src.agent.types import GraphState, NODE_INTENT_PARSER
from src.agent.query_plan import QueryPlan
from src.agent.query_planner import LLMQueryPlanner
from src.agent.query_time_range import build_month_range, resolve_time_range_from_query

if TYPE_CHECKING:
    from src.semantic_layer.semantic_layer import SemanticLayer


logger = logging.getLogger(__name__)


class IntentParserNode:
    """
    意图解析节点 - 使用 LLM 解析用户查询意图
    
    从用户自然语言查询中提取：
    - target: 查询目标（设备名/项目名）
    - time_start: 开始时间 (YYYY-MM-DD)
    - time_end: 结束时间 (YYYY-MM-DD)
    - data_type: 数据类型 (ep/i/u_line/p/qf/t 等)
    - confidence: 数据类型匹配置信度 (0.0-1.0)（仅当使用语义层时）
    
    支持可选的语义层集成：
    - 当提供 semantic_layer 参数时，使用语义匹配增强数据类型识别
    - 当 semantic_layer 为 None 时，保持原有行为（向后兼容）
    
    Attributes:
        llm: LangChain LLM 实例
        semantic_layer: 语义层实例（可选）
        default_days: 默认时间范围天数（默认 7 天）
    """
    
    # 基础系统提示（不含语义上下文）
    SYSTEM_PROMPT = """你是一个意图解析器。从用户查询中提取以下信息：

1. target: 查询目标（设备名、项目名、区域名等）
   - 如果是对比查询，保留完整表达，如 "第九味道和火锅店"、"A跟B"
   - 绝对不要因为有多个目标就返回错误！
2. time_start: 开始时间 (YYYY-MM-DD 格式)
3. time_end: 结束时间 (YYYY-MM-DD 格式)
4. data_type: 数据类型，可选值包括：
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
   - loadrate: 负载率
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
- "最近一个月"/"上个月" → 过去 30 天
- "今天" → 当天
- "昨天" → 前一天
- "本月" → 当月 1 日到今天
- "上个月" → 上月 1 日到上月最后一天
- 如果用户没有指定时间，默认使用最近 7 天

输出格式（仅输出 JSON，不要其他内容）：
{{
    "target": "目标名称（保留原始表达）",
    "time_start": "YYYY-MM-DD",
    "time_end": "YYYY-MM-DD",
    "data_type": "ep",
    "is_comparison": true或false
}}

如果无法从查询中提取有效信息，返回：
{{
    "error": "无法解析的原因"
}}"""

    # 语义增强系统提示模板
    SEMANTIC_SYSTEM_PROMPT = """你是一个意图解析器。从用户查询中提取以下信息：

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
- "最近一个月"/"上个月" → 过去 30 天
- "今天" → 当天
- "昨天" → 前一天
- "本月" → 当月 1 日到今天
- "上个月" → 上月 1 日到上月最后一天
- 如果用户没有指定时间，默认使用最近 7 天

根据语义分析，以下是与用户查询最相关的数据类型候选：

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
- loadrate: 负载率

输出格式（仅输出 JSON，不要其他内容）：
{{
    "target": "目标名称（保留原始表达）",
    "time_start": "YYYY-MM-DD",
    "time_end": "YYYY-MM-DD",
    "data_type": "数据类型标识符",
    "is_comparison": true或false
}}

如果无法从查询中提取有效信息，返回：
{{
    "error": "无法解析的原因"
}}"""

    def __init__(
        self,
        llm: BaseChatModel,
        default_days: int = 7,
        semantic_layer: Optional["SemanticLayer"] = None,
    ):
        """
        初始化意图解析节点
        
        Args:
            llm: LangChain LLM 实例
            default_days: 默认时间范围天数（默认 7 天）
            semantic_layer: 语义层实例（可选，为 None 时保持原有行为）
        """
        self.llm = llm
        self.default_days = default_days
        self.semantic_layer = semantic_layer
        self.query_planner = LLMQueryPlanner(llm)
    
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
        # 去掉 <think>...</think> 标签
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        response_text = response_text.strip()
        
        # 尝试直接解析
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # 尝试提取 JSON 块（支持嵌套）
        # 找到第一个 { 和最后一个 }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
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
    
    def _build_system_prompt(
        self,
        query: str,
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        构建系统提示，可选地包含语义上下文
        
        Args:
            query: 用户查询
            semantic_context: 语义上下文信息（可选）
        
        Returns:
            系统提示字符串
        """
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 如果没有语义上下文或没有候选，使用基础提示
        if not semantic_context or not semantic_context.get("candidates"):
            return self.SYSTEM_PROMPT.format(current_time=current_time)
        
        # 构建候选数据类型列表（限制为 top-k，k≤5）
        candidates = semantic_context.get("candidates", [])[:5]
        
        candidate_lines = []
        for i, c in enumerate(candidates, 1):
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
        
        return self.SEMANTIC_SYSTEM_PROMPT.format(
            current_time=current_time,
            candidates=candidates_text,
        )
    
    def _build_month_range(self, year: int, month: int) -> Optional[Dict[str, str]]:
        return build_month_range(year, month)


    def _resolve_time_range_from_query(self, query_text: str, now: Optional[datetime] = None) -> Optional[Dict[str, str]]:
        return resolve_time_range_from_query(query_text, now=now)

    def _extract_targets_from_plan(self, plan: QueryPlan) -> tuple[str, ...]:
        if plan.search_targets:
            return tuple(plan.search_targets)
        if plan.explicit_device_codes:
            return tuple(plan.explicit_device_codes)
        return ()

    def _build_intent_from_query_plan(
        self,
        user_query: str,
        plan: QueryPlan,
        semantic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        targets = self._extract_targets_from_plan(plan)
        target = " vs ".join(targets) if targets else (next(iter(plan.project_hints), "") or plan.current_question or user_query)

        inferred_data_type = plan.inferred_data_type
        if not inferred_data_type and semantic_context:
            best_match = semantic_context.get("best_match") or {}
            inferred_data_type = best_match.get("type_id")
            if not inferred_data_type:
                candidates = semantic_context.get("candidates") or []
                if candidates:
                    inferred_data_type = candidates[0].get("type_id")

        intent: Dict[str, Any] = {
            "target": str(target or "").strip(),
            "data_type": inferred_data_type or "ep",
            "is_comparison": plan.has_comparison_intent or len(targets) > 1,
            "query_mode": plan.query_mode,
            "response_style": plan.response_style,
            "aggregation": plan.aggregation,
            "ranking_granularity": plan.ranking_granularity,
            "ranking_order": plan.ranking_order,
            "ranking_limit": plan.ranking_limit,
            "project_hints": list(plan.project_hints),
        }

        confidence = max(plan.confidence, float((semantic_context or {}).get("confidence", 0.0) or 0.0))
        if confidence > 0:
            intent["confidence"] = confidence

        if len(targets) > 1:
            intent["comparison_targets"] = list(targets)

        resolved_time_range = None
        if plan.time_start and plan.time_end:
            resolved_time_range = {"start_time": plan.time_start, "end_time": plan.time_end}
        else:
            resolved_time_range = self._resolve_time_range_from_query(plan.current_question or user_query)
        if resolved_time_range:
            intent["time_start"] = resolved_time_range["start_time"]
            intent["time_end"] = resolved_time_range["end_time"]

        return self._apply_defaults(intent)

    def __call__(self, state: GraphState) -> GraphState:
        """
        ???????

        QueryPlan ???????intent ?????? DAG ????????
        """
        user_query = state.get("user_query", "")
        history = list(state.get("history", []))

        if not user_query or not user_query.strip():
            return {
                **state,
                "error": "??????",
                "error_node": NODE_INTENT_PARSER,
                "history": history + [{
                    "node": NODE_INTENT_PARSER,
                    "result": "??: ??????"
                }]
            }

        try:
            semantic_context = None
            if self.semantic_layer:
                semantic_context = self._get_semantic_context(user_query)
                logger.debug("semantic_context=%s", semantic_context)

            plan = self.query_planner.plan(user_query)
            intent = self._build_intent_from_query_plan(user_query, plan, semantic_context)

            is_valid, error_msg = self._validate_intent(intent)
            if not is_valid:
                return {
                    **state,
                    "query_plan": plan.to_dict(),
                    "error": error_msg,
                    "error_node": NODE_INTENT_PARSER,
                    "history": history + [{
                        "node": NODE_INTENT_PARSER,
                        "result": f"????: {error_msg}"
                    }]
                }

            semantic_info = ""
            if semantic_context:
                if semantic_context.get("candidates"):
                    semantic_info = f", semantic_confidence={intent.get('confidence', 1.0):.2f}"
                elif semantic_context.get("fallback_used"):
                    semantic_info = ", semantic_fallback=true"

            history_result = (
                f"QueryPlan={plan.query_mode}, source={plan.source}, target={intent['target']}, "
                f"range={intent['time_start']}~{intent['time_end']}, data_type={intent['data_type']}"
                f"{semantic_info}"
            )

            return {
                **state,
                "query_plan": plan.to_dict(),
                "intent": intent,
                "is_comparison": bool(intent.get("is_comparison", False)),
                "comparison_targets": intent.get("comparison_targets"),
                "history": history + [{
                    "node": NODE_INTENT_PARSER,
                    "result": history_result
                }]
            }

        except Exception as e:
            logger.exception("intent_parser_failed error=%s", e)
            return {
                **state,
                "error": f"??????: {str(e)}",
                "error_node": NODE_INTENT_PARSER,
                "history": history + [{
                    "node": NODE_INTENT_PARSER,
                    "result": f"??: {str(e)}"
                }]
            }
