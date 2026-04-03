"""
Agent Orchestrator - LLM 驱动的智能数据查询代理

采用 ReAct (Reasoning + Acting) 模式：
1. LLM 分析用户问题，决定下一步行动
2. 执行工具调用（搜索设备、获取数据等）
3. LLM 观察结果，决定是否需要继续迭代
4. 生成最终响应

特点：
- 无硬编码逻辑，完全由 LLM 驱动
- 支持查询改写，提高搜索准确度
- 支持迭代搜索，自动调整策略
"""

from typing import TypedDict, List, Dict, Optional, Any, Generator
from dataclasses import dataclass
import logging
import json
import re
import time
from datetime import datetime, timedelta

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.tools.device_tool import find_device_metadata_with_engine
from src.tools.sensor_tool import fetch_sensor_data_with_components, detect_device_data_types
from src.exceptions import AgentError
from src.agent.query_entities import allows_explicit_multi_scope_aggregation, extract_current_question_text, normalize_search_target, parse_query_entities
from src.agent.query_plan import QueryPlan, coerce_query_plan
from src.agent.query_planner import LLMQueryPlanner
from src.agent.focused_response import build_focused_sensor_response, format_metric_value
from src.agent.action_override_policy import (
    ActionOverrideContext,
    apply_action_override_policy,
    decide_metadata_override,
    decide_sensor_override,
)
from src.agent.query_plan_state import (
    build_compat_intent_from_state,
    build_query_plan_context as build_query_plan_context_from_state,
    get_comparison_targets_from_state,
    get_data_type_from_state,
    get_explicit_device_codes_from_state,
    get_project_hints_from_state,
    get_state_targets,
    has_detect_data_types_intent_from_state,
    has_device_listing_intent_from_state,
    has_pagination_intent_from_state,
    has_project_listing_intent_from_state,
    has_project_stats_intent_from_state,
    has_sensor_query_intent_from_state,
    is_comparison_query,
)
from src.agent.query_time_range import (
    build_month_range as build_shared_month_range,
    resolve_time_range_from_query as resolve_shared_time_range_from_query,
)


logger = logging.getLogger(__name__)

# 最大迭代次数，防止无限循环
MAX_ITERATIONS = 5
CLARIFICATION_CANDIDATE_LIMIT = 50
EMPTY_RESULT_MESSAGE = "当前时间范围内未查询到符合条件的数据，请尝试放宽时间范围、检查设备代号或调整过滤条件。"

CHART_TYPE_LABELS = {
    "line": "折线图",
    "bar": "柱状图",
    "scatter": "散点图",
    "boxplot": "箱线图",
    "heatmap": "热力图",
}


class AgentState(TypedDict):
    """Agent 状态"""
    user_query: str
    rewritten_query: Optional[str]
    intent: str
    keywords: Optional[List[str]]
    device_codes: Optional[List[str]]
    time_range: Optional[Dict[str, str]]
    metadata_result: Optional[List[Dict]]
    raw_data: Optional[List[Dict]]
    final_response: Optional[str]
    error: Optional[str]
    iteration: int
    history: List[Dict]  # 记录每次迭代的思考和行动


# 系统提示词 - 定义 Agent 的能力和行为
SYSTEM_PROMPT_TEMPLATE = """你是一个智能数据查询助手，可以帮助用户查询设备信息和传感器数据。

**重要：直接输出 JSON，不要使用 <think> 标签或其他包装！**

## 当前时间
{current_time}

## 你的能力
1. **搜索设备** - 根据关键词搜索设备（设备名称、项目名称、设备代号等）
2. **探测数据类型** - 检查设备有哪些类型的数据（电量、电流、电压等）
3. **获取时序数据** - 获取指定设备在指定时间范围内的传感器数据
4. **列出项目** - 列出所有可用的项目
5. **项目设备统计** - 统计各项目的设备数量
6. **直接回答** - 回答一般性问题（问候、帮助等）

## 工具说明
- `search_devices(keywords)`: 搜索设备，keywords 是搜索关键词列表
- `detect_data_types(device_codes)`: 探测设备有哪些数据类型（用户问"有哪些数据"时使用）
- `get_sensor_data(device_codes, start_time, end_time, data_type, page, page_size, value_filter)`: 获取时序数据
  - device_codes: 设备代号列表（必须从 search_devices 结果中获取！）
  - start_time, end_time: 时间范围 (YYYY-MM-DD)
  - data_type: 数据类型（见下方列表）
  - page: 页码（可选，默认1）
  - page_size: 每页记录数（可选，默认0表示不分页）
  - value_filter: 数值过滤条件（可选），如 {{"gt": 100}} 表示大于100，{{"lt": 50}} 表示小于50
  - data_type 支持的数据类型：
    - "ep" - 电量数据（默认）
    - "i" - 电流数据
    - "u_line" - 电压数据
    - "p" - 功率数据
    - "qf" - 功率因数
    - "t" - 温度数据
- `list_projects()`: 列出所有项目
- `get_project_stats()`: 获取各项目的设备数量统计，适用场景：
  - "哪个项目设备最多/最少？"
  - "各项目有多少设备？"
  - "项目设备数量排名"
  - "统计一下项目情况"
- `direct_answer(message)`: 直接回答用户

### value_filter 数值过滤使用场景
get_sensor_data 的 value_filter 参数适用于：
- "功率超过100的记录" → {{"gt": 100}}
- "电流小于50的数据" → {{"lt": 50}}
- "电压低于220的异常" → {{"lt": 220}}
- "温度高于30度的时段" → {{"gt": 30}}
- "用电量大于1000的设备" → {{"gt": 1000}}
- "功率因数低于0.8的记录" → {{"lt": 0.8}}

## 响应格式
请以 JSON 格式响应：
```json
{{
    "thought": "你的思考过程",
    "action": "search_devices | detect_data_types | get_sensor_data | list_projects | get_project_stats | direct_answer | final_answer",
    "action_input": {{...}},
    "rewritten_query": "改写后的查询（可选）"
}}
```

## 重要规则
1. **查询数据前必须先搜索设备！** 用户说的"电梯"、"变压器"不是设备代号！
2. **用户问"有哪些数据"时，先用 detect_data_types 探测**，不要直接查 ep
3. **关键词要简洁**：
   - 搜索 "电梯" 而不是 "电梯设备"
   - 搜索 "变压器" 而不是 "变压器设备"
   - 搜索 "空调" 而不是 "空调系统"
   - 不要在关键词后面加"设备"、"系统"等后缀
4. **利用对话历史**：如果上一轮已经找到设备，直接使用那些设备代号，不要重新搜索
5. **时间理解**：
   - "上个月" = 上一个自然月（如当前是2026年1月，则为2025年12月1日-31日）
   - "最近一周" = 过去7天
   - "昨天" = 昨天一整天
   - 默认查询最近7天
6. 最多迭代 {max_iter} 次
7. 电量计算：当日最大值 - 当日最小值 = 当日用电量
8. **搜索失败时**：如果交集结果为0，尝试减少关键词或使用更通用的词
9. **回退搜索结果处理**：当搜索结果标记为 fallback=true 时，说明多关键词组合未精确匹配：
   - 友好地告知用户找到了哪些相关设备
   - 列出设备名称供用户确认
   - 询问是否要查询这些设备，或请用户提供更精确的信息

## 正确流程示例

### 示例1：查询电梯用电量
用户问："电子楼的电梯最近一周用了多少电？"

第一步：搜索设备（关键词简洁，不加"设备"后缀）
```json
{{"thought": "需要先搜索电梯，关键词用'电梯'而不是'电梯设备'", "action": "search_devices", "action_input": {{"keywords": ["电子楼", "电梯"]}}}}
```
第二步：查询电量数据
```json
{{"thought": "找到设备，查询用电数据", "action": "get_sensor_data", "action_input": {{"device_codes": ["b1_b3"], "start_time": "2026-01-08", "end_time": "2026-01-15", "data_type": "ep"}}}}
```

### 示例1b：利用对话历史
用户之前问过"电子楼有哪些电梯"，AI 找到了设备 b1_b3, b1_b4
用户现在问："这些电梯昨天的电流情况"

直接使用之前的设备代号，不需要重新搜索：
```json
{{"thought": "用户说'这些电梯'，指的是上一轮找到的设备，直接使用", "action": "get_sensor_data", "action_input": {{"device_codes": ["b1_b3", "b1_b4"], "start_time": "2026-01-15", "end_time": "2026-01-15", "data_type": "i"}}}}
```

### 示例1c：比较多个位置的设备
用户问："对比一下1号楼和2号楼的变压器功率"

搜索时把所有位置关键词放在一起，让系统合并结果：
```json
{{"thought": "用户要比较1号楼和2号楼，把两个位置和设备类型一起搜索", "action": "search_devices", "action_input": {{"keywords": ["1号楼", "2号楼", "变压器"]}}}}
```
这样系统会返回1号楼和2号楼中包含"变压器"的设备（如果有的话），或者返回两个楼的所有设备供用户选择。

### 示例2：查询设备有哪些数据
用户问："计量表AA3有哪些数据记录？"

第一步：搜索设备
```json
{{"thought": "先搜索设备", "action": "search_devices", "action_input": {{"keywords": ["计量表AA3"]}}}}
```
第二步：探测数据类型
```json
{{"thought": "用户想知道有哪些数据，需要探测数据类型", "action": "detect_data_types", "action_input": {{"device_codes": ["b3_b13"]}}}}
```
第三步：返回结果
```json
{{"thought": "已探测到数据类型", "action": "final_answer", "action_input": {{"answer": "该设备有电流(13266条)、电压(13266条)、功率(4422条)数据", "show_table": false}}}}
```

### 示例3：搜索结果回退时的友好回复
用户问："对比一下1号楼和2号楼的变压器功率"
搜索 ["1号楼", "变压器"] 交集为空，回退使用 "1号楼" 的3个结果

正确的回复方式：
```json
{{"thought": "搜索结果是回退的，找到的设备可能不是变压器，需要友好地告知用户并展示找到的设备", "action": "final_answer", "action_input": {{"answer": "我在1号楼找到了以下设备：\\n1. 1号楼配电室（代号: a1_b1）\\n2. 1号楼照明（代号: a1_b2）\\n3. 1号楼空调（代号: a1_b3）\\n\\n这些设备中没有找到名称包含'变压器'的设备。请问您要查询以上哪个设备的功率数据？或者您可以提供变压器的具体名称或设备代号。", "show_table": true, "table_type": "devices"}}}}
```

### 示例4：查询哪个项目设备最多
用户问："哪个项目的设备最多？" / "各项目设备数量" / "项目统计"

直接使用 get_project_stats 工具：
```json
{{"thought": "用户想知道项目设备数量统计，使用 get_project_stats", "action": "get_project_stats", "action_input": {{}}}}
```

### 示例5：查询超过/低于某个值的数据
用户问："北京电力自动化项目里，功率超过100的设备有哪些？"

第一步：搜索项目下的设备
```json
{{"thought": "先搜索北京电力自动化项目的设备", "action": "search_devices", "action_input": {{"keywords": ["北京电力自动化"]}}}}
```
第二步：查询功率数据并过滤
```json
{{"thought": "查询功率数据，过滤大于100的记录", "action": "get_sensor_data", "action_input": {{"device_codes": ["a1_b1", "a1_b2"], "start_time": "2026-01-09", "end_time": "2026-01-16", "data_type": "p", "value_filter": {{"gt": 100}}}}}}
```

### 示例5b：查询异常数据
用户问："电压低于200的异常记录"

```json
{{"thought": "查询电压数据，过滤低于200的异常记录", "action": "get_sensor_data", "action_input": {{"device_codes": ["a1_b1"], "start_time": "2026-01-09", "end_time": "2026-01-16", "data_type": "u_line", "value_filter": {{"lt": 200}}}}}}
```

### 示例5c：查询高温报警
用户问："温度超过35度的记录"

```json
{{"thought": "查询温度数据，过滤高于35度的记录", "action": "get_sensor_data", "action_input": {{"device_codes": ["a1_b1"], "start_time": "2026-01-09", "end_time": "2026-01-16", "data_type": "t", "value_filter": {{"gt": 35}}}}}}
```

## final_answer 格式（重要！）
final_answer 的 action_input 必须包含：
```json
{{
    "thought": "总结查询结果",
    "action": "final_answer",
    "action_input": {{
        "answer": "文字回复内容（总结统计信息）",
        "show_table": true/false,
        "table_type": "sensor_data | projects | devices | project_stats"
    }}
}}
```

### table_type 说明（数据由系统自动从查询结果获取，你只需指定类型）：
- "sensor_data": 传感器时序数据（调用 get_sensor_data 后使用）
- "projects": 项目列表（调用 list_projects 后使用）
- "devices": 设备列表（调用 search_devices 后使用）
- "project_stats": 项目设备统计（调用 get_project_stats 后使用）

只返回 JSON，不要其他内容。"""


OBSERVATION_PROMPT = """## 上一步执行结果
动作: {action}
结果详情: 
{result_detail}

## 历史记录
{history}

## 用户原始问题
{user_query}

请根据执行结果决定下一步：
- 如果已经获得足够信息，使用 `final_answer` 生成最终响应
- 如果需要更多信息，继续执行其他动作
- 如果搜索没有结果，尝试调整关键词重新搜索（去掉"设备"、"系统"等后缀）
- 如果搜索使用了回退策略（fallback=true），说明多关键词交集为空，结果可能不够精确

## 关键词调整建议
如果搜索结果为0，尝试：
1. 去掉"设备"、"系统"等后缀：电梯设备 → 电梯
2. 减少关键词数量
3. 使用更通用的词

## final_answer 格式要求（必须遵守！）
action_input 必须包含结构化数据：
```json
{{
    "answer": "文字回复",
    "show_table": true/false,
    "table_type": "sensor_data | projects | devices",
    "table_data": [...]  // 项目或设备列表时必须提供
}}
```

### 不同场景的处理：
1. **项目列表查询**：show_table=true, table_type="projects", table_data=项目数组
2. **设备搜索结果**：show_table=true, table_type="devices", table_data=设备数组
3. **传感器数据**：show_table=true, table_type="sensor_data"（数据由前端自动加载）
4. **简单问答**：show_table=false

以 JSON 格式响应。"""


class LLMAgent:
    """
    LLM 驱动的智能代理
    
    采用 ReAct 模式，让 LLM 自主决定如何处理用户查询。
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        metadata_engine=None,
        data_fetcher=None,
        cache_manager=None,
        compressor=None,
        alias_memory=None,
        entity_resolver=None,
    ):
        self.llm = llm
        self.metadata_engine = metadata_engine
        self.data_fetcher = data_fetcher
        self.cache_manager = cache_manager
        self.compressor = compressor
        self.alias_memory = alias_memory or {}
        self.entity_resolver = entity_resolver
        self.query_planner = LLMQueryPlanner(self.llm)
        self._query_plan_cache: Dict[str, QueryPlan] = {}

    def _normalize_alias_key(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).lower()

    def _get_alias_memory_entry(self, keyword: str) -> Optional[Dict]:
        if not isinstance(self.alias_memory, dict):
            return None
        key = self._normalize_alias_key(keyword)
        if not key:
            return None
        cached = self.alias_memory.get(key)
        if not isinstance(cached, dict):
            return None
        device = dict(cached)
        device.setdefault("match_score", 1000.0)
        device.setdefault("matched_fields", ["session_alias"])
        device.setdefault("match_reason", "同一会话历史确认")
        return device

    def _get_cached_devices_from_query(self, user_query: str) -> List[Dict]:
        if not user_query or not isinstance(self.alias_memory, dict):
            return []
        normalized_query = self._normalize_alias_key(user_query)
        if not normalized_query:
            return []

        matches: List[Dict] = []
        seen = set()
        for alias_key, cached in self.alias_memory.items():
            if not alias_key or alias_key not in normalized_query or not isinstance(cached, dict):
                continue
            device_code = cached.get("device")
            if not device_code or device_code in seen:
                continue
            seen.add(device_code)
            device = dict(cached)
            device.setdefault("match_score", 1000.0)
            device.setdefault("matched_fields", ["session_alias"])
            device.setdefault("match_reason", "同一会话历史确认")
            matches.append(device)
        return matches

    def _get_cached_device_codes_from_query(self, user_query: str) -> List[str]:
        seen = set()
        device_codes: List[str] = []
        for item in self._get_cached_devices_from_query(user_query):
            device_code = item.get("device")
            if not device_code or device_code in seen:
                continue
            seen.add(device_code)
            device_codes.append(device_code)
        return device_codes

    def _get_cached_devices_for_aliases(self, aliases: List[str]) -> List[Dict]:
        matches: List[Dict] = []
        seen = set()
        for alias in aliases:
            cached = self._get_alias_memory_entry(alias)
            if not isinstance(cached, dict):
                continue
            dedupe_key = (cached.get("device"), cached.get("project_id"), cached.get("name"), cached.get("tg"))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            matches.append(cached)
        return matches

    def _extract_tg_values_from_devices(self, devices: List[Dict]) -> List[str]:
        values: List[str] = []
        seen = set()
        for device in devices or []:
            tg = str(device.get("tg") or "").strip()
            if not tg or tg in seen:
                continue
            seen.add(tg)
            values.append(tg)
        return values

    def _has_cross_project_collision(self, candidates: List[Dict], device_code: str) -> bool:
        normalized_code = self._normalize_alias_key(device_code)
        exact_matches = [
            item for item in (candidates or [])
            if self._normalize_alias_key(item.get("device")) == normalized_code
        ]
        scopes = {
            (
                str(item.get("project_id") or ""),
                str(item.get("project_name") or ""),
                str(item.get("project_code_name") or ""),
                str(item.get("tg") or ""),
            )
            for item in exact_matches
        }
        return len(exact_matches) > 1 and len(scopes) > 1

    def _can_auto_resolve_exact_code_candidates(self, candidates: List[Dict]) -> bool:
        if len(candidates or []) <= 1:
            return True
        top1 = float(candidates[0].get("context_score") or 0.0)
        top2 = float(candidates[1].get("context_score") or 0.0)
        return top1 > 0 and top2 <= 0

    def _lookup_exact_device_code_candidates(self, device_code: str, user_query: str = ""):
        if not device_code or not self.metadata_engine:
            return [], None
        normalized_code = self._normalize_alias_key(device_code)
        if not normalized_code:
            return [], None

        try:
            catalog = self.metadata_engine.list_all_devices()
        except Exception as exc:
            logger.warning("device.search exact_code_catalog_failed device=%s error=%s", device_code, exc)
            return [], None

        matches = []
        for item in catalog:
            if self._normalize_alias_key(getattr(item, "device", "")) != normalized_code:
                continue
            matches.append(item.to_dict())

        if not matches:
            return [], None

        reranked = self._rerank_devices_with_query_context(matches, device_code, user_query)
        query_info = {
            "type": "exact_device_code",
            "device_code": device_code,
            "candidate_count": len(reranked),
            "cross_project_collision": self._has_cross_project_collision(reranked, device_code),
        }
        context_terms = self._get_query_context_terms(device_code, user_query)
        if context_terms:
            query_info["context_terms"] = context_terms
            query_info["context_applied"] = True
        return reranked, query_info

    def _resolve_explicit_device_scope(self, explicit_codes: List[str], query_text: str) -> Dict[str, Any]:
        resolved_by_code: Dict[str, List[Dict[str, Any]]] = {}
        clarification_groups: List[Dict[str, Any]] = []
        query_infos: List[Dict[str, Any]] = []
        cached_code_count = 0

        comparison_mode = self._has_comparison_intent(query_text)
        for device_code in explicit_codes:
            normalized_code = self._normalize_alias_key(device_code)
            cached_device = None if comparison_mode else self._get_alias_memory_entry(device_code)
            if cached_device:
                resolved_by_code[normalized_code] = [dict(cached_device)]
                cached_code_count += 1
                continue

            candidates, query_info = self._lookup_exact_device_code_candidates(device_code, query_text)
            if query_info:
                query_infos.append(query_info)
            if not candidates:
                continue
            allow_multi_scope_aggregation = allows_explicit_multi_scope_aggregation(query_text, device_code)
            if self._has_cross_project_collision(candidates, device_code):
                if allow_multi_scope_aggregation:
                    resolved_by_code[normalized_code] = [dict(candidate) for candidate in candidates if isinstance(candidate, dict)]
                    continue
                if not self._can_auto_resolve_exact_code_candidates(candidates):
                    conflict_candidates = self._mark_exact_code_conflict_candidates(device_code, candidates)
                    clarification_candidates = self._select_clarification_candidates(conflict_candidates)
                    aggregate_candidate = self._build_aggregate_scope_candidate(device_code, clarification_candidates)
                    if aggregate_candidate:
                        clarification_candidates.append(aggregate_candidate)
                    clarification_groups.append(
                        {
                            "keyword": device_code,
                            "candidates": clarification_candidates,
                        }
                    )
                    continue
            resolved_by_code[normalized_code] = [dict(candidates[0])]

        resolved_devices: List[Dict[str, Any]] = []
        seen = set()
        for device_code in explicit_codes:
            resolved_group = resolved_by_code.get(self._normalize_alias_key(device_code)) or []
            for resolved_device in resolved_group:
                if not isinstance(resolved_device, dict):
                    continue
                dedupe_key = (
                    resolved_device.get("device"),
                    resolved_device.get("project_id"),
                    resolved_device.get("name"),
                    resolved_device.get("tg"),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                resolved_devices.append(resolved_device)

        complete = not clarification_groups and all(resolved_by_code.get(self._normalize_alias_key(code)) for code in explicit_codes)
        if complete:
            if cached_code_count == len(explicit_codes):
                source = "session_alias"
            elif cached_code_count > 0:
                source = "mixed_explicit_resolved"
            else:
                source = "explicit_resolved"
        else:
            source = "explicit"

        return {
            "resolved_devices": resolved_devices,
            "clarification_groups": clarification_groups,
            "query_infos": query_infos,
            "complete": complete,
            "source": source,
            "cached_code_count": cached_code_count,
        }

    def _resolve_preferred_device_scope(self, query_text: str, parsed) -> tuple[List[str], List[Dict], str]:
        explicit_codes = list(parsed.explicit_device_codes)
        if explicit_codes:
            resolution = self._resolve_explicit_device_scope(explicit_codes, query_text)
            if resolution.get("complete"):
                return explicit_codes, list(resolution.get("resolved_devices") or []), str(resolution.get("source") or "explicit_resolved")
            return explicit_codes, [], "explicit"

        cached_devices = self._get_cached_devices_from_query(query_text)
        if not cached_devices:
            return [], [], ""
        device_codes = []
        seen = set()
        for item in cached_devices:
            device_code = item.get("device")
            if not device_code or device_code in seen:
                continue
            seen.add(device_code)
            device_codes.append(device_code)
        return device_codes, cached_devices, "session_alias"

    def _build_month_range(self, year: int, month: int) -> Optional[Dict[str, str]]:
        return build_shared_month_range(year, month)

    def _resolve_time_range_from_query(self, query_text: str, now: Optional[datetime] = None) -> Optional[Dict[str, str]]:
        return resolve_shared_time_range_from_query(query_text, now=now)

    def _is_query_plan_sensor_flow(self, plan: QueryPlan) -> bool:
        return bool(
            plan.has_sensor_intent
            or plan.query_mode in {"sensor_query", "comparison", "ranked_timepoints", "ranked_buckets", "trend_decision", "anomaly_points"}
        )

    def _resolve_query_plan_time_range(self, query_text: str, plan: QueryPlan) -> Dict[str, str]:
        if plan.time_start and plan.time_end:
            return {"start_time": plan.time_start, "end_time": plan.time_end}

        resolved = self._resolve_time_range_from_query(query_text)
        if resolved:
            return resolved

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=7)
        return {
            "start_time": start_dt.strftime("%Y-%m-%d"),
            "end_time": end_dt.strftime("%Y-%m-%d"),
        }

    def _build_query_plan_summary(self, plan: QueryPlan) -> str:
        targets = list(plan.search_targets or plan.explicit_device_codes)
        target_text = ", ".join(targets[:3]) if targets else "-"
        time_text = f"{plan.time_start or '-'} ~ {plan.time_end or '-'}"
        return (
            f"模式={plan.query_mode}，指标={plan.inferred_data_type or '-'}，"
            f"目标={target_text}，时间={time_text}，风格={plan.response_style}"
        )

    def _build_query_plan_step_name(self, plan: QueryPlan) -> str:
        if plan.has_project_listing_intent or plan.query_mode == "project_listing":
            return "列出项目"
        if plan.has_project_stats_intent or plan.query_mode == "project_stats":
            return "统计项目设备"
        if plan.has_detect_data_types_intent or plan.query_mode == "detect_data_types":
            return "探测数据类型"
        if plan.has_device_listing_intent or plan.query_mode == "device_listing":
            return "搜索设备"
        if self._is_query_plan_sensor_flow(plan):
            return "获取时序数据"
        return "执行查询计划"

    def _resolve_devices_from_query_plan(self, query_text: str, plan: QueryPlan) -> Dict[str, Any]:
        explicit_codes = [str(code).strip() for code in plan.explicit_device_codes if str(code).strip()]
        if explicit_codes:
            resolution = self._resolve_explicit_device_scope(explicit_codes, query_text)
            clarification_groups = list(resolution.get("clarification_groups") or [])
            if clarification_groups:
                resolved_devices = list(resolution.get("resolved_devices") or [])
                query_infos = list(resolution.get("query_infos") or [])
                query_info = query_infos[0] if query_infos else None
                return {
                    "success": False,
                    "needs_clarification": True,
                    "clarification_required": True,
                    "clarification_candidates": clarification_groups,
                    "message": self._build_clarification_message(clarification_groups, resolved_devices),
                    "devices": resolved_devices,
                    "resolved_devices": resolved_devices,
                    "query_info": query_info,
                }

            resolved_devices = list(resolution.get("resolved_devices") or [])
            if not resolved_devices:
                return {
                    "success": False,
                    "error": f"未找到设备代号 {', '.join(explicit_codes)} 对应的设备。",
                }

            return {
                "success": True,
                "device_codes": [item.get("device") for item in resolved_devices if item.get("device")],
                "resolved_devices": resolved_devices,
                "tg_values": self._extract_tg_values_from_devices(resolved_devices),
                "query_info": (list(resolution.get("query_infos") or []) or [None])[0],
            }

        cached_devices = self._get_cached_devices_from_query(query_text)
        if cached_devices and not plan.search_targets:
            return {
                "success": True,
                "device_codes": [item.get("device") for item in cached_devices if item.get("device")],
                "resolved_devices": cached_devices,
                "tg_values": self._extract_tg_values_from_devices(cached_devices),
                "query_info": None,
            }

        search_targets = self._extract_search_targets(list(plan.search_targets), query_text, bool(plan.has_comparison_intent))
        if not search_targets:
            return {
                "success": False,
                "error": "未识别到可查询的设备或目标，请补充设备、项目或别名。",
            }

        search_result = self._action_search_devices(
            {
                "keywords": search_targets,
                "user_query": query_text,
                "comparison_mode": bool(plan.has_comparison_intent),
            }
        )

        if search_result.get("needs_clarification"):
            return {
                "success": False,
                "needs_clarification": True,
                "clarification_required": True,
                "clarification_candidates": search_result.get("clarification_candidates") or [],
                "message": search_result.get("message") or "匹配到多个候选设备，请先确认。",
                "devices": search_result.get("devices") or [],
                "resolved_devices": search_result.get("resolved_devices") or [],
                "query_info": search_result.get("query_info"),
            }

        if self._is_query_plan_sensor_flow(plan) and search_result.get("fallback") and not plan.has_device_listing_intent:
            fallback_candidates = self._select_clarification_candidates(search_result.get("devices") or [])
            return {
                "success": False,
                "needs_clarification": True,
                "clarification_required": True,
                "clarification_candidates": [
                    {
                        "keyword": search_result.get("fallback_keyword") or (search_targets[0] if search_targets else "device"),
                        "candidates": fallback_candidates,
                    }
                ],
                "message": search_result.get("message") or "匹配不够精确，请先确认设备。",
                "devices": search_result.get("devices") or [],
                "resolved_devices": search_result.get("resolved_devices") or [],
                "query_info": search_result.get("query_info"),
            }

        resolved_devices = list(search_result.get("resolved_devices") or search_result.get("devices") or [])
        device_codes = list(search_result.get("device_codes") or [item.get("device") for item in resolved_devices if item.get("device")])
        if not device_codes:
            return {
                "success": False,
                "error": search_result.get("message") or "未找到匹配的设备。",
            }

        return {
            "success": True,
            "device_codes": device_codes,
            "resolved_devices": resolved_devices,
            "tg_values": self._extract_tg_values_from_devices(resolved_devices),
            "query_info": search_result.get("query_info"),
            "search_result": search_result,
        }

    def _build_sensor_query_params_from_plan(self, query_text: str, plan: QueryPlan, scope_result: Dict[str, Any]) -> Dict[str, Any]:
        time_range = self._resolve_query_plan_time_range(query_text, plan)
        fetch_all = (
            plan.query_mode in {"comparison", "ranked_timepoints", "ranked_buckets", "trend_decision", "anomaly_points"}
            and not plan.has_pagination_intent
        ) or (len(scope_result.get("device_codes") or []) > 1 and not plan.has_pagination_intent)
        return {
            "device_codes": list(scope_result.get("device_codes") or []),
            "tg_values": list(scope_result.get("tg_values") or []),
            "start_time": time_range["start_time"],
            "end_time": time_range["end_time"],
            "data_type": plan.inferred_data_type or "ep",
            "page": 1,
            "page_size": 0 if fetch_all else 50,
            "user_query": query_text,
            "query_plan": self._serialize_query_plan(plan),
        }

    def _build_sensor_table_preview(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_data = result.get("data")
        if isinstance(raw_data, str):
            try:
                raw_data = json.loads(raw_data)
            except json.JSONDecodeError:
                return None

        if not isinstance(raw_data, list) or not raw_data:
            return None

        preview_limit = 50
        table_rows: List[Dict[str, Any]] = []
        for item in raw_data[:preview_limit]:
            if isinstance(item, dict):
                is_aggregated_row = any(key in item for key in ("average", "total", "diff", "max", "min"))
                if is_aggregated_row:
                    time_field = item.get("date") or item.get("month") or item.get("year") or item.get("time") or "-"
                    value_field = item.get("average")
                    if value_field is None:
                        value_field = item.get("total")
                    if value_field is None:
                        value_field = item.get("diff")
                    if value_field is None:
                        value_field = item.get("max")
                    if value_field is None:
                        value_field = item.get("min")
                    table_rows.append(
                        {
                            "time": time_field,
                            "device": item.get("device", "-"),
                            "tag": item.get("tag", "-"),
                            "value": round(value_field, 2) if isinstance(value_field, float) else value_field,
                            "record_count": item.get("record_count", ""),
                        }
                    )
                else:
                    table_rows.append(
                        {
                            "time": item.get("logTime") or item.get("time", ""),
                            "device": item.get("device", "-"),
                            "tag": item.get("tag", "-"),
                            "value": item.get("val") if item.get("val") is not None else "",
                        }
                    )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                table_rows.append(
                    {
                        "time": item[0],
                        "device": "-",
                        "tag": "-",
                        "value": item[1],
                    }
                )

        if not table_rows:
            return None

        total_count = int(result.get("total_count") or len(raw_data) or len(table_rows))
        return {
            "success": True,
            "data": table_rows,
            "total_count": total_count,
            "page": int(result.get("page") or 1),
            "page_size": min(preview_limit, len(table_rows)),
            "total_pages": int(result.get("total_pages") or 1),
            "has_more": bool(result.get("has_more", False)) or total_count > len(table_rows),
            "statistics": result.get("statistics"),
            "analysis": result.get("analysis"),
            "focused_table": result.get("focused_table"),
            "chart_specs": result.get("chart_specs"),
            "show_charts": result.get("show_charts", False),
            "is_sampled": result.get("is_sampled", False),
            "aggregation_type": result.get("aggregation_type"),
        }

    def _build_direct_answer_from_analysis(self, result: Dict[str, Any]) -> str:
        analysis = result.get("analysis") or {}
        headline = str(analysis.get("headline") or "").strip()
        if not headline:
            return self._build_sensor_terminal_response(result, "")

        lines = [headline]
        for card in (analysis.get("summary_cards") or [])[:2]:
            label = str(card.get("label") or "").strip()
            value = str(card.get("value") or "").strip()
            detail = str(card.get("detail") or "").strip()
            if not label or not value:
                continue
            line = f"- {label}: {value}"
            if detail:
                line += f"（{detail}）"
            lines.append(line)
        return "\n".join(lines).strip()

    def _build_sensor_response_by_style(self, plan: QueryPlan, result: Dict[str, Any]) -> str:
        if self._is_empty_sensor_result(result):
            return EMPTY_RESULT_MESSAGE

        focused_response = self._build_focused_sensor_response(result)
        if focused_response:
            return focused_response

        analysis = result.get("analysis") or {}
        if plan.response_style == "direct_answer" and analysis.get("mode") != "comparison":
            return self._build_direct_answer_from_analysis(result)
        if plan.response_style == "list" and result.get("success"):
            metric = str(analysis.get("metric") or plan.inferred_data_type or "数据")
            total_count = int(result.get("total_count") or 0)
            return f"已获取{metric}数据，共 {total_count} 条记录，请查看表格。"
        return self._build_sensor_terminal_response(result, "")

    def _execute_heuristic_action_fallback(self, user_query: str, plan: QueryPlan, heuristic_action: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(heuristic_action, dict):
            return None

        action = str(heuristic_action.get("action") or "").strip()
        if not action or action == "final_answer":
            return None

        state = self._init_state(user_query)
        action_input = self._normalize_action_input(
            heuristic_action.get("action_input") or {},
            action,
            state,
        )
        if action in {"get_sensor_data", "detect_data_types"}:
            action_input = self._hydrate_confirmed_scope_from_query(action_input)

        if action == "search_devices":
            result = self._action_search_devices(action_input)
            return {
                "kind": "devices",
                "step": "搜索设备",
                "result": result,
                "query_info": result.get("query_info"),
                "query_params": action_input,
            }

        if action == "detect_data_types":
            result = self._action_detect_data_types(action_input)
            return {
                "kind": "detect_data_types",
                "step": "探测数据类型",
                "result": result,
                "query_info": result.get("query_info"),
                "query_params": action_input,
            }

        if action == "get_sensor_data":
            result = self._action_get_sensor_data(action_input)
            return {
                "kind": "sensor_data",
                "step": "获取时序数据",
                "result": result,
                "query_info": result.get("query_info"),
                "query_params": action_input,
            }

        if action == "list_projects":
            result = self._action_list_projects()
            return {"kind": "projects", "step": "列出项目", "result": result, "query_info": result.get("query_info")}

        if action == "get_project_stats":
            result = self._action_get_project_stats()
            return {"kind": "project_stats", "step": "获取项目统计", "result": result, "query_info": result.get("query_info")}

        if action == "direct_answer":
            message = str(action_input.get("message") or heuristic_action.get("thought") or "").strip()
            return {
                "kind": "direct_answer",
                "step": "直接回答",
                "result": {"success": True, "message": message or "我已完成处理，请查看上方结果。"},
                "query_info": None,
            }

        return None

    def _execute_query_plan(self, user_query: str, plan: QueryPlan) -> Dict[str, Any]:
        query_text = self._get_current_question_text(user_query)

        if plan.has_project_listing_intent or plan.query_mode == "project_listing":
            result = self._action_list_projects()
            return {"kind": "projects", "step": "列出项目", "result": result, "query_info": result.get("query_info")}

        if plan.has_project_stats_intent or plan.query_mode == "project_stats":
            result = self._action_get_project_stats()
            return {"kind": "project_stats", "step": "获取项目统计", "result": result, "query_info": result.get("query_info")}

        if plan.has_device_listing_intent and not self._is_query_plan_sensor_flow(plan):
            targets = self._extract_search_targets(list(plan.search_targets or plan.explicit_device_codes), query_text, bool(plan.has_comparison_intent))
            result = self._action_search_devices({"keywords": targets, "user_query": query_text, "comparison_mode": bool(plan.has_comparison_intent)})
            return {"kind": "devices", "step": "搜索设备", "result": result, "query_info": result.get("query_info")}

        if plan.has_detect_data_types_intent or plan.query_mode == "detect_data_types":
            scope_result = self._resolve_devices_from_query_plan(query_text, plan)
            if not scope_result.get("success"):
                return {
                    "kind": "clarification" if scope_result.get("needs_clarification") else "error",
                    "step": "解析设备范围",
                    "result": scope_result,
                    "query_info": scope_result.get("query_info"),
                }
            result = self._action_detect_data_types(
                {
                    "device_codes": scope_result.get("device_codes") or [],
                    "tg_values": scope_result.get("tg_values") or [],
                    "user_query": query_text,
                }
            )
            return {
                "kind": "detect_data_types",
                "step": "探测数据类型",
                "result": result,
                "query_info": result.get("query_info"),
                "query_params": {
                    "device_codes": list(scope_result.get("device_codes") or []),
                    "tg_values": list(scope_result.get("tg_values") or []),
                    "query_plan": self._serialize_query_plan(plan),
                },
            }

        if self._is_query_plan_sensor_flow(plan):
            scope_result = self._resolve_devices_from_query_plan(query_text, plan)
            if not scope_result.get("success"):
                return {
                    "kind": "clarification" if scope_result.get("needs_clarification") else "error",
                    "step": "解析设备范围",
                    "result": scope_result,
                    "query_info": scope_result.get("query_info"),
                }
            query_params = self._build_sensor_query_params_from_plan(query_text, plan, scope_result)
            result = self._action_get_sensor_data(query_params)
            return {
                "kind": "sensor_data",
                "step": "获取时序数据",
                "result": result,
                "query_info": result.get("query_info"),
                "query_params": query_params,
            }

        heuristic_execution = self._execute_heuristic_action_fallback(
            query_text,
            plan,
            self._build_heuristic_action(self._init_state(query_text), fast_path=True),
        )
        if heuristic_execution is not None:
            logger.info(
                "query.plan.fallback action=%s mode=%s source=%s",
                heuristic_execution.get("kind"),
                plan.query_mode,
                plan.source,
            )
            return heuristic_execution

        return {
            "kind": "direct_answer",
            "step": "直接回答",
            "result": {
                "success": True,
                "message": "我已完成 QueryPlan 解析，但当前还需要更具体的查询条件才能继续执行。",
            },
            "query_info": None,
        }

    def _build_final_event_from_query_plan(self, user_query: str, plan: QueryPlan, execution: Dict[str, Any], total_duration_ms: int) -> Dict[str, Any]:
        result = execution.get("result") or {}
        kind = execution.get("kind")
        final_event: Dict[str, Any] = {
            "type": "final_answer",
            "query_plan": self._serialize_query_plan(plan),
            "total_duration_ms": total_duration_ms,
            "show_table": False,
            "table_type": "",
            "query_params": execution.get("query_params"),
        }

        if result.get("needs_clarification"):
            final_event.update(
                {
                    "response": result.get("message") or "匹配到多个候选设备，请先确认。",
                    "clarification_required": True,
                    "clarification_candidates": result.get("clarification_candidates") or [],
                    "devices": result.get("devices") or result.get("resolved_devices") or [],
                }
            )
            return final_event

        if kind == "projects":
            projects = result.get("projects") or []
            final_event.update(
                {
                    "response": f"已找到 {len(projects)} 个项目，请查看下表。" if projects else "当前没有可用项目。",
                    "show_table": bool(projects),
                    "table_type": "projects" if projects else "",
                    "projects": projects,
                }
            )
            return final_event

        if kind == "project_stats":
            stats = result.get("stats") or []
            final_event.update(
                {
                    "response": "已整理各项目设备数量统计，请查看下表。" if stats else "当前没有可用的项目设备统计。",
                    "show_table": bool(stats),
                    "table_type": "project_stats" if stats else "",
                    "project_stats": stats,
                }
            )
            return final_event

        if kind == "devices":
            devices = result.get("devices") or []
            response = str(result.get("message") or "").strip()
            if not response:
                response = f"已找到 {len(devices)} 个相关设备，请查看下表。" if devices else "未找到匹配的设备。"
            final_event.update(
                {
                    "response": response,
                    "show_table": bool(devices),
                    "table_type": "devices" if devices else "",
                    "devices": devices,
                }
            )
            return final_event

        if kind == "detect_data_types":
            if not result.get("success"):
                final_event["response"] = str(result.get("error") or "探测数据类型失败。")
                return final_event
            summary = str(result.get("summary") or "").strip()
            final_event["response"] = f"已检测到以下数据类型：{summary}" if summary else "当前设备未检测到可用数据类型。"
            return final_event

        if kind == "sensor_data":
            if not result.get("success"):
                final_event["response"] = str(result.get("error") or EMPTY_RESULT_MESSAGE)
                return final_event
            raw_data = result.get("data")
            if isinstance(raw_data, str):
                try:
                    raw_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    raw_data = None
            final_event.update(
                {
                    "response": self._build_sensor_response_by_style(plan, result),
                    "show_table": True,
                    "table_type": "sensor_data",
                    "analysis": result.get("analysis"),
                    "chart_specs": result.get("chart_specs"),
                    "show_charts": result.get("show_charts", False),
                    "table_preview": self._build_sensor_table_preview(result),
                    "_chart_cache": {
                        "raw_data": raw_data if isinstance(raw_data, list) else None,
                        "statistics": result.get("statistics"),
                        "chart_specs": result.get("chart_specs"),
                    } if raw_data else None,
                }
            )
            return final_event

        final_event["response"] = str(result.get("message") or "我已经完成 QueryPlan 解析，但还需要你补充更具体的查询条件。")
        return final_event

    def run(self, user_query: str) -> str:
        """同步执行 QueryPlan -> DB -> answer_style 主流程。"""
        final_response = "抱歉，我暂时无法完成本次查询。"
        for event in self.run_with_progress(user_query):
            if event.get("type") == "final_answer":
                final_response = str(event.get("response") or final_response)
        return final_response

    def run_with_progress(self, user_query: str) -> Generator[Dict, None, None]:
        """按 QueryPlan 主流程流式返回执行进度。"""
        request_started_perf = time.perf_counter()
        query_text = self._get_current_question_text(user_query)

        def _timestamp_ms() -> int:
            return int(time.time() * 1000)

        def _duration_ms(started_perf: float) -> int:
            return int((time.perf_counter() - started_perf) * 1000)

        try:
            planning_step = "解析 QueryPlan"
            planning_started_perf = time.perf_counter()
            yield {"type": "step_start", "step": planning_step, "timestamp_ms": _timestamp_ms()}
            plan = self._get_query_plan(query_text)
            yield {
                "type": "step_done",
                "step": planning_step,
                "info": self._build_query_plan_summary(plan),
                "duration_ms": _duration_ms(planning_started_perf),
            }

            execution_step = self._build_query_plan_step_name(plan)
            execution_started_perf = time.perf_counter()
            yield {"type": "step_start", "step": execution_step, "timestamp_ms": _timestamp_ms()}
            execution = self._execute_query_plan(query_text, plan)
            execution_result = execution.get("result")
            execution_summary = self._summarize_result(execution_result) if execution_result is not None else execution.get("kind", execution_step)
            yield {
                "type": "step_done",
                "step": execution_step,
                "info": execution_summary,
                "query_info": execution.get("query_info"),
                "duration_ms": _duration_ms(execution_started_perf),
            }

            response_step = "生成响应"
            response_started_perf = time.perf_counter()
            yield {"type": "step_start", "step": response_step, "timestamp_ms": _timestamp_ms()}
            final_event = self._build_final_event_from_query_plan(
                query_text,
                plan,
                execution,
                _duration_ms(request_started_perf),
            )
            yield {
                "type": "step_done",
                "step": response_step,
                "info": f"按 {plan.response_style} 输出最终回答",
                "duration_ms": _duration_ms(response_started_perf),
            }
            yield final_event
        except Exception as exc:
            logger.exception("query_plan.pipeline.failed error=%s", exc)
            yield {
                "type": "final_answer",
                "response": f"处理你的请求时遇到问题：{str(exc)}",
                "show_table": False,
                "table_type": "",
                "query_params": None,
                "query_plan": None,
                "total_duration_ms": _duration_ms(request_started_perf),
            }

    def _get_system_prompt(self) -> str:
        """获取带当前时间的系统提示词"""
        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S (星期%w)")
        # 转换星期
        weekday_map = {"0": "日", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六"}
        for k, v in weekday_map.items():
            current_time = current_time.replace(f"星期{k}", f"星期{v}")
        return SYSTEM_PROMPT_TEMPLATE.format(current_time=current_time, max_iter=MAX_ITERATIONS)
    
    def _think(self, state: AgentState) -> Optional[Dict]:
        """让 LLM 思考下一步行动"""
        try:
            system_prompt = self._get_system_prompt()
            
            if state["iteration"] == 1:
                # 第一次思考
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"用户问题: {state['user_query']}")
                ]
            else:
                # 后续思考，包含历史
                history_text = self._format_history(state["history"])
                last_action = state["history"][-1] if state["history"] else {}
                
                # 获取详细结果
                result_detail = last_action.get("result_detail", last_action.get("result", ""))
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=OBSERVATION_PROMPT.format(
                        action=last_action.get("action", ""),
                        result_detail=result_detail,
                        history=history_text,
                        user_query=state["user_query"]
                    ))
                ]
            
            response = self.llm.invoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析 JSON 响应
            return self._parse_llm_response(response_text)
            
        except Exception as e:
            logger.error(f"Think failed: {e}")
            return None
    
    def _parse_llm_response(self, response_text: str) -> Optional[Dict]:
        """解析 LLM 的 JSON 响应"""
        try:
            # 移除 <think> 标签
            cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            # 移除 markdown 代码块
            if "```" in cleaned:
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            # 提取 JSON
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return json.loads(cleaned)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response text: {response_text}")
            
            # 尝试从文本中提取意图
            text_lower = response_text.lower()
            if "direct_answer" in text_lower or "你好" in response_text or "帮助" in response_text:
                return {
                    "thought": "用户在打招呼或询问帮助",
                    "action": "direct_answer",
                    "action_input": {"message": self._extract_direct_answer(response_text)}
                }
            
            return None
    
    def _extract_direct_answer(self, text: str) -> str:
        """从文本中提取直接回答"""
        # 尝试提取 message 字段
        match = re.search(r'"message"\s*:\s*"([^"]*)"', text)
        if match:
            return match.group(1)
        return "你好！我是数据查询助手，可以帮你查询设备信息和传感器数据。"
    
    def _execute_action(self, action: str, action_input: Dict) -> Any:
        """Execute action."""
        try:
            if action in {"get_sensor_data", "detect_data_types"}:
                action_input = self._hydrate_confirmed_scope_from_query(action_input)
            if action == "search_devices":
                result = self._action_search_devices(action_input)
            elif action == "get_sensor_data":
                result = self._action_get_sensor_data(action_input)
            elif action == "detect_data_types":
                result = self._action_detect_data_types(action_input)
            elif action == "list_projects":
                result = self._action_list_projects()
            elif action == "get_project_stats":
                result = self._action_get_project_stats()
            else:
                result = {"error": f"Unknown action: {action}"}
            return self._enrich_action_result_query_info(action, action_input, result)
        except Exception as e:
            logger.error(f"Action {action} failed: {e}")
            return {"error": str(e)}

    def _hydrate_confirmed_scope_from_query(self, action_input: Dict) -> Dict:
        hydrated = dict(action_input or {})
        if hydrated.get("tg_values"):
            return hydrated

        query_text = str(hydrated.get("user_query") or "").strip()
        if not query_text:
            return hydrated

        plan = self._get_query_plan(query_text)
        explicit_codes = list(plan.explicit_device_codes)
        if not explicit_codes:
            explicit_codes = list(hydrated.get("device_codes") or [])
        if not explicit_codes:
            return hydrated

        resolution = self._resolve_explicit_device_scope(explicit_codes, query_text)
        if not resolution.get("complete"):
            return hydrated

        resolved_devices = list(resolution.get("resolved_devices") or [])
        hydrated["device_codes"] = explicit_codes
        hydrated["tg_values"] = self._extract_tg_values_from_devices(resolved_devices)
        return hydrated
    
    def _normalize_action_input(self, action_input: Dict, action: str, state: AgentState) -> Dict:
        """Normalize action_input to preserve explicit device codes and comparison intent."""
        normalized_input = dict(action_input or {})
        query_text = self._get_current_question_text(state.get("user_query", ""))

        if action == "search_devices":
            if query_text:
                normalized_input["user_query"] = query_text
                normalized_input.setdefault("query_plan", self._serialize_query_plan(self._get_query_plan(query_text)))
            if query_text and self._has_comparison_intent(query_text):
                normalized_input["comparison_mode"] = True
            explicit_device_codes = self._extract_explicit_device_codes(query_text)
            if explicit_device_codes and not normalized_input.get("keywords"):
                normalized_input["keywords"] = explicit_device_codes
            elif query_text and not normalized_input.get("keywords"):
                cached_device_codes = self._get_cached_device_codes_from_query(query_text)
                if cached_device_codes:
                    normalized_input["keywords"] = cached_device_codes
            return normalized_input

        if action not in {"get_sensor_data", "detect_data_types"}:
            return normalized_input

        if action == "get_sensor_data" and query_text:
            normalized_input["user_query"] = query_text
            normalized_input.setdefault("query_plan", self._serialize_query_plan(self._get_query_plan(query_text)))
            resolved_time_range = self._resolve_time_range_from_query(query_text)
            if resolved_time_range:
                previous_start = normalized_input.get("start_time")
                previous_end = normalized_input.get("end_time")
                normalized_input["start_time"] = resolved_time_range["start_time"]
                normalized_input["end_time"] = resolved_time_range["end_time"]
                if previous_start != normalized_input["start_time"] or previous_end != normalized_input["end_time"]:
                    logger.info(
                        "Resolved time range from query: %s -> %s ~ %s (previous=%s ~ %s)",
                        query_text,
                        normalized_input["start_time"],
                        normalized_input["end_time"],
                        previous_start,
                        previous_end,
                    )

        explicit_device_codes = self._extract_explicit_device_codes(query_text)
        if not explicit_device_codes:
            cached_device_codes = self._get_cached_device_codes_from_query(query_text)
            if cached_device_codes:
                normalized_input["device_codes"] = cached_device_codes
                if (
                    action == "get_sensor_data"
                    and len(cached_device_codes) > 1
                    and self._has_comparison_intent(query_text)
                    and not self._has_pagination_intent(query_text)
                ):
                    normalized_input["page"] = 1
                    normalized_input["page_size"] = 0
            return normalized_input

        current_device_codes = normalized_input.get("device_codes", [])
        if isinstance(current_device_codes, str):
            current_device_codes = [current_device_codes]
        current_device_codes = [
            code.strip()
            for code in current_device_codes
            if isinstance(code, str) and code.strip()
        ]

        if explicit_device_codes != current_device_codes:
            logger.info(
                "Explicit device codes detected, overriding LLM device_codes: %s -> %s",
                current_device_codes,
                explicit_device_codes,
            )
            normalized_input["device_codes"] = explicit_device_codes

        if (
            action == "get_sensor_data"
            and len(explicit_device_codes) > 1
            and self._has_comparison_intent(query_text)
            and not self._has_pagination_intent(query_text)
        ):
            normalized_input["page"] = 1
            normalized_input["page_size"] = 0

        return normalized_input

    def _get_current_question_text(self, user_query: str) -> str:
        """Extract the current question body from history-augmented input."""
        return extract_current_question_text(user_query)

    def _init_state(self, user_query: str) -> AgentState:
        query_text = self._get_current_question_text(str(user_query or ""))
        return {
            "user_query": query_text,
            "rewritten_query": None,
            "intent": "",
            "keywords": None,
            "device_codes": None,
            "time_range": None,
            "metadata_result": None,
            "raw_data": None,
            "final_response": None,
            "error": None,
            "iteration": 1,
            "history": [],
        }

    def _get_query_plan(self, user_query: str) -> QueryPlan:
        query_text = self._get_current_question_text(user_query)
        cache_key = query_text.strip()
        if not cache_key:
            return self.query_planner.plan(user_query)
        cached = self._query_plan_cache.get(cache_key)
        if cached is not None:
            return cached
        plan = self.query_planner.plan(query_text)
        self._query_plan_cache[cache_key] = plan
        logger.info(
            "query.plan source=%s mode=%s data_type=%s targets=%s explicit=%s confidence=%.2f",
            plan.source,
            plan.query_mode,
            plan.inferred_data_type,
            list(plan.search_targets),
            list(plan.explicit_device_codes),
            plan.confidence,
        )
        return plan

    def _serialize_query_plan(self, plan: QueryPlan) -> Dict[str, Any]:
        return plan.to_dict()

    def _build_query_state_snapshot(
        self,
        user_query: str,
        query_plan: Optional[Dict[str, Any]] = None,
        extra_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        current_question = self._get_current_question_text(str(user_query or ""))
        plan = coerce_query_plan(query_plan)
        if plan is None and current_question:
            plan = self._get_query_plan(current_question)
        if plan is not None:
            state["query_plan"] = self._serialize_query_plan(plan)
        if isinstance(extra_intent, dict) and extra_intent:
            state["intent"] = dict(extra_intent)
        return state

    def _build_query_plan_context(
        self,
        user_query: str,
        query_plan: Optional[Dict[str, Any]] = None,
        extra_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_query_plan_context_from_state(
            self._build_query_state_snapshot(
                user_query=user_query,
                query_plan=query_plan,
                extra_intent=extra_intent,
            )
        )

    def _build_compat_intent(
        self,
        user_query: str,
        query_plan: Optional[Dict[str, Any]] = None,
        extra_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return build_compat_intent_from_state(
            self._build_query_state_snapshot(
                user_query=user_query,
                query_plan=query_plan,
                extra_intent=extra_intent,
            )
        )

    def _enrich_query_info_with_query_plan_context(
        self,
        query_info: Optional[Dict[str, Any]],
        *,
        user_query: str,
        query_plan: Optional[Dict[str, Any]] = None,
        extra_intent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        enriched = dict(query_info or {})
        enriched["query_plan_context"] = self._build_query_plan_context(
            user_query=user_query,
            query_plan=query_plan,
            extra_intent=extra_intent,
        )
        return enriched

    def _build_action_override_context(self, action: str, action_input: Dict[str, Any], state: AgentState) -> ActionOverrideContext:
        query_text = self._get_current_question_text(state.get("user_query", ""))
        query_plan = action_input.get("query_plan") if isinstance(action_input, dict) else None
        query_state = self._build_query_state_snapshot(query_text, query_plan=query_plan)
        history_actions = tuple(
            str(item.get("action") or "")
            for item in state.get("history", [])
            if str(item.get("action") or "")
        )
        cached_device_codes = tuple(self._get_cached_device_codes_from_query(query_text))

        preferred_device_codes: tuple[str, ...] = ()
        preferred_tg_values: tuple[str, ...] = ()
        preferred_source = ""
        resolved_time_range = None

        if query_text:
            plan = self._get_query_plan(query_text)
            device_codes, resolved_devices, source = self._resolve_preferred_device_scope(query_text, plan)
            preferred_device_codes = tuple(device_codes)
            preferred_tg_values = tuple(self._extract_tg_values_from_devices(resolved_devices))
            preferred_source = source
            resolved_time_range = self._resolve_time_range_from_query(query_text)

        return ActionOverrideContext(
            query_state=query_state,
            action=action,
            action_input=dict(action_input or {}),
            history_actions=history_actions,
            has_cached_device_codes=bool(cached_device_codes),
            preferred_device_codes=preferred_device_codes,
            preferred_tg_values=preferred_tg_values,
            preferred_source=preferred_source,
            resolved_time_range=resolved_time_range,
        )

    def _apply_action_override_policy(self, action: str, action_input: Dict, state: AgentState):
        decision = apply_action_override_policy(
            self._build_action_override_context(action, action_input, state)
        )
        return decision.action, decision.action_input, decision.reason

    def _build_action_intent_hints(self, action: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action_input, dict):
            return {}

        def _normalize_list(value: Any) -> List[str]:
            if value is None:
                return []
            raw_items = value if isinstance(value, list) else [value]
            results: List[str] = []
            for item in raw_items:
                text = str(item or "").strip()
                if text:
                    results.append(text)
            return results

        raw_intent: Dict[str, Any] = {}
        if action == "search_devices":
            search_targets = _normalize_list(action_input.get("keywords"))
            comparison_targets = search_targets if bool(action_input.get("comparison_mode")) and len(search_targets) > 1 else []
            raw_intent = {
                "target": " vs ".join(search_targets) if len(search_targets) > 1 else (search_targets[0] if search_targets else ""),
                "search_targets": search_targets,
                "comparison_targets": comparison_targets,
                "query_mode": "comparison" if comparison_targets else "device_listing",
                "response_style": "list",
                "is_comparison": bool(comparison_targets),
            }
        elif action in {"get_sensor_data", "detect_data_types"}:
            device_codes = _normalize_list(action_input.get("device_codes"))
            comparison_targets = device_codes if len(device_codes) > 1 else []
            raw_intent = {
                "target": " vs ".join(device_codes) if len(device_codes) > 1 else (device_codes[0] if device_codes else ""),
                "search_targets": device_codes,
                "comparison_targets": comparison_targets,
                "query_mode": "detect_data_types" if action == "detect_data_types" else ("comparison" if comparison_targets else "sensor_query"),
                "response_style": "list" if action == "detect_data_types" else "structured_analysis",
                "is_comparison": bool(comparison_targets),
            }
            if action == "get_sensor_data":
                raw_intent["data_type"] = action_input.get("data_type") or "ep"
                raw_intent["time_start"] = action_input.get("start_time")
                raw_intent["time_end"] = action_input.get("end_time")
        else:
            return {}

        query_plan = action_input.get("query_plan") if isinstance(action_input, dict) else None
        if query_plan:
            raw_intent.pop("query_mode", None)
            raw_intent.pop("response_style", None)

        return self._build_compat_intent(
            user_query=str(action_input.get("user_query") or ""),
            query_plan=query_plan,
            extra_intent=raw_intent,
        )

    def _enrich_action_result_query_info(self, action: str, action_input: Dict[str, Any], result: Any) -> Any:
        if not isinstance(result, dict):
            return result
        if action not in {"search_devices", "get_sensor_data", "detect_data_types"} and result.get("query_info") is None:
            return result

        enriched = dict(result)
        enriched["query_info"] = self._enrich_query_info_with_query_plan_context(
            enriched.get("query_info"),
            user_query=str(action_input.get("user_query") or ""),
            query_plan=action_input.get("query_plan") if isinstance(action_input, dict) else None,
            extra_intent=self._build_action_intent_hints(action, action_input),
        )
        return enriched

    def _extract_explicit_device_codes(self, user_query: str) -> List[str]:
        """Extract explicit device codes in their original order, deduplicated."""
        return get_explicit_device_codes_from_state(self._build_query_state_snapshot(user_query))

    def _has_comparison_intent(self, user_query: str) -> bool:
        return is_comparison_query(self._build_query_state_snapshot(user_query))

    def _has_pagination_intent(self, user_query: str) -> bool:
        return has_pagination_intent_from_state(self._build_query_state_snapshot(user_query))

    def _has_sensor_query_intent(self, user_query: str) -> bool:
        return has_sensor_query_intent_from_state(self._build_query_state_snapshot(user_query))

    def _has_project_listing_intent(self, user_query: str) -> bool:
        return has_project_listing_intent_from_state(self._build_query_state_snapshot(user_query))

    def _has_project_stats_intent(self, user_query: str) -> bool:
        return has_project_stats_intent_from_state(self._build_query_state_snapshot(user_query))

    def _infer_data_type_from_query(self, user_query: str) -> str:
        return get_data_type_from_state(self._build_query_state_snapshot(user_query), default="ep")

    def _has_detect_data_types_intent(self, user_query: str) -> bool:
        return has_detect_data_types_intent_from_state(self._build_query_state_snapshot(user_query))

    def _build_heuristic_action(self, state: AgentState, fast_path: bool = False) -> Optional[Dict]:
        query_text = self._get_current_question_text(state.get("user_query", ""))
        if not query_text:
            return None

        plan = self._get_query_plan(query_text)

        if plan.has_project_listing_intent:
            result = {
                "thought": "\u89c4\u5219\u8bc6\u522b\u4e3a\u9879\u76ee\u5217\u8868\u67e5\u8be2",
                "action": "list_projects",
                "action_input": {},
            }
            result["_heuristic_reason"] = "fast_path_project_list" if fast_path else "heuristic_project_list"
            return result

        if plan.has_project_stats_intent:
            result = {
                "thought": "\u89c4\u5219\u8bc6\u522b\u4e3a\u9879\u76ee\u5217\u8868\u67e5\u8be2",
                "action": "get_project_stats",
                "action_input": {},
            }
            result["_heuristic_reason"] = "fast_path_project_stats" if fast_path else "heuristic_project_stats"
            return result

        device_codes, resolved_devices, source = self._resolve_preferred_device_scope(query_text, plan)

        if device_codes:
            if plan.has_detect_data_types_intent:
                result = {
                    "thought": f"\u89c4\u5219\u8bc6\u522b\u4e3a{source}\u8bbe\u5907\u6570\u636e\u7c7b\u578b\u63a2\u6d4b",
                    "action": "detect_data_types",
                    "action_input": {
                        "device_codes": device_codes,
                        "tg_values": self._extract_tg_values_from_devices(resolved_devices),
                        "user_query": query_text,
                    },
                }
                result["_heuristic_reason"] = f"fast_path_{source}_detect_data_types" if fast_path else f"heuristic_{source}_detect_data_types"
                return result

            if not plan.has_sensor_intent:
                return None

            action_input = {
                "device_codes": device_codes,
                "tg_values": self._extract_tg_values_from_devices(resolved_devices),
                "data_type": plan.inferred_data_type or self._infer_data_type_from_query(query_text),
                "query_plan": self._serialize_query_plan(plan),
                "user_query": query_text,
                "page": 1,
            }
            if len(device_codes) > 1 and plan.has_comparison_intent and not plan.has_pagination_intent:
                action_input["page_size"] = 0
            result = {
                "thought": f"\u89c4\u5219\u8bc6\u522b\u4e3a{source}\u8bbe\u5907\u65f6\u5e8f\u67e5\u8be2",
                "action": "get_sensor_data",
                "action_input": action_input,
            }
            result["_heuristic_reason"] = f"fast_path_{source}_sensor_query" if fast_path else f"heuristic_{source}_sensor_query"
            return result

        if plan.has_sensor_intent and plan.search_targets:
            result = {
                "thought": "\u89c4\u5219\u8bc6\u522b\u4e3a\u901a\u7528\u5b9e\u4f53\u68c0\u7d22\u540e\u7eed\u67e5\u6570\u573a\u666f",
                "action": "search_devices",
                "action_input": {
                    "keywords": list(plan.search_targets),
                    "comparison_mode": plan.has_comparison_intent,
                    "user_query": query_text,
                    "query_plan": self._serialize_query_plan(plan),
                },
            }
            result["_heuristic_reason"] = "fast_path_generic_sensor_search" if fast_path else "heuristic_generic_sensor_search"
            return result

        return None

    def _maybe_force_metadata_action(self, action: str, action_input: Dict, state: AgentState):
        decision = decide_metadata_override(
            self._build_action_override_context(action, action_input, state)
        )
        if decision is None:
            return action, action_input, None
        return decision.action, decision.action_input, decision.reason

    def _maybe_force_sensor_action(self, action: str, action_input: Dict, state: AgentState):
        decision = decide_sensor_override(
            self._build_action_override_context(action, action_input, state)
        )
        if decision is None:
            return action, action_input, None
        return decision.action, decision.action_input, decision.reason

    def _normalize_search_keyword(self, keyword: str) -> str:
        return normalize_search_target(keyword)

    def _dedupe_keywords(self, keywords: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for keyword in keywords:
            normalized = self._normalize_search_keyword(keyword)
            if not normalized:
                continue
            lookup = normalized.lower()
            if lookup in seen:
                continue
            seen.add(lookup)
            result.append(normalized)
        return result

    def _extract_search_targets(self, keywords: List[str], user_query: str, comparison_mode: bool) -> List[str]:
        normalized_keywords = self._dedupe_keywords(keywords)
        query_state = self._build_query_state_snapshot(user_query)
        parsed_targets = self._dedupe_keywords(get_state_targets(query_state))
        comparison_targets = [
            self._normalize_search_keyword(keyword)
            for keyword in get_comparison_targets_from_state(query_state)
            if self._normalize_search_keyword(keyword)
        ]

        if comparison_mode and comparison_targets:
            return comparison_targets

        if parsed_targets:
            if comparison_mode:
                return parsed_targets
            return self._dedupe_keywords(parsed_targets + normalized_keywords)

        return normalized_keywords

    def _is_explicit_device_code_keyword(self, keyword: str) -> bool:
        return bool(re.fullmatch(r'[a-zA-Z]\d*_[a-zA-Z0-9_]+', str(keyword or '').strip()))

    def _is_device_listing_intent(self, user_query: str) -> bool:
        return has_device_listing_intent_from_state(self._build_query_state_snapshot(user_query))

    def _normalize_context_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).lower()

    def _get_query_context_terms(self, keyword: str, user_query: str) -> List[str]:
        normalized_keyword = self._normalize_context_text(keyword)
        terms: List[str] = []
        seen = set()
        for term in get_project_hints_from_state(self._build_query_state_snapshot(user_query)):
            normalized_term = self._normalize_context_text(term)
            if not normalized_term or normalized_term == normalized_keyword or normalized_term in seen:
                continue
            seen.add(normalized_term)
            terms.append(term)
        return terms

    def _score_context_overlap(
        self,
        term: str,
        value: Any,
        *,
        exact_weight: float,
        contains_weight: float,
        token_weight: float,
    ) -> float:
        normalized_term = self._normalize_context_text(term)
        normalized_value = self._normalize_context_text(value)
        if not normalized_term or not normalized_value:
            return 0.0
        if normalized_term == normalized_value:
            return exact_weight
        if normalized_term in normalized_value or normalized_value in normalized_term:
            return contains_weight
        tokens = [token for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", normalized_term) if token]
        if tokens and all(token in normalized_value for token in tokens):
            return token_weight
        return 0.0

    def _rerank_devices_with_query_context(self, devices: List[Dict], keyword: str, user_query: str) -> List[Dict]:
        if not devices or not user_query:
            return devices

        context_terms = self._get_query_context_terms(keyword, user_query)
        if not context_terms:
            return devices

        reranked_devices: List[Dict] = []
        for device in devices:
            context_bonus = 0.0
            matched_terms: List[str] = []

            for term in context_terms:
                term_bonus = max(
                    self._score_context_overlap(term, device.get("project_name"), exact_weight=42.0, contains_weight=36.0, token_weight=28.0),
                    self._score_context_overlap(term, device.get("project_code_name"), exact_weight=38.0, contains_weight=32.0, token_weight=24.0),
                    self._score_context_overlap(term, device.get("name"), exact_weight=14.0, contains_weight=10.0, token_weight=8.0),
                    self._score_context_overlap(term, device.get("device_type"), exact_weight=10.0, contains_weight=8.0, token_weight=6.0),
                )
                if term_bonus > 0:
                    context_bonus += term_bonus
                    if term not in matched_terms:
                        matched_terms.append(term)

            updated_device = dict(device)
            if context_bonus > 0:
                updated_device["context_score"] = round(context_bonus, 2)
                updated_device["match_score"] = round(float(device.get("match_score") or 0.0) + context_bonus, 2)
                matched_fields = list(updated_device.get("matched_fields") or [])
                if "query_context" not in matched_fields:
                    matched_fields.append("query_context")
                updated_device["matched_fields"] = matched_fields

                context_reason = f"上下文命中: {'、'.join(matched_terms)}"
                original_reason = str(updated_device.get("match_reason") or "").strip()
                updated_device["match_reason"] = f"{original_reason} | {context_reason}" if original_reason else context_reason

            reranked_devices.append(updated_device)

        reranked_devices.sort(
            key=lambda item: (
                -float(item.get("match_score") or 0.0),
                -float(item.get("context_score") or 0.0),
                str(item.get("device") or ""),
                str(item.get("name") or ""),
            )
        )
        return reranked_devices

    def _search_device_candidates(self, keyword: str, user_query: str = ""):
        cached_device = self._get_alias_memory_entry(keyword)
        if cached_device:
            logger.info("device.search alias_cache_hit keyword=%s device=%s", keyword, cached_device.get("device"))
            return [cached_device], {"type": "session_alias", "alias": keyword, "device": cached_device.get("device")}

        if self._is_explicit_device_code_keyword(keyword):
            exact_candidates, query_info = self._lookup_exact_device_code_candidates(keyword, user_query)
            if exact_candidates:
                logger.info(
                    "device.search exact_code_hit keyword=%s count=%s top_device=%s",
                    keyword,
                    len(exact_candidates),
                    exact_candidates[0].get("device"),
                )
                return exact_candidates, query_info

        if self.entity_resolver is not None:
            try:
                resolution_result = self.entity_resolver.search_device_candidates(keyword, top_k=CLARIFICATION_CANDIDATE_LIMIT)
                resolved_devices = self._rerank_devices_with_query_context(
                    resolution_result.to_dict_list(),
                    keyword,
                    user_query,
                )
                if resolved_devices:
                    query_info = dict(resolution_result.query_info or {})
                    context_terms = self._get_query_context_terms(keyword, user_query)
                    if context_terms:
                        query_info["context_terms"] = context_terms
                        query_info["context_applied"] = True
                    logger.info(
                        "device.search entity_resolver_hit keyword=%s count=%s top_device=%s",
                        keyword,
                        len(resolved_devices),
                        resolved_devices[0].get("device"),
                    )
                    return resolved_devices, query_info
            except Exception as exc:
                logger.warning("device.search entity_resolver_failed keyword=%s error=%s", keyword, exc)

        if not self.metadata_engine:
            return [], None
        devices = find_device_metadata_with_engine(keyword, self.metadata_engine)
        result_devices = []
        query_info = None
        for device in devices:
            if "_query_info" in device:
                query_info = device["_query_info"]
                continue
            if "error" not in device:
                result_devices.append(device)
        result_devices = self._rerank_devices_with_query_context(result_devices, keyword, user_query)
        result_devices.sort(
            key=lambda item: (
                -float(item.get("match_score") or 0.0),
                str(item.get("device") or ""),
                str(item.get("name") or ""),
            )
        )
        if query_info is not None:
            context_terms = self._get_query_context_terms(keyword, user_query)
            if context_terms:
                query_info = dict(query_info)
                query_info["context_terms"] = context_terms
                query_info["context_applied"] = True
        return result_devices, query_info

    def _mark_exact_code_conflict_candidates(self, device_code: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_code = self._normalize_alias_key(device_code)
        marked: List[Dict[str, Any]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            current_code = self._normalize_alias_key(candidate.get("device"))
            if current_code == normalized_code:
                updated = dict(candidate)
                updated["match_type"] = "exact_code_conflict"
                updated["match_score"] = None
                updated["match_reason"] = "\u7cbe\u786e\u7801\u547d\u4e2d"
                marked.append(updated)
            else:
                marked.append(dict(candidate))
        return marked

    def _select_clarification_candidates(self, devices: List[Dict], limit: int = CLARIFICATION_CANDIDATE_LIMIT) -> List[Dict]:
        candidates: List[Dict] = []
        seen = set()
        for device in devices:
            dedupe_key = (device.get("device"), device.get("project_id"), device.get("name"))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(
                {
                    "device": device.get("device"),
                    "name": device.get("name"),
                    "project_id": device.get("project_id"),
                    "project_name": device.get("project_name"),
                    "project_code_name": device.get("project_code_name"),
                    "tg": device.get("tg"),
                    "match_score": round(float(device.get("match_score") or 0.0), 2),
                    "matched_fields": list(device.get("matched_fields") or []),
                    "match_reason": device.get("match_reason"),
                    "match_type": device.get("match_type"),
                    "scope_mode": device.get("scope_mode"),
                }
            )
            if len(candidates) >= limit:
                break
        return candidates

    def _build_aggregate_scope_candidate(self, keyword: str, candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        device_codes = {str(item.get("device") or "").strip() for item in candidates if str(item.get("device") or "").strip()}
        if len(device_codes) != 1:
            return None
        device_code = next(iter(device_codes))
        scopes = {
            (
                str(item.get("project_id") or ""),
                str(item.get("project_name") or item.get("project_code_name") or ""),
                str(item.get("tg") or ""),
            )
            for item in candidates
        }
        if len(scopes) <= 1:
            return None
        return {
            "device": device_code,
            "name": f"\u6240\u6709 {device_code} \u8bbe\u5907",
            "project_id": "__aggregate_all__",
            "project_name": "\u8de8\u9879\u76ee\u6c47\u603b",
            "project_code_name": None,
            "tg": None,
            "match_type": "aggregate_all_option",
            "match_score": None,
            "match_reason": "\u6c47\u603b\u8be5\u8bbe\u5907\u5728\u6240\u6709\u5339\u914d\u9879\u76ee\u4e0b\u7684\u6570\u636e",
            "matched_fields": ["aggregate_scope"],
            "scope_mode": "aggregate_all",
        }


    def _should_request_device_clarification(
        self,
        keyword: str,
        devices: List[Dict],
        user_query: str,
        comparison_mode: bool = False,
    ) -> bool:
        if self._is_explicit_device_code_keyword(keyword):
            return self._has_cross_project_collision(devices, keyword)
        if not devices or len(devices) <= 1:
            return False
        if self._is_device_listing_intent(user_query) and not comparison_mode:
            return False
        if comparison_mode:
            return True
        if self._has_sensor_query_intent(user_query):
            return True
        if not comparison_mode and not self._has_sensor_query_intent(user_query):
            return False

        top1 = float(devices[0].get("match_score") or 0.0)
        top2 = float(devices[1].get("match_score") or 0.0)
        strong_candidates = [
            device
            for device in devices
            if float(device.get("match_score") or 0.0) >= max(top1 - 8.0, 60.0)
        ]
        if len(strong_candidates) > 1:
            return True
        if top1 < 70.0 and len(devices) > 1:
            return True
        if top1 > 0 and top2 / top1 >= 0.92:
            return True
        return False

    def _build_clarification_message(self, clarification_groups: List[Dict], resolved_devices: Optional[List[Dict]] = None) -> str:
        lines: List[str] = []
        resolved_devices = resolved_devices or []
        if resolved_devices:
            resolved_parts = []
            for item in resolved_devices:
                label = item.get("keyword") or item.get("device") or "设备"
                device_code = item.get("device") or "-"
                name = item.get("name") or "未命名设备"
                resolved_parts.append(f"{label} -> {device_code}（{name}）")
            lines.append("已确认：" + "；".join(resolved_parts))

        if len(clarification_groups) == 1:
            group = clarification_groups[0]
            preview = "、".join(
                f"{candidate.get('device') or '-'}（{candidate.get('name') or '未命名设备'}）"
                for candidate in group.get("candidates", [])[:3]
            )
            lines.append(
                f"“{group.get('keyword') or '该目标'}”匹配到多个候选设备：{preview}。请确认你指的是哪一个。"
            )
        else:
            summary = "；".join(
                f"{group.get('keyword') or '目标'} 有 {len(group.get('candidates', []))} 个候选"
                for group in clarification_groups
            )
            lines.append(
                f"我找到了多个可能匹配的设备，暂时不能直接继续查询：{summary}。请先确认候选设备。"
            )
        return "\n".join(lines)

    def _action_search_devices(self, params: Dict) -> Dict:
        """Search devices with candidate scoring and clarification fallback."""
        keywords = params.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]

        user_query = params.get("user_query", "")
        comparison_mode = bool(params.get("comparison_mode"))
        search_targets = self._extract_search_targets(keywords, user_query, comparison_mode)

        logger.info(
            "device.search keywords=%s comparison_mode=%s user_query=%s",
            search_targets,
            comparison_mode,
            re.sub(r"\s+", " ", str(user_query or "")).strip()[:120],
        )

        if not search_targets:
            return {"devices": [], "count": 0, "message": "未提供有效搜索关键词"}

        if comparison_mode:
            clarification_groups: List[Dict] = []
            resolved_devices: List[Dict] = []
            query_infos: List[Dict] = []

            for keyword in search_targets:
                result_devices, query_info = self._search_device_candidates(keyword, user_query)
                if query_info:
                    query_infos.append(query_info)

                allow_multi_scope_aggregation = (
                    self._is_explicit_device_code_keyword(keyword)
                    and allows_explicit_multi_scope_aggregation(user_query, keyword)
                )
                if self._should_request_device_clarification(keyword, result_devices, user_query, comparison_mode=True) and not allow_multi_scope_aggregation:
                    exact_conflict_candidates = self._mark_exact_code_conflict_candidates(keyword, result_devices) if self._is_explicit_device_code_keyword(keyword) else result_devices
                    clarification_candidates = self._select_clarification_candidates(exact_conflict_candidates)
                    aggregate_candidate = self._build_aggregate_scope_candidate(keyword, clarification_candidates)
                    if aggregate_candidate:
                        clarification_candidates.append(aggregate_candidate)
                    clarification_groups.append(
                        {
                            "keyword": keyword,
                            "candidates": clarification_candidates,
                        }
                    )
                    continue

                if result_devices:
                    selected_devices = result_devices if allow_multi_scope_aggregation else result_devices[:1]
                    for selected_device in selected_devices:
                        best_device = dict(selected_device)
                        best_device["keyword"] = keyword
                        resolved_devices.append(best_device)

            query_info = query_infos[0] if query_infos else None
            deduped_devices = []
            seen = set()
            for device in resolved_devices:
                dedupe_key = (device.get("device"), device.get("project_id"), device.get("name"))
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                deduped_devices.append(device)

            if clarification_groups:
                return {
                    "devices": deduped_devices,
                    "resolved_devices": deduped_devices,
                    "count": len(deduped_devices),
                    "device_codes": [item.get("device") for item in deduped_devices if item.get("device")],
                    "query_info": query_info,
                    "needs_clarification": True,
                    "clarification_required": True,
                    "clarification_candidates": clarification_groups,
                    "message": self._build_clarification_message(clarification_groups, deduped_devices),
                }

            return {
                "devices": deduped_devices[:50],
                "count": len(deduped_devices),
                "device_codes": [item.get("device") for item in deduped_devices[:50] if item.get("device")],
                "query_info": query_info,
            }

        if len(search_targets) == 1:
            try:
                result_devices, query_info = self._search_device_candidates(search_targets[0], user_query)
                if self._should_request_device_clarification(search_targets[0], result_devices, user_query, comparison_mode=False):
                    clarification_groups = [
                        {
                            "keyword": search_targets[0],
                            "candidates": self._select_clarification_candidates(self._mark_exact_code_conflict_candidates(search_targets[0], result_devices) if self._is_explicit_device_code_keyword(search_targets[0]) else result_devices),
                        }
                    ]
                    return {
                        "devices": result_devices[:50],
                        "count": len(result_devices),
                        "device_codes": [item.get("device") for item in result_devices[:50] if item.get("device")],
                        "query_info": query_info,
                        "needs_clarification": True,
                        "clarification_required": True,
                        "clarification_candidates": clarification_groups,
                        "message": self._build_clarification_message(clarification_groups),
                    }

                return {
                    "devices": result_devices[:50],
                    "count": len(result_devices),
                    "device_codes": [item.get("device") for item in result_devices[:50] if item.get("device")],
                    "query_info": query_info,
                }
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return {"devices": [], "count": 0, "error": str(e)}

        keyword_results = []
        for keyword in search_targets:
            try:
                result_devices, query_info = self._search_device_candidates(keyword, user_query)
                logger.info("device.search keyword=%s count=%s", keyword, len(result_devices))
                keyword_results.append((keyword, result_devices, query_info))
            except Exception as e:
                logger.error(f"Search failed for keyword '{keyword}': {e}")
                keyword_results.append((keyword, [], None))

        non_empty_results = [(keyword, devices, query_info) for keyword, devices, query_info in keyword_results if devices]
        if not non_empty_results:
            return {"devices": [], "count": 0, "message": "未找到匹配的设备"}

        def get_key(device: Dict):
            return (device.get("device"), device.get("project_id"), device.get("name"))

        shared_keys = None
        for _, devices, _ in non_empty_results:
            device_keys = {get_key(device) for device in devices if device.get("device")}
            shared_keys = device_keys if shared_keys is None else shared_keys & device_keys

        if shared_keys:
            merged_devices = {}
            for _, devices, _ in non_empty_results:
                for device in devices:
                    key = get_key(device)
                    if key not in shared_keys:
                        continue
                    existing = merged_devices.get(key)
                    if existing is None or float(device.get("match_score") or 0.0) > float(existing.get("match_score") or 0.0):
                        merged_devices[key] = device
            final_devices = sorted(
                merged_devices.values(),
                key=lambda item: (
                    -float(item.get("match_score") or 0.0),
                    str(item.get("device") or ""),
                    str(item.get("name") or ""),
                ),
            )
            query_info = next((query_info for _, _, query_info in non_empty_results if query_info), None)
            return {
                "devices": final_devices[:50],
                "count": len(final_devices),
                "device_codes": [item.get("device") for item in final_devices[:50] if item.get("device")],
                "query_info": query_info,
            }

        fallback_keyword, fallback_devices, fallback_query_info = min(
            non_empty_results,
            key=lambda item: (len(item[1]), -float(item[1][0].get("match_score") or 0.0)),
        )
        failed_keywords = [keyword for keyword, devices, _ in keyword_results if not devices]
        logger.info("device.search fallback_keyword=%s count=%s", fallback_keyword, len(fallback_devices))
        return {
            "devices": fallback_devices[:50],
            "count": len(fallback_devices),
            "device_codes": [item.get("device") for item in fallback_devices[:50] if item.get("device")],
            "query_info": fallback_query_info,
            "fallback": True,
            "fallback_keyword": fallback_keyword,
            "failed_keywords": failed_keywords,
            "message": f"关键词组合未精确匹配，已返回“{fallback_keyword}”的候选设备。",
        }

    def _build_explicit_code_clarification_result(self, device_codes: List[str], user_query: str) -> Optional[Dict]:
        if not device_codes:
            return None

        resolution = self._resolve_explicit_device_scope(device_codes, user_query)
        clarification_groups = list(resolution.get("clarification_groups") or [])
        if not clarification_groups:
            return None

        resolved_devices = list(resolution.get("resolved_devices") or [])
        query_infos = list(resolution.get("query_infos") or [])
        query_info = query_infos[0] if query_infos else None
        return {
            "devices": resolved_devices,
            "resolved_devices": resolved_devices,
            "count": len(resolved_devices),
            "device_codes": [item.get("device") for item in resolved_devices if item.get("device")],
            "query_info": query_info,
            "needs_clarification": True,
            "clarification_required": True,
            "clarification_candidates": clarification_groups,
            "message": self._build_clarification_message(clarification_groups, resolved_devices),
        }

    def _action_get_sensor_data(self, params: Dict) -> Dict:
        """Fetch time series data."""
        device_codes = params.get("device_codes", [])
        if isinstance(device_codes, str):
            device_codes = [device_codes]

        tg_values = params.get("tg_values") or []
        if isinstance(tg_values, str):
            tg_values = [tg_values]

        start_time = params.get("start_time")
        end_time = params.get("end_time")
        data_type = params.get("data_type", "ep")
        page = params.get("page", 1)
        page_size = params.get("page_size", 0)
        value_filter = params.get("value_filter")
        user_query = params.get("user_query", "")
        query_plan = params.get("query_plan")

        if not tg_values:
            clarification_result = self._build_explicit_code_clarification_result(device_codes, user_query)
            if clarification_result is not None:
                logger.info("orchestrator.sensor.explicit_code_clarification devices=%s", device_codes)
                return clarification_result

        if start_time and " " in str(start_time):
            start_time = str(start_time).split(" ")[0]
        if end_time and " " in str(end_time):
            end_time = str(end_time).split(" ")[0]

        resolved_time_range = self._resolve_time_range_from_query(user_query)
        if resolved_time_range:
            start_time = resolved_time_range["start_time"]
            end_time = resolved_time_range["end_time"]
        elif not start_time or not end_time:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)
            start_time = start_time or start_dt.strftime("%Y-%m-%d")
            end_time = end_time or end_dt.strftime("%Y-%m-%d")

        request_log = {
            "event": "orchestrator.sensor.request",
            "device_codes": device_codes[:8],
            "device_count": len(device_codes),
            "tg_values": tg_values[:8],
            "tg_count": len(tg_values),
            "start_time": start_time,
            "end_time": end_time,
            "data_type": data_type,
            "page": page,
            "page_size": page_size,
            "value_filter": value_filter,
            "user_query": re.sub(r"\s+", " ", str(user_query or "")).strip()[:120],
        }
        logger.info("%s", json.dumps(request_log, ensure_ascii=False, default=str))

        if not device_codes:
            return {"error": "未指定设备代号", "success": False}

        try:
            if self.data_fetcher:
                result = fetch_sensor_data_with_components(
                    device_codes=device_codes,
                    tg_values=tg_values,
                    start_time=start_time,
                    end_time=end_time,
                    data_fetcher=self.data_fetcher,
                    cache_manager=self.cache_manager,
                    compressor=self.compressor,
                    output_format="json",
                    data_type=data_type,
                    page=page,
                    page_size=page_size,
                    user_query=user_query,
                    query_plan=query_plan,
                    value_filter=value_filter,
                )
                analysis = result.get("analysis") or {}
                chart_specs = result.get("chart_specs") or []
                logger.info(
                    "%s",
                    json.dumps(
                        {
                            "event": "orchestrator.sensor.result",
                            "device_codes": device_codes[:8],
                            "device_count": len(device_codes),
                            "success": result.get("success"),
                            "total_count": result.get("total_count"),
                            "page": result.get("page"),
                            "page_size": result.get("page_size"),
                            "total_pages": result.get("total_pages"),
                            "has_more": result.get("has_more"),
                            "analysis_mode": analysis.get("mode") if isinstance(analysis, dict) else None,
                            "chart_count": len(chart_specs) if isinstance(chart_specs, list) else 0,
                            "error": result.get("error"),
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                )
                return result
            else:
                return {"error": "数据获取器未配置", "success": False}
        except Exception as e:
            logger.exception(
                "orchestrator.sensor.error %s",
                json.dumps({**request_log, "event": "orchestrator.sensor.error", "error": str(e)}, ensure_ascii=False, default=str),
            )
            return {"error": str(e), "success": False}

    def _action_list_projects(self) -> Dict:
        """列出所有项目"""
        try:
            if self.metadata_engine:
                projects = self.metadata_engine.list_projects()
                return {
                    "projects": projects,
                    "count": len(projects)
                }
            else:
                return {"error": "元数据引擎未配置", "projects": []}
        except Exception as e:
            return {"error": str(e), "projects": []}
    
    def _action_get_project_stats(self) -> Dict:
        """获取各项目的设备数量统计"""
        try:
            if self.metadata_engine:
                stats = self.metadata_engine.get_project_device_stats()
                return {
                    "stats": stats,
                    "count": len(stats),
                    "success": True
                }
            else:
                return {"error": "元数据引擎未配置", "stats": [], "success": False}
        except Exception as e:
            return {"error": str(e), "stats": [], "success": False}
    
    def _action_detect_data_types(self, params: Dict) -> Dict:
        """探测设备有哪些数据类型"""
        device_codes = params.get("device_codes", [])
        if isinstance(device_codes, str):
            device_codes = [device_codes]
        tg_values = params.get("tg_values") or []
        if isinstance(tg_values, str):
            tg_values = [tg_values]
        user_query = str(params.get("user_query") or "")

        if not device_codes:
            return {"error": "未指定设备代号", "success": False}

        if not tg_values:
            clarification_result = self._build_explicit_code_clarification_result(device_codes, user_query)
            if clarification_result is not None:
                logger.info("detect.data_types.explicit_code_clarification devices=%s", device_codes)
                return clarification_result

        try:
            if self.data_fetcher:
                result = detect_device_data_types(
                    device_codes=device_codes,
                    data_fetcher=self.data_fetcher,
                    tg_values=tg_values,
                )
                logger.info("detect.data_types devices=%s summary=%s", device_codes, result.get("summary", "无数据"))
                return result
            else:
                return {"error": "数据获取器未配置", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def _is_empty_sensor_result(self, result: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(result, dict) or not result.get("success"):
            return False

        total_count = result.get("total_count")
        if isinstance(total_count, int):
            return total_count == 0

        data = result.get("data")
        if data in (None, "", [], {}, ()):
            return True
        if isinstance(data, str) and data.strip() in {"", "[]", "{}"}:
            return True
        return False

    def _build_sensor_terminal_response(self, result: Optional[Dict[str, Any]], fallback_answer: str = "") -> str:
        if self._is_empty_sensor_result(result):
            return EMPTY_RESULT_MESSAGE
        focused_response = self._build_focused_sensor_response(result)
        if focused_response:
            return focused_response
        if isinstance(result, dict) and result.get("analysis"):
            return self._format_structured_analysis_response(result, fallback_answer)
        if isinstance(result, dict) and result.get("success"):
            total_count = result.get("total_count")
            if isinstance(total_count, int):
                return f"已获取 {total_count} 条数据记录。"
            return fallback_answer or "已获取查询结果。"
        return fallback_answer or "未获取到可用结果。"

    def _format_metric_value(self, value: Any, unit: str = "") -> str:
        return format_metric_value(value, unit)

    def _build_focused_sensor_response(self, result: Optional[Dict[str, Any]]) -> str:
        if not isinstance(result, dict):
            return ""
        focused = result.get("focused_result")
        if not isinstance(focused, dict):
            return ""
        return build_focused_sensor_response(focused, total_count=int(result.get("total_count") or 0))

    
    def _build_chart_follow_up(self, analysis: Dict[str, Any], chart_specs: List[Dict[str, Any]], show_charts: bool) -> str:
        if not chart_specs:
            return ""
        first_chart = chart_specs[0] if isinstance(chart_specs, list) and chart_specs else {}
        chart_type = str(first_chart.get("chart_type") or "").strip().lower()
        chart_label = CHART_TYPE_LABELS.get(chart_type) or str(first_chart.get("title") or "").strip() or ("对比图" if analysis.get("mode") == "comparison" else "趋势图")
        if show_charts:
            return f"- 已按你的要求准备了{chart_label}，可结合下方图表继续查看。"
        return f"- 如果你需要，我可以继续帮你画一份{chart_label}。"

    def _format_structured_analysis_response(self, result: Dict[str, Any], fallback_answer: str = "") -> str:
        analysis = result.get("analysis") or {}
        if not analysis:
            return fallback_answer or "暂无可用分析结果。"

        stats = result.get("statistics") or {}
        unit = analysis.get("unit", "")
        total_count = result.get("total_count", 0)
        show_charts = bool(result.get("show_charts"))
        chart_specs = result.get("chart_specs") or []
        lines: List[str] = []

        lines.append("【结论】")
        lines.append(analysis.get("headline") or fallback_answer or "已完成本次数据分析。")
        lines.append("")

        if analysis.get("mode") == "comparison":
            rankings = analysis.get("rankings", {})
            devices = analysis.get("devices", [])

            def ranking_text(items: List[Dict[str, Any]], unit_text: str = "", suffix: str = "") -> str:
                if not items:
                    return "-"
                rank_display_limit = 8
                visible_items = items if len(items) <= rank_display_limit else items[:rank_display_limit]
                content = " > ".join(
                    f"{item.get('name')}({self._format_metric_value(item.get('value'), unit_text)})"
                    for item in visible_items
                )
                if len(items) > rank_display_limit:
                    content += f" > ...共 {len(items)} 项"
                return content + suffix

            lines.append("【对比排名】")
            lines.append(f"- 均值排名: {ranking_text(rankings.get('avg', []), unit)}")
            lines.append(f"- 总量排名: {ranking_text(rankings.get('sum', []), unit)}")
            lines.append(f"- 稳定性排名: {ranking_text(rankings.get('stability', []), '%', '（CV 越低越稳定）')}")
            lines.append("")

            lines.append("【逐项观察】")
            for device in devices[:5]:
                lines.append(
                    f"- {device.get('name')}: 趋势{device.get('trend', '平稳')}，均值{self._format_metric_value(device.get('avg'), unit)}，"
                    f"总量{self._format_metric_value(device.get('sum'), unit)}，波动CV {device.get('cv', 0)}%，异常占比 {device.get('anomaly_ratio', 0)}%。"
                )
        else:
            peak_valley = analysis.get("peak_valley") or {}
            lines.append("【关键指标】")
            lines.append(f"- 数据量: {total_count} 条")
            if stats.get("avg") is not None:
                lines.append(f"- 平均值: {self._format_metric_value(stats.get('avg'), unit)}")
            if stats.get("max") is not None:
                lines.append(f"- 最大值: {self._format_metric_value(stats.get('max'), unit)}")
            if stats.get("min") is not None:
                lines.append(f"- 最小值: {self._format_metric_value(stats.get('min'), unit)}")
            lines.append("")

            lines.append("【趋势判断】")
            lines.append(f"- 整体趋势: {stats.get('trend', '平稳')}")
            lines.append(f"- 阶段变化: {stats.get('change_rate', 0)}%")
            lines.append(f"- 波动系数: {stats.get('cv', 0)}%")
            lines.append("")

            lines.append("【异常与峰谷】")
            lines.append(f"- 异常点: {stats.get('anomaly_count', len(analysis.get('anomalies', [])))} 个")
            if peak_valley:
                lines.append(
                    f"- 峰值: {peak_valley.get('peak_time', '-')} / {self._format_metric_value(peak_valley.get('peak_value'), unit)}"
                )
                lines.append(
                    f"- 谷值: {peak_valley.get('valley_time', '-')} / {self._format_metric_value(peak_valley.get('valley_value'), unit)}"
                )
                lines.append(
                    f"- 峰谷差: {self._format_metric_value(peak_valley.get('gap'), unit)}"
                )

        insights = analysis.get("insights", [])
        if insights:
            lines.append("")
            lines.append("【关键发现】")
            for insight in insights[:3]:
                lines.append(f"- {insight}")

        chart_follow_up = self._build_chart_follow_up(analysis, chart_specs, show_charts)
        if chart_follow_up:
            lines.append("")
            lines.append("【下一步】")
            lines.append(chart_follow_up)

        return "\n".join(line for line in lines if line is not None).strip()

    def _observe(self, state: AgentState, action: str, result: Any) -> AgentState:
        """观察执行结果，更新状态"""
        if action == "search_devices":
            state["metadata_result"] = result.get("devices", [])
            state["device_codes"] = result.get("device_codes", [])
        elif action == "get_sensor_data":
            state["raw_data"] = result.get("data")
        elif action == "list_projects":
            state["metadata_result"] = result.get("projects", [])
        
        return state
    
    def _summarize_result(self, result: Any) -> str:
        """Generate short action summary."""
        if isinstance(result, dict):
            if result.get("needs_clarification"):
                candidate_count = sum(len(group.get("candidates", [])) for group in result.get("clarification_candidates", []))
                return f"找到 {candidate_count} 个候选设备，等待用户确认"
            if "error" in result:
                return f"错误: {result['error']}"
            if "count" in result:
                if "devices" in result:
                    return f"找到 {result['count']} 个设备"
                if "projects" in result:
                    return f"找到 {result['count']} 个项目"
            if "total_count" in result:
                return f"获取 {result['total_count']} 条数据记录"
            if "success" in result and not result["success"]:
                return f"失败: {result.get('error', '未知错误')}"
        return str(result)[:100]

    def _format_result_detail(self, result: Any) -> str:
        """Format detail for the model and debug logs."""
        if not isinstance(result, dict):
            return str(result)[:2000]

        lines = []

        if result.get("needs_clarification"):
            lines.append("需要用户确认以下候选设备：")
            for group in result.get("clarification_candidates", []):
                lines.append(f"- {group.get('keyword') or '目标'}")
                for index, candidate in enumerate(group.get("candidates", [])[:5], 1):
                    project_name = candidate.get("project_name") or candidate.get("project_code_name") or "未知项目"
                    reason = candidate.get("match_reason") or ", ".join(candidate.get("matched_fields") or []) or "模糊匹配"
                    score = candidate.get("match_score")
                    lines.append(
                        f"  {index}. {candidate.get('device') or '-'} | {candidate.get('name') or '未命名设备'} | {project_name} | score={score} | {reason}"
                    )
            resolved_devices = result.get("resolved_devices") or []
            if resolved_devices:
                lines.append("已确认设备：")
                for item in resolved_devices:
                    lines.append(
                        f"  - {item.get('keyword') or item.get('device') or '设备'} -> {item.get('device') or '-'} ({item.get('name') or '未命名设备'})"
                    )
            return "\n".join(lines)

        if "devices" in result:
            devices = result.get("devices", [])
            lines.append(f"找到 {len(devices)} 个设备")
            for index, device in enumerate(devices[:20], 1):
                name = device.get("name", "未知")
                device_code = device.get("device") or device.get("device_code", "未知")
                project = device.get("project_name", "未知项目")
                score = device.get("match_score")
                reason = device.get("match_reason")
                suffix = ""
                if score is not None:
                    suffix += f", 匹配分={score}"
                if reason:
                    suffix += f", 原因={reason}"
                lines.append(f"  {index}. {name} (代号: {device_code}, 项目: {project}{suffix})")
            if len(devices) > 20:
                lines.append(f"  ... 还有 {len(devices) - 20} 个设备")

        if "projects" in result:
            projects = result.get("projects", [])
            lines.append(f"共 {len(projects)} 个项目")
            for project in projects[:15]:
                name = project.get("project_name", "未知")
                pid = project.get("id", "")
                lines.append(f"  - {name} (ID: {pid})")
            if len(projects) > 15:
                lines.append(f"  ... 还有 {len(projects) - 15} 个项目")

        if "stats" in result:
            stats = result.get("stats", [])
            lines.append(f"项目设备数量统计（共 {len(stats)} 个项目）:")
            for index, stat in enumerate(stats[:10], 1):
                name = stat.get("project_name", "未知")
                count_value = stat.get("device_count", 0)
                lines.append(f"  {index}. {name}: {count_value} 个设备")
            if len(stats) > 10:
                lines.append(f"  ... 还有 {len(stats) - 10} 个项目")

        if "data" in result:
            data = result.get("data") or []
            lines.append(f"共 {len(data)} 条数据")

        if "message" in result and result.get("message"):
            lines.append(str(result.get("message")))

        return "\n".join(lines)[:4000]

    def _generate_fallback_response(self, state: AgentState) -> str:
        """生成兜底响应"""
        parts = []
        
        if state.get("metadata_result"):
            count = len(state["metadata_result"])
            parts.append(f"找到 {count} 个相关结果。")
        
        if state.get("raw_data"):
            parts.append("已获取时序数据。")
        
        if not parts:
            parts.append("抱歉，未能找到相关信息。请尝试：")
            parts.append("- 使用更具体的设备名称或代号")
            parts.append("- 指定项目名称")
            parts.append("- 输入 '有哪些项目' 查看可用项目")
        
        return "\n".join(parts)


# 兼容旧接口
class AgentOrchestrator(LLMAgent):
    """兼容旧接口的 Agent"""
    pass


class StreamingAgentOrchestrator(LLMAgent):
    """支持流式输出的 Agent"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        llm_non_streaming: BaseChatModel = None,
        **kwargs
    ):
        # 使用非流式 LLM 进行推理
        super().__init__(llm=llm_non_streaming or llm, **kwargs)
        self.llm_streaming = llm


def create_agent(
    llm: BaseChatModel,
    metadata_engine=None,
    data_fetcher=None,
    cache_manager=None,
    compressor=None,
    alias_memory=None,
    entity_resolver=None,
) -> LLMAgent:
    """创建 Agent 实例"""
    return LLMAgent(
        llm=llm,
        metadata_engine=metadata_engine,
        data_fetcher=data_fetcher,
        cache_manager=cache_manager,
        compressor=compressor,
        alias_memory=alias_memory,
        entity_resolver=entity_resolver,
    )


def create_agent_with_streaming(
    llm: BaseChatModel,
    llm_non_streaming: BaseChatModel = None,
    metadata_engine=None,
    data_fetcher=None,
    cache_manager=None,
    compressor=None,
    alias_memory=None,
    entity_resolver=None,
) -> StreamingAgentOrchestrator:
    """创建支持流式输出的 Agent"""
    return StreamingAgentOrchestrator(
        llm=llm,
        llm_non_streaming=llm_non_streaming,
        metadata_engine=metadata_engine,
        data_fetcher=data_fetcher,
        cache_manager=cache_manager,
        compressor=compressor,
        alias_memory=alias_memory,
        entity_resolver=entity_resolver,
    )
