"""
Synthesizer Node - 综合响应生成节点

优先基于 `query_plan` 和统计结果生成最终回答；当 LLM 不可用时，退化为可读的规则化摘要。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.query_plan_state import (
    get_comparison_targets_from_state,
    get_data_type_from_state,
    get_query_mode_from_state,
    get_response_style_from_state,
    get_target_label_from_state,
    get_time_range_from_state,
    is_comparison_query,
)
from src.agent.focused_response import build_focused_sensor_response
from src.agent.types import GraphState, NODE_SYNTHESIZER
from src.tools.sensor_tool import _build_focused_result


logger = logging.getLogger(__name__)


ERROR_MESSAGES = {
    "intent_parser": "抱歉，我暂时没能理解你的问题。请补充设备、指标或时间范围后再试一次。",
    "metadata_mapper": "未找到与“{target}”匹配的设备，请检查设备名称、代号或换一种说法。",
    "sharding_router": "查询时间范围过大或时间条件无效，请缩小时间范围后重试。",
    "parallel_fetcher": "数据获取失败，请稍后重试。",
    "default": "处理你的请求时遇到问题，请稍后重试。",
}

DATA_TYPE_UNITS = {
    "ep": "kWh",
    "i": "A",
    "u_line": "V",
    "u": "V",
    "ua": "V",
    "ub": "V",
    "uc": "V",
    "p": "kW",
    "qf": "",
    "t": "°C",
    "sd": "%",
    "f": "Hz",
    "soc": "%",
    "gffddl": "kWh",
    "loadrate": "%",
}

DATA_TYPE_NAMES = {
    "ep": "用电量",
    "i": "电流",
    "u_line": "电压",
    "u": "电压",
    "ua": "A相电压",
    "ub": "B相电压",
    "uc": "C相电压",
    "p": "功率",
    "qf": "功率因数",
    "t": "温度",
    "sd": "湿度",
    "f": "频率",
    "soc": "电池容量",
    "gffddl": "光伏发电量",
    "loadrate": "负载率",
}


class SynthesizerNode:
    """综合响应节点。"""

    SYSTEM_PROMPT = """你是一个数据分析助手。请基于下面给出的结构化信息，直接、准确地回答用户问题。

输出要求：
1. 先直接回答结论，再补充关键依据。
2. 只基于已提供的统计信息回答，不要编造不存在的事实。
3. 优先使用设备中文名；没有中文名时再使用设备代号。
4. 单位必须与指标匹配。
5. 回答简洁、专业、自然。

用户问题：{user_query}
设备范围：{device_info}
指标：{data_type_name}（{data_type}）
单位：{unit}
时间范围：{time_start} 至 {time_end}
数据总量：{data_count} 条
回答风格：{style_instruction}

统计信息：
{statistics_info}
"""

    COMPARISON_SYSTEM_PROMPT = """你是一个数据分析助手。请基于下面给出的对比统计信息，生成一段简洁、专业、直接的对比结论。

输出要求：
1. 先给出核心对比结论，再列出关键依据。
2. 明确指出谁更高、谁更稳定、差异是否明显。
3. 只基于提供的数据回答，不要虚构。
4. 对比对象较多时，优先突出最关键排名与差异。

用户问题：{user_query}
对比对象：{comparison_targets}
指标：{data_type_name}（{data_type}）
单位：{unit}
时间范围：{time_start} 至 {time_end}
回答风格：{style_instruction}

对比统计：
{comparison_statistics_info}
"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    @staticmethod
    def _format_number(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value.is_integer():
                return str(int(value))
            return f"{value:.2f}"
        return str(value)

    def _get_error_message(self, error: str, error_node: Optional[str], state: GraphState) -> str:
        if error_node and error_node in ERROR_MESSAGES:
            template = ERROR_MESSAGES[error_node]
            if error_node == "metadata_mapper":
                return template.format(target=get_target_label_from_state(state) or "目标设备")
            return template

        error_lower = str(error or "").lower()
        if "circuit" in error_lower or "日期" in str(error or ""):
            return ERROR_MESSAGES["sharding_router"]
        if "connection" in error_lower or "timeout" in error_lower:
            return ERROR_MESSAGES["parallel_fetcher"]
        return ERROR_MESSAGES["default"]

    def _format_device_info(self, device_names: Optional[Dict[str, str]], state: Optional[GraphState] = None) -> str:
        if device_names:
            values: List[str] = []
            seen = set()
            for name in device_names.values():
                text = str(name or "").strip()
                if not text:
                    continue
                lowered = text.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                values.append(text)
            if values:
                if len(values) == 1:
                    return values[0]
                if len(values) <= 5:
                    return "、".join(values)
                return f"{values[0]}、{values[1]} 等 {len(values)} 个设备"

        if state is not None:
            return get_target_label_from_state(state) or "未识别设备"
        return "未识别设备"

    def _format_statistics(self, statistics: Optional[Dict[str, Any]], unit: str) -> str:
        if not statistics:
            return "暂无统计信息。"

        parts: List[str] = []
        if "count" in statistics:
            parts.append(f"- 数据量：{self._format_number(statistics['count'])} 条")
        if "avg" in statistics:
            parts.append(f"- 平均值：{self._format_number(statistics['avg'])} {unit}".rstrip())
        if "max" in statistics:
            parts.append(f"- 最大值：{self._format_number(statistics['max'])} {unit}".rstrip())
        if "min" in statistics:
            parts.append(f"- 最小值：{self._format_number(statistics['min'])} {unit}".rstrip())
        if "sum" in statistics:
            parts.append(f"- 总量：{self._format_number(statistics['sum'])} {unit}".rstrip())
        if "median" in statistics:
            parts.append(f"- 中位数：{self._format_number(statistics['median'])} {unit}".rstrip())
        if "std" in statistics:
            parts.append(f"- 标准差：{self._format_number(statistics['std'])} {unit}".rstrip())
        if "cv" in statistics:
            parts.append(f"- 波动系数：{self._format_number(statistics['cv'])}%")
        if "change_rate" in statistics:
            parts.append(f"- 阶段变化：{self._format_number(statistics['change_rate'])}%")
        if "trend" in statistics:
            parts.append(f"- 趋势判断：{statistics['trend']}")
        if "anomaly_count" in statistics:
            parts.append(
                f"- 异常点：{self._format_number(statistics['anomaly_count'])} 个"
                f"（占比 {self._format_number(statistics.get('anomaly_ratio', 0))}%）"
            )

        time_dist = statistics.get("time_distribution")
        if isinstance(time_dist, dict):
            peak_hour = time_dist.get("peak_hour")
            peak_value = time_dist.get("peak_value")
            low_hour = time_dist.get("low_hour")
            low_value = time_dist.get("low_value")
            if peak_hour is not None and peak_value is not None:
                parts.append(
                    f"- 高峰时段：{peak_hour} 时左右，约 {self._format_number(peak_value)} {unit}".rstrip()
                )
            if low_hour is not None and low_value is not None:
                parts.append(
                    f"- 低谷时段：{low_hour} 时左右，约 {self._format_number(low_value)} {unit}".rstrip()
                )

        return "\n".join(parts) if parts else "暂无统计信息。"

    def _get_style_instruction(self, state: GraphState) -> str:
        response_style = get_response_style_from_state(state)
        query_mode = get_query_mode_from_state(state)
        if response_style == "direct_answer":
            return "直接回答结论，尽量用 1 到 3 句话说清楚。"
        if response_style == "compare":
            return "突出对比排名、差异和稳定性判断。"
        if response_style == "list":
            return "适合使用条目化列举，突出关键信息。"
        if query_mode in {"ranked_buckets", "ranked_timepoints"}:
            return "优先回答排名结果，并说明对应时间点或时间桶。"
        if query_mode == "trend_decision":
            return "优先判断上升、下降或基本稳定，并说明依据。"
        if query_mode == "anomaly_points":
            return "优先指出是否存在异常时间点，并给出异常数量、代表性时刻和判断依据。"
        return "先结论后依据，保持简洁。"

    def _determine_table_type(self, data_type: str, data_count: int) -> tuple[bool, Optional[str]]:
        if data_count > 0:
            return True, "sensor_data"
        return False, None

    def _build_focused_result_from_state(self, state: GraphState) -> Optional[Dict[str, Any]]:
        if is_comparison_query(state):
            return None
        raw_data = state.get("raw_data") or []
        if not raw_data:
            return None
        data_type = get_data_type_from_state(state, default="ep")
        analysis = {
            "mode": "single",
            "metric": DATA_TYPE_NAMES.get(data_type, data_type),
            "unit": DATA_TYPE_UNITS.get(data_type, ""),
        }
        return _build_focused_result(
            list(raw_data),
            analysis,
            str(state.get("user_query") or ""),
            data_type,
            query_plan=state.get("query_plan"),
        )

    def _generate_response_with_llm(self, state: GraphState) -> str:
        comparison_statistics = state.get("comparison_statistics") or {}
        if is_comparison_query(state) and comparison_statistics:
            return self._generate_comparison_response(state)

        device_names = state.get("device_names") or {}
        statistics = state.get("statistics") or {}
        total_count = int(state.get("total_count") or 0)
        time_start, time_end = get_time_range_from_state(state)

        data_type = get_data_type_from_state(state, default="ep")
        unit = DATA_TYPE_UNITS.get(data_type, "")
        data_type_name = DATA_TYPE_NAMES.get(data_type, data_type)
        device_info = self._format_device_info(device_names, state)
        statistics_info = self._format_statistics(statistics, unit)

        prompt = self.SYSTEM_PROMPT.format(
            user_query=state.get("user_query", ""),
            device_info=device_info,
            data_type=data_type,
            data_type_name=data_type_name,
            unit=unit,
            time_start=time_start or "未指定",
            time_end=time_end or "未指定",
            statistics_info=statistics_info,
            data_count=total_count,
            style_instruction=self._get_style_instruction(state),
        )

        try:
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="请直接给出最终回答。"),
            ]
            response = self.llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            return str(text).strip()
        except Exception as exc:
            logger.exception("LLM 生成回答失败: %s", exc)
            return self._generate_fallback_response(state)

    def _generate_comparison_response(self, state: GraphState) -> str:
        comparison_targets = get_comparison_targets_from_state(state)
        comparison_statistics = state.get("comparison_statistics") or {}
        if not comparison_targets:
            comparison_targets = list(comparison_statistics.keys())

        time_start, time_end = get_time_range_from_state(state)
        data_type = get_data_type_from_state(state, default="ep")
        unit = DATA_TYPE_UNITS.get(data_type, "")
        data_type_name = DATA_TYPE_NAMES.get(data_type, data_type)

        comparison_info_parts: List[str] = []
        ordered_targets = comparison_targets or list(comparison_statistics.keys())
        for target in ordered_targets:
            stats = comparison_statistics.get(target, {})
            if not stats:
                comparison_info_parts.append(f"- {target}：暂无统计结果")
                continue
            if "error" in stats:
                comparison_info_parts.append(f"- {target}：查询失败，原因：{stats['error']}")
                continue

            lines = [f"- {target}："]
            if "count" in stats:
                lines.append(f"  - 数据量：{self._format_number(stats['count'])} 条")
            if "avg" in stats:
                lines.append(f"  - 均值：{self._format_number(stats['avg'])} {unit}".rstrip())
            if "sum" in stats:
                lines.append(f"  - 总量：{self._format_number(stats['sum'])} {unit}".rstrip())
            if "max" in stats:
                lines.append(f"  - 最大值：{self._format_number(stats['max'])} {unit}".rstrip())
            if "min" in stats:
                lines.append(f"  - 最小值：{self._format_number(stats['min'])} {unit}".rstrip())
            if "std" in stats:
                lines.append(f"  - 标准差：{self._format_number(stats['std'])} {unit}".rstrip())
            if "cv" in stats:
                lines.append(f"  - 波动系数：{self._format_number(stats['cv'])}%")
            if "trend" in stats:
                lines.append(f"  - 趋势：{stats['trend']}")
            comparison_info_parts.append("\n".join(lines))

        comparison_statistics_info = "\n\n".join(comparison_info_parts) or "暂无对比统计信息。"

        extra_sections = [
            self._format_filter_info(state.get("filter_info") or {}),
            self._format_data_sources(state.get("comparison_device_groups") or {}, state.get("device_names") or {}),
            comparison_statistics_info,
        ]
        comparison_context = "\n\n".join(section for section in extra_sections if section)

        prompt = self.COMPARISON_SYSTEM_PROMPT.format(
            user_query=state.get("user_query", ""),
            comparison_targets="、".join(comparison_targets) if comparison_targets else "未指定",
            data_type=data_type,
            data_type_name=data_type_name,
            unit=unit,
            time_start=time_start or "未指定",
            time_end=time_end or "未指定",
            style_instruction=self._get_style_instruction(state),
            comparison_statistics_info=comparison_context,
        )

        try:
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="请给出对比结论。"),
            ]
            response = self.llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            return str(text).strip()
        except Exception as exc:
            logger.exception("对比回答 LLM 生成失败: %s", exc)
            return self._generate_comparison_fallback(state)

    def _generate_fallback_response(self, state: GraphState) -> str:
        device_names = state.get("device_names") or {}
        statistics = state.get("statistics") or {}
        total_count = int(state.get("total_count") or 0)
        data_type = get_data_type_from_state(state, default="ep")
        unit = DATA_TYPE_UNITS.get(data_type, "")
        data_type_name = DATA_TYPE_NAMES.get(data_type, data_type)
        device_info = self._format_device_info(device_names, state)

        parts = [f"已获取 {device_info} 的{data_type_name}数据"]
        if "avg" in statistics:
            parts.append(f"平均值为 {self._format_number(statistics['avg'])} {unit}".rstrip())
        if "max" in statistics:
            parts.append(f"最大值为 {self._format_number(statistics['max'])} {unit}".rstrip())
        if "min" in statistics:
            parts.append(f"最小值为 {self._format_number(statistics['min'])} {unit}".rstrip())
        parts.append(f"共 {total_count} 条记录")
        return "，".join(parts) + "。"

    def _generate_comparison_fallback(self, state: GraphState) -> str:
        comparison_targets = get_comparison_targets_from_state(state)
        comparison_statistics = state.get("comparison_statistics") or {}
        if not comparison_targets:
            comparison_targets = list(comparison_statistics.keys())

        data_type = get_data_type_from_state(state, default="ep")
        unit = DATA_TYPE_UNITS.get(data_type, "")
        data_type_name = DATA_TYPE_NAMES.get(data_type, data_type)

        lines = [f"{data_type_name}对比结果："]
        if comparison_targets:
            lines.append(f"对比对象：{'、'.join(comparison_targets)}。")

        for target in comparison_targets:
            stats = comparison_statistics.get(target, {})
            if not stats:
                lines.append(f"- {target}：暂无统计结果")
                continue
            if "error" in stats:
                lines.append(f"- {target}：查询失败，原因：{stats['error']}")
                continue

            avg = stats.get("avg")
            total = stats.get("sum")
            count = stats.get("count")
            fragments = [f"- {target}："]
            if avg is not None:
                fragments.append(f"均值 {self._format_number(avg)} {unit}".rstrip())
            if total is not None:
                fragments.append(f"总量 {self._format_number(total)} {unit}".rstrip())
            if count is not None:
                fragments.append(f"共 {self._format_number(count)} 条")
            lines.append("，".join(fragments))

        return "\n".join(lines)

    def __call__(self, state: GraphState) -> GraphState:
        history = list(state.get("history", []))
        error = state.get("error")
        error_node = state.get("error_node")

        if error:
            friendly_message = self._get_error_message(str(error), error_node, state)
            return {
                **state,
                "final_response": friendly_message,
                "show_table": False,
                "table_type": None,
                "history": history + [{
                    "node": NODE_SYNTHESIZER,
                    "result": f"错误处理完成: {error_node or 'unknown'}",
                }],
            }

        try:
            total_count = int(state.get("total_count") or 0)
            data_type = get_data_type_from_state(state, default="ep")
            focused_result = self._build_focused_result_from_state(state)
            if focused_result:
                response = build_focused_sensor_response(focused_result, total_count=total_count)
                show_table, table_type = self._determine_table_type(data_type, total_count)
                return {
                    **state,
                    "focused_result": focused_result,
                    "final_response": response,
                    "show_table": show_table,
                    "table_type": table_type,
                    "history": history + [{
                        "node": NODE_SYNTHESIZER,
                        "result": f"生成响应完成: {len(response)} 字, 显示表格={show_table}",
                    }],
                }
            response = self._generate_response_with_llm(state)
            show_table, table_type = self._determine_table_type(data_type, total_count)
            return {
                **state,
                "final_response": response,
                "show_table": show_table,
                "table_type": table_type,
                "history": history + [{
                    "node": NODE_SYNTHESIZER,
                    "result": f"生成响应完成: {len(response)} 字, 显示表格={show_table}",
                }],
            }
        except Exception as exc:
            logger.exception("综合响应生成失败: %s", exc)
            return {
                **state,
                "final_response": "已获取结果，但生成最终回答时发生异常，请稍后重试。",
                "show_table": False,
                "table_type": None,
                "history": history + [{
                    "node": NODE_SYNTHESIZER,
                    "result": f"异常: {str(exc)}",
                }],
            }

    def _format_filter_info(self, filter_info: dict) -> str:
        if not filter_info or not filter_info.get("strategy"):
            return ""

        confidence = float(filter_info.get("confidence", 0) or 0)
        reason = str(filter_info.get("reason") or "").strip()
        original_count = int(filter_info.get("original_count", 0) or 0)
        filtered_count = int(filter_info.get("filtered_count", 0) or 0)

        if confidence >= 0.85:
            return f"筛选说明：{reason}"
        if confidence >= 0.60:
            return (
                f"筛选说明：当前结果基于中等置信度过滤（{confidence:.0%}）。\n"
                f"{reason}\n"
                f"原始候选：{original_count} 个；过滤后：{filtered_count} 个。"
            )
        return (
            f"筛选说明：当前对 {original_count} 个候选设备做了保守过滤。\n"
            f"建议结合原始明细进一步复核。"
        )

    def _format_data_sources(self, comparison_device_groups: dict, device_names: dict) -> str:
        if not comparison_device_groups:
            return ""

        lines = ["数据来源："]
        for target, device_codes in comparison_device_groups.items():
            codes = list(device_codes or [])
            if not codes:
                continue
            lines.append(f"- {target}（{len(codes)} 个设备）")
            for code in codes[:5]:
                name = device_names.get(code, code)
                lines.append(f"  - {code}: {name}")
            if len(codes) > 5:
                lines.append(f"  - 其余 {len(codes) - 5} 个设备已省略")
        return "\n".join(lines)
