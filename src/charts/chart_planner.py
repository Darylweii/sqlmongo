from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import httpx

from src.charts.chart_types import CHART_TYPE_ALIASES, CHART_TYPE_LABELS, DEFAULT_CHART_TYPE, SUPPORTED_CHART_TYPES


logger = logging.getLogger(__name__)

_LLM_CLIENT: Any = None

_GENERIC_CHART_KEYWORDS = (
    "图",
    "图表",
    "画图",
    "画一个图",
    "画张图",
    "画一张图",
    "帮我画",
    "可视化",
    "chart",
    "plot",
    "echart",
    "echarts",
    "matplotlib",
)
_SCATTER_HINTS = ("散点", "异常", "离群", "波动点", "分布点")
_BOXPLOT_HINTS = ("箱线", "箱型", "分布", "稳定", "波动")
_HEATMAP_HINTS = ("热力", "热区", "热度", "密度", "按小时分布", "时段密度")
_BAR_HINTS = ("柱状", "柱形", "均值", "总量", "排名", "排行", "分时", "按小时", "小时")
_EXPLICIT_CHART_PATTERNS = (
    (re.compile(r"(热力|热区|热度|小时分布|时段密度).*图?"), "heatmap"),
    (re.compile(r"(箱线|箱型).*图?"), "boxplot"),
    (re.compile(r"(散点).*图?"), "scatter"),
    (re.compile(r"(柱状|柱形|柱).*对比图?"), "bar"),
    (re.compile(r"(均值|总量|汇总|排名).*柱状图?"), "bar"),
    (re.compile(r"(柱状|柱形).*图?"), "bar"),
    (re.compile(r"(趋势|折线|曲线).*对比图?"), "line"),
    (re.compile(r"(趋势|折线|曲线).*图?"), "line"),
    (re.compile(r"对比图"), "line"),
)


def extract_chart_request(user_query: str) -> Dict[str, Any]:
    text = str(user_query or "").strip().lower()
    compact = re.sub(r"\s+", "", text)
    requested_chart_type = _match_explicit_chart_type(compact) or _match_alias_chart_type(compact)
    wants_chart = any(keyword in compact for keyword in _GENERIC_CHART_KEYWORDS) or requested_chart_type is not None
    return {
        "requested_chart_type": requested_chart_type,
        "wants_chart": wants_chart,
        "is_auto_chart": requested_chart_type is None,
        "raw_query": text,
        "compact_query": compact,
    }


def _match_alias_chart_type(compact_query: str) -> Optional[str]:
    for alias, chart_type in CHART_TYPE_ALIASES.items():
        if alias.lower() in compact_query:
            return chart_type
    return None


def _match_explicit_chart_type(compact_query: str) -> Optional[str]:
    for pattern, chart_type in _EXPLICIT_CHART_PATTERNS:
        if pattern.search(compact_query):
            return chart_type
    return None


def plan_chart_specs(
    *,
    normalized_records: List[Dict[str, Any]],
    analysis: Dict[str, Any] | None,
    data_type: str,
    device_names: Dict[str, str] | None,
    user_query: str,
    comparison_slots: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if not normalized_records or not analysis:
        return []

    request = extract_chart_request(user_query)
    explicit_type = request.get("requested_chart_type")
    if explicit_type:
        return [
            _build_chart_plan(
                explicit_type,
                analysis,
                request=request,
                comparison_slots=comparison_slots,
                planner_source="rule",
                reason=f"用户明确指定{CHART_TYPE_LABELS.get(explicit_type, explicit_type)}",
            )
        ]

    chart_type, reason = _pick_rule_based_chart_type(request, analysis, normalized_records, comparison_slots)
    if chart_type:
        return [
            _build_chart_plan(
                chart_type,
                analysis,
                request=request,
                comparison_slots=comparison_slots,
                planner_source="rule",
                reason=reason,
            )
        ]

    llm_choice = None
    if request.get("wants_chart"):
        llm_choice = _invoke_llm_chart_planner(request, analysis)
    if llm_choice and llm_choice.get("chart_type") in SUPPORTED_CHART_TYPES:
        return [
            _build_chart_plan(
                str(llm_choice["chart_type"]),
                analysis,
                request=request,
                comparison_slots=comparison_slots,
                planner_source="llm",
                reason=str(llm_choice.get("reason") or "LLM 兜底选择了更合适的图表类型"),
            )
        ]

    return [
        _build_chart_plan(
            DEFAULT_CHART_TYPE,
            analysis,
            request=request,
            comparison_slots=comparison_slots,
            planner_source="fallback",
            reason="未命中明确规则，回退为默认折线图",
        )
    ]


def _pick_rule_based_chart_type(
    request: Dict[str, Any],
    analysis: Dict[str, Any],
    normalized_records: List[Dict[str, Any]],
    comparison_slots: Optional[List[Dict[str, Any]]] = None,
) -> tuple[Optional[str], str]:
    compact = str(request.get("compact_query") or "")
    wants_chart = bool(request.get("wants_chart"))
    mode = str(analysis.get("mode") or "").strip().lower()
    has_multi_device = len({str(item.get("device") or "") for item in normalized_records}) > 1
    has_multi_slot = len(comparison_slots or []) > 1

    if any(keyword in compact for keyword in _HEATMAP_HINTS):
        return "heatmap", "问题包含热区或时段密度语义，优先使用热力图"
    if any(keyword in compact for keyword in _SCATTER_HINTS):
        return "scatter", "问题关注异常点或离散波动，优先使用散点图"
    if any(keyword in compact for keyword in ("箱线", "箱型")):
        return "boxplot", "问题明确指定箱线图"
    if any(keyword in compact for keyword in _BAR_HINTS):
        if mode == "comparison" or has_multi_device or has_multi_slot:
            return "bar", "问题关注均值、总量或排名，优先使用柱状图"
        return "bar", "问题关注分时分布，优先使用柱状图"

    if mode == "comparison" or has_multi_device or has_multi_slot:
        if any(keyword in compact for keyword in _BOXPLOT_HINTS):
            return "boxplot", "多项对比且关注分布稳定性，优先使用箱线图"
        if wants_chart:
            return None, ""
        return "line", "多项时序对比默认使用折线图"

    if any(keyword in compact for keyword in ("分布", "时段", "小时")):
        return "bar", "单设备查询关注时段分布，优先使用柱状图"
    return "line", "单设备时序查询默认使用折线图"


def _build_chart_plan(
    chart_type: str,
    analysis: Dict[str, Any],
    *,
    request: Dict[str, Any],
    comparison_slots: Optional[List[Dict[str, Any]]],
    planner_source: str,
    reason: str,
) -> Dict[str, Any]:
    metric = str(analysis.get("metric") or "数据")
    mode = str(analysis.get("mode") or "").strip().lower()
    title_map = {
        "line": f"{metric}{'趋势对比' if mode == 'comparison' else '趋势图'}",
        "bar": f"{metric}{'对比柱状图' if mode == 'comparison' else '柱状图'}",
        "scatter": f"{metric}散点图",
        "boxplot": f"{metric}箱线图",
        "heatmap": f"{metric}热力图",
    }
    id_map = {
        "line": "comparison-line" if mode == "comparison" else "trend-line",
        "bar": "comparison-bar" if mode == "comparison" else "hourly-bar",
        "scatter": "scatter-chart",
        "boxplot": "comparison-boxplot" if mode == "comparison" else "distribution-boxplot",
        "heatmap": "heatmap-chart",
    }
    height_map = {"line": 360, "bar": 340, "scatter": 360, "boxplot": 340, "heatmap": 380}
    params = _build_planner_params(chart_type, analysis, request, comparison_slots)
    return {
        "chart_type": chart_type,
        "id": id_map.get(chart_type, f"{chart_type}-chart"),
        "title": title_map.get(chart_type, f"{metric}{CHART_TYPE_LABELS.get(chart_type, '图表')}"),
        "reason": reason,
        "planner_source": planner_source,
        "priority": 1,
        "height": height_map.get(chart_type, 360),
        "params": params,
    }


def _build_planner_params(
    chart_type: str,
    analysis: Dict[str, Any],
    request: Dict[str, Any],
    comparison_slots: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    compact = str(request.get("compact_query") or "")
    has_slots = len(comparison_slots or []) > 1
    highlight_anomalies = bool((analysis.get("anomalies") or [])) or any(keyword in compact for keyword in _SCATTER_HINTS)
    anomaly_only = any(keyword in compact for keyword in ("异常", "离群", "波动点", "异常点"))
    if chart_type == "line":
        return {
            "series_dimension": "slot" if has_slots else ("tag" if str(analysis.get("analysis_scope_mode") or "") == "three_phase_joint" else "device"),
            "comparison_slots": comparison_slots or [],
            "aggregation": "raw",
            "focus": "trend",
            "highlight_anomalies": highlight_anomalies,
            "title_hint": "",
            "subtitle_hint": "",
        }
    if chart_type == "bar":
        return {
            "series_dimension": "slot" if has_slots else "device",
            "comparison_slots": comparison_slots or [],
            "aggregation": "device_summary" if str(analysis.get("mode") or "") == "comparison" or has_slots else "hourly_avg",
            "focus": "summary",
            "highlight_anomalies": False,
            "title_hint": "",
            "subtitle_hint": "",
        }
    if chart_type == "scatter":
        return {
            "series_dimension": "slot" if has_slots else ("tag" if str(analysis.get("analysis_scope_mode") or "") == "three_phase_joint" else "device"),
            "comparison_slots": comparison_slots or [],
            "aggregation": "anomaly_only" if anomaly_only else "raw",
            "focus": "anomaly",
            "highlight_anomalies": True,
            "title_hint": "",
            "subtitle_hint": "",
        }
    if chart_type == "boxplot":
        return {
            "series_dimension": "slot" if has_slots else ("tag" if str(analysis.get("analysis_scope_mode") or "") == "three_phase_joint" else "device"),
            "comparison_slots": comparison_slots or [],
            "aggregation": "device_summary",
            "focus": "distribution",
            "highlight_anomalies": False,
            "title_hint": "",
            "subtitle_hint": "",
        }
    if chart_type == "heatmap":
        return {
            "series_dimension": "slot" if has_slots else ("tag" if str(analysis.get("analysis_scope_mode") or "") == "three_phase_joint" else "device"),
            "comparison_slots": comparison_slots or [],
            "aggregation": "hourly_avg",
            "focus": "heat",
            "highlight_anomalies": False,
            "title_hint": "",
            "subtitle_hint": "",
        }
    return {}


def _get_llm_client() -> Optional[Any]:
    global _LLM_CLIENT
    if _LLM_CLIENT is not None:
        return _LLM_CLIENT
    try:
        from langchain_openai import ChatOpenAI

        http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(12.0))
        _LLM_CLIENT = ChatOpenAI(
            model=os.getenv("VLLM_MODEL") or os.getenv("LLM_MODEL", "/models/Qwen3-32B-AWQ"),
            openai_api_base=os.getenv("VLLM_API_BASE") or os.getenv("LLM_BASE_URL"),
            openai_api_key=os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "not-needed",
            temperature=0.0,
            max_tokens=256,
            http_client=http_client,
            request_timeout=12.0,
        )
        return _LLM_CLIENT
    except Exception as exc:
        logger.warning("chart.planner.llm_init_failed error=%s", exc)
        return None


def _invoke_llm_chart_planner(request: Dict[str, Any], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    llm = _get_llm_client()
    if llm is None:
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(
                content=(
                    "你是图表规划器。只能在 line、bar、scatter、boxplot、heatmap 中选一个最合适的图表类型。"
                    "输出 JSON：{\"chart_type\":\"...\",\"reason\":\"...\",\"confidence\":0.0}"
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "query": request.get("raw_query"),
                        "analysis_mode": analysis.get("mode"),
                        "metric": analysis.get("metric"),
                        "insights": (analysis.get("insights") or [])[:3],
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", response)
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
            chart_type = str(payload.get("chart_type") or "").strip().lower()
            if chart_type in SUPPORTED_CHART_TYPES:
                return {
                    "chart_type": chart_type,
                    "reason": str(payload.get("reason") or "LLM 兜底选择的图表类型"),
                    "confidence": payload.get("confidence"),
                }
        for chart_type in SUPPORTED_CHART_TYPES:
            if re.search(rf"\b{chart_type}\b", text, re.IGNORECASE):
                return {
                    "chart_type": chart_type,
                    "reason": "LLM 文本回复命中了图表类型",
                    "confidence": None,
                }
    except Exception as exc:
        logger.warning("chart.planner.llm_failed error=%s", exc)
    return None
