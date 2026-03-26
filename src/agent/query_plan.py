from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from src.agent.query_entities import extract_requested_metric_tags, parse_query_entities
from src.agent.query_time_range import resolve_time_range_from_query

VALID_QUERY_MODES = {
    "general",
    "sensor_query",
    "comparison",
    "detect_data_types",
    "project_listing",
    "project_stats",
    "device_listing",
    "trend_decision",
    "ranked_timepoints",
    "ranked_buckets",
    "anomaly_points",
}
VALID_RESPONSE_STYLES = {"direct_answer", "structured_analysis", "clarify", "list", "compare"}
VALID_RANKING_ORDERS = {"asc", "desc"}
VALID_RANKING_GRANULARITIES = {"hour", "day", "week", "month"}
VALID_AGGREGATIONS = {
    "raw",
    "bucket",
    "delta",
    "avg",
    "sum",
    "max",
    "min",
    "trend_window",
    "period_compare",
    "compare",
}


@dataclass(frozen=True)
class QueryPlan:
    current_question: str
    source: str = "fallback"
    query_mode: str = "general"
    inferred_data_type: Optional[str] = None
    explicit_device_codes: Tuple[str, ...] = ()
    search_targets: Tuple[str, ...] = ()
    project_hints: Tuple[str, ...] = ()
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    has_sensor_intent: bool = False
    has_detect_data_types_intent: bool = False
    has_project_listing_intent: bool = False
    has_project_stats_intent: bool = False
    has_device_listing_intent: bool = False
    has_comparison_intent: bool = False
    has_pagination_intent: bool = False
    has_time_reference: bool = False
    has_ranked_point_intent: bool = False
    ranking_order: Optional[str] = None
    ranking_limit: Optional[int] = None
    ranking_granularity: Optional[str] = None
    has_trend_decision_intent: bool = False
    has_anomaly_point_intent: bool = False
    aggregation: Optional[str] = None
    response_style: str = "structured_analysis"
    period_compare_targets: Tuple[str, ...] = ()
    confidence: float = 0.0
    raw_plan: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["explicit_device_codes"] = list(self.explicit_device_codes)
        payload["search_targets"] = list(self.search_targets)
        payload["project_hints"] = list(self.project_hints)
        payload["period_compare_targets"] = list(self.period_compare_targets)
        payload["raw_plan"] = dict(self.raw_plan or {})
        return payload


_PERIOD_COMPARE_TOKENS = (
    "\u540c\u6bd4",
    "\u73af\u6bd4",
    "\u6628\u5929\u548c\u524d\u5929",
    "\u6628\u5929\u8ddf\u524d\u5929",
    "\u4eca\u5929\u548c\u6628\u5929",
    "\u4eca\u5929\u8ddf\u6628\u5929",
    "\u672c\u5468\u548c\u4e0a\u5468",
    "\u672c\u5468\u8ddf\u4e0a\u5468",
    "\u4e0a\u5468\u548c\u672c\u5468",
)


def _as_tuple(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        items = value
    elif isinstance(value, list):
        items = tuple(value)
    else:
        items = (value,)
    seen = set()
    result = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(text)
    return tuple(result)


def _normalize_optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _normalize_query_mode(value: Any, fallback: str = "general") -> str:
    text = str(value or "").strip()
    return text if text in VALID_QUERY_MODES else fallback


def _normalize_response_style(value: Any, fallback: str = "structured_analysis") -> str:
    text = str(value or "").strip()
    return text if text in VALID_RESPONSE_STYLES else fallback


def _normalize_ranking_order(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if text in VALID_RANKING_ORDERS else None


def _normalize_ranking_granularity(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if text in VALID_RANKING_GRANULARITIES else None


def _normalize_aggregation(value: Any) -> Optional[str]:
    text = str(value or "").strip().lower()
    return text if text in VALID_AGGREGATIONS else None


def _normalize_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _normalize_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return fallback
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return fallback


def _infer_response_style(query_mode: str) -> str:
    if query_mode in {"project_listing", "device_listing", "detect_data_types"}:
        return "list"
    if query_mode == "comparison":
        return "compare"
    if query_mode in {"ranked_buckets", "ranked_timepoints", "trend_decision", "anomaly_points"}:
        return "direct_answer"
    if query_mode == "general":
        return "direct_answer"
    return "structured_analysis"


def _infer_aggregation(query_mode: str, ranking_granularity: Optional[str]) -> Optional[str]:
    if query_mode == "ranked_timepoints":
        return "raw"
    if query_mode == "ranked_buckets":
        return "bucket"
    if query_mode == "trend_decision":
        return "trend_window"
    if query_mode == "anomaly_points":
        return "raw"
    if query_mode == "comparison":
        return "compare"
    if ranking_granularity:
        return "bucket"
    return None


def _extract_period_compare_targets(current_question: str) -> Tuple[str, ...]:
    matched = []
    for token in _PERIOD_COMPARE_TOKENS:
        if token in current_question:
            matched.append(token)
    return _as_tuple(matched)


def fallback_query_plan(user_query: str) -> QueryPlan:
    parsed = parse_query_entities(user_query)
    aggregation = _infer_aggregation(parsed.query_mode, parsed.ranking_granularity)
    response_style = _infer_response_style(parsed.query_mode)
    resolved_time_range = resolve_time_range_from_query(parsed.current_question)
    requested_tags = list(extract_requested_metric_tags(parsed.current_question))
    raw_plan = {"requested_tags": requested_tags} if requested_tags else {}
    return QueryPlan(
        current_question=parsed.current_question,
        source="fallback",
        query_mode=parsed.query_mode,
        inferred_data_type=parsed.inferred_data_type,
        explicit_device_codes=parsed.explicit_device_codes,
        search_targets=parsed.search_targets,
        project_hints=parsed.project_hints,
        time_start=(resolved_time_range or {}).get("start_time"),
        time_end=(resolved_time_range or {}).get("end_time"),
        has_sensor_intent=parsed.has_sensor_intent,
        has_detect_data_types_intent=parsed.has_detect_data_types_intent,
        has_project_listing_intent=parsed.has_project_listing_intent,
        has_project_stats_intent=parsed.has_project_stats_intent,
        has_device_listing_intent=parsed.has_device_listing_intent,
        has_comparison_intent=parsed.has_comparison_intent,
        has_pagination_intent=parsed.has_pagination_intent,
        has_time_reference=parsed.has_time_reference,
        has_ranked_point_intent=parsed.has_ranked_point_intent,
        ranking_order=parsed.ranking_order,
        ranking_limit=parsed.ranking_limit,
        ranking_granularity=parsed.ranking_granularity,
        has_trend_decision_intent=parsed.has_trend_decision_intent,
        has_anomaly_point_intent=parsed.has_anomaly_point_intent,
        aggregation=aggregation,
        response_style=response_style,
        period_compare_targets=_extract_period_compare_targets(parsed.current_question),
        confidence=0.35,
        raw_plan=raw_plan,
    )


def coerce_query_plan(data: Any) -> Optional[QueryPlan]:
    if data is None:
        return None
    if isinstance(data, QueryPlan):
        return data
    if not isinstance(data, Mapping):
        return None

    current_question = str(data.get("current_question") or "").strip()
    query_mode = _normalize_query_mode(data.get("query_mode"), fallback="general")
    ranking_granularity = _normalize_ranking_granularity(data.get("ranking_granularity"))
    ranking_order = _normalize_ranking_order(data.get("ranking_order"))
    response_style = _normalize_response_style(data.get("response_style"), fallback=_infer_response_style(query_mode))
    aggregation = _normalize_aggregation(data.get("aggregation")) or _infer_aggregation(query_mode, ranking_granularity)

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0))

    return QueryPlan(
        current_question=current_question,
        source=str(data.get("source") or "fallback").strip() or "fallback",
        query_mode=query_mode,
        inferred_data_type=str(data.get("inferred_data_type") or "").strip() or None,
        explicit_device_codes=_as_tuple(data.get("explicit_device_codes")),
        search_targets=_as_tuple(data.get("search_targets")),
        project_hints=_as_tuple(data.get("project_hints")),
        time_start=_normalize_optional_text(data.get("time_start")),
        time_end=_normalize_optional_text(data.get("time_end")),
        has_sensor_intent=_normalize_bool(data.get("has_sensor_intent"), fallback=query_mode not in {"general", "project_listing", "project_stats", "device_listing"}),
        has_detect_data_types_intent=_normalize_bool(data.get("has_detect_data_types_intent"), fallback=query_mode == "detect_data_types"),
        has_project_listing_intent=_normalize_bool(data.get("has_project_listing_intent"), fallback=query_mode == "project_listing"),
        has_project_stats_intent=_normalize_bool(data.get("has_project_stats_intent"), fallback=query_mode == "project_stats"),
        has_device_listing_intent=_normalize_bool(data.get("has_device_listing_intent"), fallback=query_mode == "device_listing"),
        has_comparison_intent=_normalize_bool(data.get("has_comparison_intent"), fallback=query_mode == "comparison"),
        has_pagination_intent=_normalize_bool(data.get("has_pagination_intent"), fallback=False),
        has_time_reference=_normalize_bool(data.get("has_time_reference"), fallback=False),
        has_ranked_point_intent=_normalize_bool(data.get("has_ranked_point_intent"), fallback=query_mode in {"ranked_timepoints", "ranked_buckets"}),
        ranking_order=ranking_order,
        ranking_limit=_normalize_int(data.get("ranking_limit")),
        ranking_granularity=ranking_granularity,
        has_trend_decision_intent=_normalize_bool(data.get("has_trend_decision_intent"), fallback=query_mode == "trend_decision"),
        has_anomaly_point_intent=_normalize_bool(data.get("has_anomaly_point_intent"), fallback=query_mode == "anomaly_points"),
        aggregation=aggregation,
        response_style=response_style,
        period_compare_targets=_as_tuple(data.get("period_compare_targets")),
        confidence=confidence,
        raw_plan=dict(data.get("raw_plan") or {}),
    )
