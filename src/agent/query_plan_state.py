from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from src.agent.query_plan import QueryPlan, coerce_query_plan


def _normalize_items(values: Any, *, dedupe: bool = True) -> List[str]:
    if values is None:
        return []
    if isinstance(values, (list, tuple, set)):
        raw_items = list(values)
    else:
        raw_items = [values]

    results: List[str] = []
    seen = set()
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        if dedupe:
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
        results.append(text)
    return results


def get_query_plan_from_state(state: Mapping[str, Any]) -> Optional[QueryPlan]:
    return coerce_query_plan(state.get("query_plan"))


def get_explicit_device_codes_from_state(state: Mapping[str, Any]) -> List[str]:
    plan = get_query_plan_from_state(state)
    if plan:
        explicit_codes = _normalize_items(plan.explicit_device_codes)
        if explicit_codes:
            return explicit_codes

    intent = get_intent_from_state(state)
    return _normalize_items(intent.get("explicit_device_codes"))


def get_intent_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    intent = state.get("intent")
    return dict(intent) if isinstance(intent, Mapping) else {}


def get_state_targets(state: Mapping[str, Any]) -> List[str]:
    plan = get_query_plan_from_state(state)
    if plan:
        targets = _normalize_items(plan.search_targets or plan.explicit_device_codes)
        if targets:
            return targets

    intent = get_intent_from_state(state)
    direct_targets = _normalize_items(intent.get("search_targets"))
    if direct_targets:
        return direct_targets

    return _normalize_items(intent.get("target"))


def get_primary_target_from_state(state: Mapping[str, Any]) -> str:
    targets = get_state_targets(state)
    return targets[0] if targets else ""


def get_comparison_targets_from_state(state: Mapping[str, Any]) -> List[str]:
    plan = get_query_plan_from_state(state)
    if plan and (plan.has_comparison_intent or len(plan.search_targets) > 1 or len(plan.explicit_device_codes) > 1):
        targets = _normalize_items(plan.search_targets or plan.explicit_device_codes, dedupe=False)
        if len(targets) > 1:
            return targets

    state_targets = _normalize_items(state.get("comparison_targets"), dedupe=False)
    if len(state_targets) > 1:
        return state_targets

    intent = get_intent_from_state(state)
    intent_targets = _normalize_items(intent.get("comparison_targets"), dedupe=False)
    if len(intent_targets) > 1:
        return intent_targets

    targets = get_state_targets(state)
    return targets if len(targets) > 1 else []


def is_comparison_query(state: Mapping[str, Any]) -> bool:
    if bool(state.get("is_comparison")):
        return True

    plan = get_query_plan_from_state(state)
    if plan and (plan.has_comparison_intent or len(plan.search_targets) > 1 or len(plan.explicit_device_codes) > 1):
        return True

    intent = get_intent_from_state(state)
    if bool(intent.get("is_comparison")):
        return True

    return len(get_comparison_targets_from_state(state)) > 1


def get_requested_tags_from_state(state: Mapping[str, Any]) -> List[str]:
    plan = get_query_plan_from_state(state)
    if plan and isinstance(plan.raw_plan, dict):
        return _normalize_items(plan.raw_plan.get("requested_tags"))

    intent = get_intent_from_state(state)
    return _normalize_items(intent.get("requested_tags"))


def get_data_type_from_state(state: Mapping[str, Any], default: str = "ep") -> str:
    plan = get_query_plan_from_state(state)
    if plan and plan.inferred_data_type:
        return plan.inferred_data_type

    intent = get_intent_from_state(state)
    value = str(intent.get("data_type") or "").strip()
    return value or default


def get_query_mode_from_state(state: Mapping[str, Any], default: str = "general") -> str:
    plan = get_query_plan_from_state(state)
    if plan and plan.query_mode:
        return plan.query_mode

    intent = get_intent_from_state(state)
    value = str(intent.get("query_mode") or "").strip()
    return value or default


def get_response_style_from_state(state: Mapping[str, Any], default: str = "structured_analysis") -> str:
    plan = get_query_plan_from_state(state)
    if plan and plan.response_style:
        return plan.response_style

    intent = get_intent_from_state(state)
    value = str(intent.get("response_style") or "").strip()
    return value or default


def get_aggregation_from_state(state: Mapping[str, Any], default: Optional[str] = None) -> Optional[str]:
    plan = get_query_plan_from_state(state)
    if plan and plan.aggregation:
        return plan.aggregation

    intent = get_intent_from_state(state)
    value = str(intent.get("aggregation") or "").strip()
    return value or default


def get_ranking_limit_from_state(state: Mapping[str, Any]) -> Optional[int]:
    plan = get_query_plan_from_state(state)
    if plan and plan.ranking_limit is not None:
        return plan.ranking_limit

    intent = get_intent_from_state(state)
    value = intent.get("ranking_limit")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_ranking_granularity_from_state(state: Mapping[str, Any]) -> Optional[str]:
    plan = get_query_plan_from_state(state)
    if plan and plan.ranking_granularity:
        return plan.ranking_granularity

    intent = get_intent_from_state(state)
    value = str(intent.get("ranking_granularity") or "").strip()
    return value or None


def get_time_range_from_state(state: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    plan = get_query_plan_from_state(state)
    if plan and plan.time_start and plan.time_end:
        return plan.time_start, plan.time_end

    intent = get_intent_from_state(state)
    time_start = str(intent.get("time_start") or "").strip() or None
    time_end = str(intent.get("time_end") or "").strip() or None
    return time_start, time_end


def get_project_hints_from_state(state: Mapping[str, Any]) -> List[str]:
    plan = get_query_plan_from_state(state)
    if plan:
        project_hints = _normalize_items(plan.project_hints)
        if project_hints:
            return project_hints

    intent = get_intent_from_state(state)
    return _normalize_items(intent.get("project_hints"))


def get_target_label_from_state(state: Mapping[str, Any], joiner: str = " vs ") -> str:
    comparison_targets = get_comparison_targets_from_state(state)
    if len(comparison_targets) > 1:
        return joiner.join(comparison_targets)

    primary = get_primary_target_from_state(state)
    if primary:
        return primary

    intent = get_intent_from_state(state)
    return str(intent.get("target") or "").strip()


def has_sensor_query_intent_from_state(state: Mapping[str, Any]) -> bool:
    plan = get_query_plan_from_state(state)
    if plan:
        return bool(plan.has_sensor_intent)

    intent = get_intent_from_state(state)
    if bool(intent.get("has_sensor_intent")):
        return True

    return get_query_mode_from_state(state) in {"sensor_query", "comparison", "ranked_timepoints", "ranked_buckets", "trend_decision", "anomaly_points"}



def has_detect_data_types_intent_from_state(state: Mapping[str, Any]) -> bool:
    plan = get_query_plan_from_state(state)
    if plan:
        return bool(plan.has_detect_data_types_intent)

    intent = get_intent_from_state(state)
    if bool(intent.get("has_detect_data_types_intent")):
        return True

    return get_query_mode_from_state(state) == "detect_data_types"



def has_project_listing_intent_from_state(state: Mapping[str, Any]) -> bool:
    plan = get_query_plan_from_state(state)
    if plan:
        return bool(plan.has_project_listing_intent)

    intent = get_intent_from_state(state)
    if bool(intent.get("has_project_listing_intent")):
        return True

    return get_query_mode_from_state(state) == "project_listing"



def has_project_stats_intent_from_state(state: Mapping[str, Any]) -> bool:
    plan = get_query_plan_from_state(state)
    if plan:
        return bool(plan.has_project_stats_intent)

    intent = get_intent_from_state(state)
    if bool(intent.get("has_project_stats_intent")):
        return True

    return get_query_mode_from_state(state) == "project_stats"



def has_device_listing_intent_from_state(state: Mapping[str, Any]) -> bool:
    plan = get_query_plan_from_state(state)
    if plan:
        return bool(plan.has_device_listing_intent)

    intent = get_intent_from_state(state)
    if bool(intent.get("has_device_listing_intent")):
        return True

    return get_query_mode_from_state(state) == "device_listing"



def has_pagination_intent_from_state(state: Mapping[str, Any]) -> bool:
    plan = get_query_plan_from_state(state)
    if plan:
        return bool(plan.has_pagination_intent)

    intent = get_intent_from_state(state)
    return bool(intent.get("has_pagination_intent"))


def build_query_plan_context(state: Mapping[str, Any]) -> Dict[str, Any]:
    plan = get_query_plan_from_state(state)
    time_start, time_end = get_time_range_from_state(state)
    return {
        "source": plan.source if plan else None,
        "query_mode": get_query_mode_from_state(state),
        "data_type": get_data_type_from_state(state, default="ep"),
        "requested_tags": get_requested_tags_from_state(state),
        "response_style": get_response_style_from_state(state),
        "aggregation": get_aggregation_from_state(state),
        "ranking_limit": get_ranking_limit_from_state(state),
        "ranking_granularity": get_ranking_granularity_from_state(state),
        "time_start": time_start,
        "time_end": time_end,
        "targets": get_state_targets(state),
        "comparison_targets": get_comparison_targets_from_state(state),
        "is_comparison": is_comparison_query(state),
        "confidence": plan.confidence if plan else None,
    }


def build_compat_intent_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    intent = get_intent_from_state(state)
    if not intent:
        intent = {}

    target = get_target_label_from_state(state)
    if target and not str(intent.get("target") or "").strip():
        intent["target"] = target

    data_type = get_data_type_from_state(state, default="ep")
    if data_type and not str(intent.get("data_type") or "").strip():
        intent["data_type"] = data_type

    comparison_targets = get_comparison_targets_from_state(state)
    if comparison_targets and not intent.get("comparison_targets"):
        intent["comparison_targets"] = comparison_targets

    time_start, time_end = get_time_range_from_state(state)
    if time_start and not str(intent.get("time_start") or "").strip():
        intent["time_start"] = time_start
    if time_end and not str(intent.get("time_end") or "").strip():
        intent["time_end"] = time_end

    if "query_mode" not in intent:
        intent["query_mode"] = get_query_mode_from_state(state)
    if "response_style" not in intent:
        intent["response_style"] = get_response_style_from_state(state)

    aggregation = get_aggregation_from_state(state)
    if aggregation and not str(intent.get("aggregation") or "").strip():
        intent["aggregation"] = aggregation

    ranking_limit = get_ranking_limit_from_state(state)
    if ranking_limit is not None and intent.get("ranking_limit") in (None, ""):
        intent["ranking_limit"] = ranking_limit

    ranking_granularity = get_ranking_granularity_from_state(state)
    if ranking_granularity and not str(intent.get("ranking_granularity") or "").strip():
        intent["ranking_granularity"] = ranking_granularity

    if "is_comparison" not in intent:
        intent["is_comparison"] = is_comparison_query(state)

    return intent
