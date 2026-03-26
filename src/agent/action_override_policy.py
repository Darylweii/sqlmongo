from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from src.agent.query_plan_state import (
    get_data_type_from_state,
    get_explicit_device_codes_from_state,
    has_detect_data_types_intent_from_state,
    has_pagination_intent_from_state,
    has_project_listing_intent_from_state,
    has_project_stats_intent_from_state,
    has_sensor_query_intent_from_state,
    is_comparison_query,
)

OVERRIDE_CANDIDATES = {"search_devices", "detect_data_types", "direct_answer", "final_answer"}


@dataclass(frozen=True)
class ActionOverrideContext:
    query_state: Mapping[str, Any]
    action: str
    action_input: Mapping[str, Any]
    history_actions: Tuple[str, ...] = ()
    has_cached_device_codes: bool = False
    preferred_device_codes: Tuple[str, ...] = ()
    preferred_tg_values: Tuple[str, ...] = ()
    preferred_source: str = ""
    resolved_time_range: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class ActionOverrideDecision:
    action: str
    action_input: Dict[str, Any]
    reason: Optional[str] = None



def _history_contains(history_actions: Sequence[str], action: str) -> bool:
    return any(item == action for item in history_actions)



def decide_metadata_override(context: ActionOverrideContext) -> Optional[ActionOverrideDecision]:
    if has_sensor_query_intent_from_state(context.query_state):
        return None
    if get_explicit_device_codes_from_state(context.query_state) or context.has_cached_device_codes:
        return None
    if context.action not in OVERRIDE_CANDIDATES:
        return None

    if has_project_listing_intent_from_state(context.query_state):
        if context.action == "list_projects" or _history_contains(context.history_actions, "list_projects"):
            return None
        return ActionOverrideDecision(
            action="list_projects",
            action_input={},
            reason=f"override_from_{context.action}_project_list",
        )

    if has_project_stats_intent_from_state(context.query_state):
        if context.action == "get_project_stats" or _history_contains(context.history_actions, "get_project_stats"):
            return None
        return ActionOverrideDecision(
            action="get_project_stats",
            action_input={},
            reason=f"override_from_{context.action}_project_stats",
        )

    return None



def decide_sensor_override(context: ActionOverrideContext) -> Optional[ActionOverrideDecision]:
    if has_detect_data_types_intent_from_state(context.query_state):
        return None
    if not context.preferred_device_codes:
        return None
    if not has_sensor_query_intent_from_state(context.query_state):
        return None
    if context.action == "get_sensor_data" or _history_contains(context.history_actions, "get_sensor_data"):
        return None
    if context.action not in OVERRIDE_CANDIDATES:
        return None

    forced_input = dict(context.action_input or {})
    forced_input["device_codes"] = list(context.preferred_device_codes)
    forced_input["tg_values"] = list(context.preferred_tg_values)
    forced_input["data_type"] = forced_input.get("data_type") or get_data_type_from_state(context.query_state, default="ep")
    if context.resolved_time_range:
        forced_input["start_time"] = context.resolved_time_range["start_time"]
        forced_input["end_time"] = context.resolved_time_range["end_time"]
    forced_input.setdefault("page", 1)
    if len(context.preferred_device_codes) > 1 and is_comparison_query(context.query_state) and not has_pagination_intent_from_state(context.query_state):
        forced_input["page_size"] = 0

    source = context.preferred_source or "resolved"
    return ActionOverrideDecision(
        action="get_sensor_data",
        action_input=forced_input,
        reason=f"override_from_{context.action}_{source}",
    )



def apply_action_override_policy(context: ActionOverrideContext) -> ActionOverrideDecision:
    metadata_decision = decide_metadata_override(context)
    if metadata_decision is not None:
        return metadata_decision

    sensor_decision = decide_sensor_override(context)
    if sensor_decision is not None:
        return sensor_decision

    return ActionOverrideDecision(
        action=context.action,
        action_input=dict(context.action_input or {}),
        reason=None,
    )
