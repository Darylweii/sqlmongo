from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.agent.action_override_policy import ActionOverrideContext, apply_action_override_policy
from src.agent.query_plan_state import (
    build_query_plan_context,
    get_comparison_targets_from_state,
    get_explicit_device_codes_from_state,
    get_primary_target_from_state,
    get_query_plan_from_state,
    get_time_range_from_state,
    is_comparison_query,
)
from src.agent.types import GraphState, NODE_ACTION_OVERRIDE_POLICY
from src.metadata.metadata_engine import MetadataEngine

logger = logging.getLogger(__name__)

_DEFAULT_DAG_ACTION = "search_devices"


class ActionOverridePolicyNode:
    """Apply the shared action override policy inside the DAG pipeline."""

    def __init__(self, metadata_engine: Optional[MetadataEngine] = None):
        self.metadata_engine = metadata_engine

    def _build_context(self, state: GraphState) -> ActionOverrideContext:
        start_time, end_time = get_time_range_from_state(state)
        resolved_time_range = None
        if start_time and end_time:
            resolved_time_range = {
                "start_time": start_time,
                "end_time": end_time,
            }

        preferred_device_codes = tuple(
            str(code).strip()
            for code in (state.get("device_codes") or [])
            if str(code).strip()
        )
        preferred_source = "resolved" if preferred_device_codes else ""

        query_state: Dict[str, Any] = {}
        if state.get("query_plan") is not None:
            query_state["query_plan"] = state.get("query_plan")
        if state.get("intent") is not None:
            query_state["intent"] = state.get("intent")

        return ActionOverrideContext(
            query_state=query_state,
            action=_DEFAULT_DAG_ACTION,
            action_input={},
            history_actions=tuple(),
            has_cached_device_codes=bool(state.get("device_codes")),
            preferred_device_codes=preferred_device_codes,
            preferred_tg_values=tuple(),
            preferred_source=preferred_source,
            resolved_time_range=resolved_time_range,
        )

    def _append_history(self, state: GraphState, message: str) -> List[Dict[str, Any]]:
        history = list(state.get("history", []))
        history.append({"node": NODE_ACTION_OVERRIDE_POLICY, "result": message})
        return history

    def _lookup_device_names(self, device_codes: List[str]) -> Dict[str, str]:
        device_names: Dict[str, str] = {}
        if self.metadata_engine is None:
            return device_names

        for device_code in device_codes:
            try:
                devices, _ = self.metadata_engine.search_devices(device_code)
            except Exception as exc:
                logger.debug("action_override_policy.lookup_failed device=%s error=%s", device_code, exc)
                continue

            for device in devices:
                matched_code = str(getattr(device, "device", "") or "").strip()
                if matched_code != device_code:
                    continue
                device_name = str(getattr(device, "name", "") or "").strip()
                if device_name and device_code not in device_names:
                    device_names[device_code] = device_name
                    break

        return device_names

    def _build_comparison_groups(self, state: GraphState, device_codes: List[str]) -> Dict[str, List[str]]:
        comparison_targets = list(get_comparison_targets_from_state(state) or [])
        if not comparison_targets:
            return {code: [code] for code in device_codes}

        if len(comparison_targets) == len(device_codes):
            return {
                str(target): [device_code]
                for target, device_code in zip(comparison_targets, device_codes)
            }

        remaining_codes = list(device_codes)
        groups: Dict[str, List[str]] = {}
        for target in comparison_targets:
            lowered_target = str(target).lower()
            matched_code = next(
                (
                    code
                    for code in remaining_codes
                    if code.lower() == lowered_target
                    or code.lower() in lowered_target
                    or lowered_target in code.lower()
                ),
                None,
            )
            if matched_code is None:
                continue
            groups[str(target)] = [matched_code]
            remaining_codes.remove(matched_code)

        for device_code in remaining_codes:
            groups[device_code] = [device_code]
        return groups

    def _build_metadata_query_info(self, state: GraphState, payload_key: str, payload: List[Dict[str, Any]]) -> Dict[str, Any]:
        query_info = {
            payload_key: payload,
            "total_count": len(payload),
            "query_plan_context": build_query_plan_context(state),
        }
        return query_info

    def _search_project_candidates(self, keyword: str) -> List[Dict[str, Any]]:
        if self.metadata_engine is None:
            return []
        text = str(keyword or "").strip()
        if not text:
            return []
        try:
            return list(self.metadata_engine.search_projects(text, limit=10) or [])
        except Exception as exc:
            logger.debug("action_override_policy.project_search_failed keyword=%s error=%s", text, exc)
            return []

    def _build_project_clarification_candidates(self, projects: List[Dict[str, Any]], decision_mode: str = "recommend_confirm") -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for index, project in enumerate(projects, start=1):
            score = float(project.get("match_score") or 0.0)
            candidates.append(
                {
                    "device": project.get("project_name") or project.get("project_code_name") or project.get("id"),
                    "name": project.get("project_code_name") or project.get("code_name") or "项目候选",
                    "project_id": project.get("id"),
                    "project_name": project.get("project_name"),
                    "project_code_name": project.get("project_code_name") or project.get("code_name"),
                    "tg": None,
                    "match_type": "project_candidate",
                    "match_score": score,
                    "match_reason": project.get("match_reason") or "项目名称模糊匹配",
                    "matched_fields": project.get("matched_fields") or ["project_name"],
                    "retrieval_source": "project_scope",
                    "confidence_level": "high" if score >= 88 else "medium" if score >= 60 else "low",
                    "is_recommended": index == 1,
                    "recommendation_rank": index,
                    "decision_mode": decision_mode,
                    "candidate_kind": "project",
                }
            )
        return candidates

    def _get_project_stats_ranking_spec(self, state: GraphState) -> Dict[str, Any]:
        plan = get_query_plan_from_state(state)
        current_question = str((plan.current_question if plan else "") or "").strip()
        ranking_limit = plan.ranking_limit if plan else None
        ranking_order = str(plan.ranking_order or "").strip().lower() if plan and plan.ranking_order else ""
        compact = current_question.replace(" ", "").lower()

        generic_tokens = (
            "哪个项目",
            "哪一个项目",
            "各项目",
            "每个项目",
            "项目设备",
            "项目排名",
            "排行",
            "排名",
            "top",
            "前十",
            "前5",
            "前3",
        )
        is_generic = (not str(get_primary_target_from_state(state) or "").strip()) or any(token in compact for token in generic_tokens)

        if not ranking_order:
            if any(token in compact for token in ("最少", "最小", "最低")):
                ranking_order = "asc"
            elif any(token in compact for token in ("最多", "最大", "最高", "排行", "排名", "top", "前")):
                ranking_order = "desc"

        if ranking_limit is None:
            if "前十" in compact or "top10" in compact:
                ranking_limit = 10
            elif "前5" in compact or "top5" in compact:
                ranking_limit = 5
            elif "前3" in compact or "top3" in compact:
                ranking_limit = 3
            elif ranking_order:
                ranking_limit = 1

        return {
            "current_question": current_question,
            "ranking_order": ranking_order or None,
            "ranking_limit": ranking_limit,
            "is_generic": is_generic,
        }

    def _handle_sensor_override(
        self,
        state: GraphState,
        *,
        device_codes: List[str],
        reason: Optional[str],
    ) -> GraphState:
        is_comparison = is_comparison_query(state) and len(device_codes) > 1
        device_names = dict(state.get("device_names") or {})
        device_names.update(self._lookup_device_names(device_codes))

        intent = dict(state.get("intent") or {})
        if device_codes and not str(intent.get("target") or "").strip():
            intent["target"] = " vs ".join(device_codes) if is_comparison else device_codes[0]
        intent["is_comparison"] = is_comparison
        if is_comparison and not intent.get("comparison_targets"):
            intent["comparison_targets"] = list(get_comparison_targets_from_state(state) or device_codes)

        message = f"override search_devices -> get_sensor_data ({reason or 'shared_policy'})"
        return {
            **state,
            "intent": intent,
            "device_codes": list(device_codes),
            "device_names": device_names or None,
            "is_comparison": is_comparison,
            "comparison_targets": list(get_comparison_targets_from_state(state) or device_codes) if is_comparison else state.get("comparison_targets"),
            "comparison_device_groups": self._build_comparison_groups(state, device_codes) if is_comparison else state.get("comparison_device_groups"),
            "override_action": "get_sensor_data",
            "override_reason": reason,
            "override_terminal": False,
            "history": self._append_history(state, message),
        }

    def _handle_projects_override(self, state: GraphState, *, reason: Optional[str]) -> GraphState:
        if self.metadata_engine is None:
            return {
                **state,
                "error": "metadata engine is not configured",
                "error_node": NODE_ACTION_OVERRIDE_POLICY,
                "history": self._append_history(state, "override failed: metadata engine unavailable"),
            }

        projects = self.metadata_engine.list_projects()
        response = f"已找到 {len(projects)} 个项目，请查看下表。" if projects else "当前没有可用项目。"
        return {
            **state,
            "final_response": response,
            "show_table": bool(projects),
            "table_type": "projects" if projects else None,
            "query_info": self._build_metadata_query_info(state, "projects", projects),
            "total_count": len(projects),
            "override_action": "list_projects",
            "override_reason": reason,
            "override_terminal": True,
            "history": self._append_history(state, f"override search_devices -> list_projects ({reason or 'shared_policy'})"),
        }

    def _handle_project_stats_override(self, state: GraphState, *, reason: Optional[str]) -> GraphState:
        if self.metadata_engine is None:
            return {
                **state,
                "error": "metadata engine is not configured",
                "error_node": NODE_ACTION_OVERRIDE_POLICY,
                "history": self._append_history(state, "override failed: metadata engine unavailable"),
            }

        stats = self.metadata_engine.get_project_device_stats()
        ranking_spec = self._get_project_stats_ranking_spec(state)
        if stats and ranking_spec.get("is_generic") and ranking_spec.get("ranking_order"):
            reverse = str(ranking_spec.get("ranking_order")) == "desc"
            limited_stats = sorted(
                list(stats),
                key=lambda item: (int(item.get("device_count") or 0), str(item.get("project_name") or "")),
                reverse=reverse,
            )[: max(1, int(ranking_spec.get("ranking_limit") or 1))]
            top_item = limited_stats[0]
            extreme_label = "最多" if reverse else "最少"
            response = (
                f"设备数量{extreme_label}的项目是 {top_item.get('project_name') or '-'}，共有 {int(top_item.get('device_count') or 0)} 个设备。"
                if int(ranking_spec.get("ranking_limit") or 1) == 1
                else f"已整理设备数量{extreme_label}的前 {len(limited_stats)} 个项目，请查看下表。"
            )
            return {
                **state,
                "final_response": response,
                "show_table": True,
                "table_type": "project_stats",
                "query_info": self._build_metadata_query_info(state, "stats", limited_stats),
                "total_count": len(limited_stats),
                "override_action": "get_project_stats",
                "override_reason": reason,
                "override_terminal": True,
                "history": self._append_history(state, f"override search_devices -> get_project_stats_ranked ({reason or 'shared_policy'})"),
            }

        target = str(get_primary_target_from_state(state) or "").strip()
        if target:
            project_candidates = self._search_project_candidates(target)
            if project_candidates:
                top_score = float(project_candidates[0].get("match_score") or 0.0)
                second_score = float(project_candidates[1].get("match_score") or 0.0) if len(project_candidates) > 1 else -1.0
                if len(project_candidates) > 1 and (top_score < 88 or (top_score - second_score) < 10):
                    clarification = {
                        "keyword": target,
                        "candidate_kind": "project",
                        "decision_mode": "recommend_confirm",
                        "candidates": self._build_project_clarification_candidates(project_candidates[:10]),
                    }
                    return {
                        **state,
                        "clarification_required": True,
                        "clarification_candidates": [clarification],
                        "final_response": f"“{target}”匹配到多个相近项目，请先确认你想统计哪一个项目的设备数量。",
                        "show_table": False,
                        "table_type": None,
                        "override_action": "get_project_stats",
                        "override_reason": reason,
                        "override_terminal": True,
                        "history": self._append_history(state, f"override search_devices -> clarify_project_stats ({reason or 'shared_policy'})"),
                    }

                if top_score >= 88:
                    matched_project = project_candidates[0]
                    project_id = str(matched_project.get("id") or "").strip()
                    filtered_stats = [item for item in stats if str(item.get("id") or "").strip() == project_id]
                    if not filtered_stats:
                        filtered_stats = [
                            {
                                "id": matched_project.get("id"),
                                "project_name": matched_project.get("project_name"),
                                "code_name": matched_project.get("project_code_name") or matched_project.get("code_name"),
                                "device_count": 0,
                            }
                        ]
                    item = filtered_stats[0]
                    response = f"{item.get('project_name') or target}共有 {int(item.get('device_count') or 0)} 个设备。"
                    return {
                        **state,
                        "final_response": response,
                        "show_table": True,
                        "table_type": "project_stats",
                        "query_info": self._build_metadata_query_info(state, "stats", filtered_stats),
                        "total_count": len(filtered_stats),
                        "override_action": "get_project_stats",
                        "override_reason": reason,
                        "override_terminal": True,
                        "history": self._append_history(state, f"override search_devices -> get_project_stats_single ({reason or 'shared_policy'})"),
                    }

        response = "已整理各项目设备数量统计，请查看下表。" if stats else "当前没有可用的项目设备统计。"
        return {
            **state,
            "final_response": response,
            "show_table": bool(stats),
            "table_type": "project_stats" if stats else None,
            "query_info": self._build_metadata_query_info(state, "stats", stats),
            "total_count": len(stats),
            "override_action": "get_project_stats",
            "override_reason": reason,
            "override_terminal": True,
            "history": self._append_history(state, f"override search_devices -> get_project_stats ({reason or 'shared_policy'})"),
        }

    def __call__(self, state: GraphState) -> GraphState:
        decision = apply_action_override_policy(self._build_context(state))

        if decision.action == "get_sensor_data":
            device_codes = [
                str(code).strip()
                for code in (decision.action_input.get("device_codes") or [])
                if str(code).strip()
            ]
            if not device_codes:
                return {
                    **state,
                    "error": "action override resolved to get_sensor_data without device codes",
                    "error_node": NODE_ACTION_OVERRIDE_POLICY,
                    "history": self._append_history(state, "override failed: missing device codes"),
                }
            return self._handle_sensor_override(state, device_codes=device_codes, reason=decision.reason)

        if decision.action == "list_projects":
            return self._handle_projects_override(state, reason=decision.reason)

        if decision.action == "get_project_stats":
            return self._handle_project_stats_override(state, reason=decision.reason)

        message = "override kept default action search_devices"
        return {
            **state,
            "override_action": decision.action,
            "override_reason": decision.reason,
            "override_terminal": False,
            "history": self._append_history(state, message),
        }
