"""Metadata mapper node with hybrid device resolution."""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.language_models import BaseChatModel

from src.agent.query_entities import allows_explicit_multi_scope_aggregation
from src.agent.query_plan_state import (
    build_compat_intent_from_state,
    get_comparison_targets_from_state,
    get_primary_target_from_state,
    get_project_hints_from_state,
    has_device_listing_intent_from_state,
    is_comparison_query,
)
from src.agent.types import GraphState, NODE_METADATA_MAPPER
from src.agent.utils.hybrid_device_resolver import HybridDeviceResolver
from src.exceptions import DatabaseConnectionError, MetadataEngineError
from src.metadata.metadata_engine import MetadataEngine


logger = logging.getLogger(__name__)


class MetadataMapperNode:
    def __init__(
        self,
        metadata_engine: MetadataEngine,
        coder_llm: Optional[BaseChatModel] = None,
        use_llm_sql: bool = False,
        device_search=None,
        enable_semantic_fallback: bool = True,
    ):
        self.metadata_engine = metadata_engine
        self.coder_llm = coder_llm
        self.use_llm_sql = use_llm_sql and coder_llm is not None
        self.resolver = HybridDeviceResolver(
            metadata_engine=metadata_engine,
            device_search=device_search,
            enable_semantic_fallback=enable_semantic_fallback,
        )

    def _normalize_scope_text(self, value: object) -> str:
        return "".join(str(value or "").strip().lower().split())

    def _get_alias_memory_row(self, state: GraphState, target: str) -> dict | None:
        alias_memory = state.get("alias_memory") if isinstance(state.get("alias_memory"), dict) else {}
        normalized_target = self._normalize_scope_text(target)
        if not normalized_target:
            return None
        entry = alias_memory.get(normalized_target)
        if not isinstance(entry, dict):
            return None
        device_code = str(entry.get("device") or "").strip()
        if not device_code:
            return None
        return {
            "device": device_code,
            "name": entry.get("name"),
            "device_type": entry.get("device_type", ""),
            "project_id": entry.get("project_id", ""),
            "project_name": entry.get("project_name"),
            "project_code_name": entry.get("project_code_name"),
            "tg": entry.get("tg"),
            "match_type": "session_alias",
            "match_score": 1.0,
            "match_reason": "session_alias",
            "retrieval_source": "session_alias",
            "confidence_level": "high",
            "decision_mode": "auto_resolve",
            "recommendation_rank": 1,
            "is_recommended": False,
        }

    def _device_to_row(self, device) -> dict:
        if hasattr(device, "to_dict"):
            payload = device.to_dict()
            payload.setdefault("device", getattr(device, "device", ""))
            payload.setdefault("name", getattr(device, "name", ""))
            payload.setdefault("device_type", getattr(device, "device_type", ""))
            payload.setdefault("project_id", getattr(device, "project_id", ""))
            payload.setdefault("project_name", getattr(device, "project_name", None))
            payload.setdefault("project_code_name", getattr(device, "project_code_name", None))
            payload.setdefault("tg", getattr(device, "tg", None))
            return payload
        return {
            "device": getattr(device, "device", ""),
            "name": getattr(device, "name", ""),
            "device_type": getattr(device, "device_type", ""),
            "project_id": getattr(device, "project_id", ""),
            "project_name": getattr(device, "project_name", None),
            "project_code_name": getattr(device, "project_code_name", None),
            "tg": getattr(device, "tg", None),
        }

    def _is_explicit_device_code(self, value: str) -> bool:
        return bool(re.fullmatch(r"[a-zA-Z]\d*_[a-zA-Z0-9_]+", str(value or "").strip()))

    def _project_hint_score(self, row: dict, hint: str) -> int:
        normalized_hint = self._normalize_scope_text(hint)
        if not normalized_hint:
            return 0
        project_name = self._normalize_scope_text(row.get("project_name"))
        project_code_name = self._normalize_scope_text(row.get("project_code_name"))
        if normalized_hint == project_name:
            return 100
        if normalized_hint == project_code_name:
            return 95
        if normalized_hint and normalized_hint in project_name:
            return 80
        if normalized_hint and normalized_hint in project_code_name:
            return 75
        return 0

    def _select_clarification_candidates(self, rows: list[dict], limit: int = 10) -> list[dict]:
        result = []
        seen = set()
        for row in rows:
            candidate = {
                "device": row.get("device"),
                "name": row.get("name"),
                "project_id": row.get("project_id"),
                "project_name": row.get("project_name"),
                "project_code_name": row.get("project_code_name"),
                "tg": row.get("tg"),
                "match_type": row.get("match_type"),
                "match_score": row.get("match_score"),
                "match_reason": row.get("match_reason"),
                "matched_fields": row.get("matched_fields"),
                "scope_mode": row.get("scope_mode"),
                "retrieval_source": row.get("retrieval_source"),
                "confidence_level": row.get("confidence_level"),
                "is_recommended": bool(row.get("is_recommended", False)),
                "recommendation_rank": row.get("recommendation_rank"),
                "decision_mode": row.get("decision_mode"),
            }
            dedupe_key = (candidate["device"], candidate["project_id"], candidate["tg"], candidate["name"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            result.append(candidate)
            if len(result) >= limit:
                break
        return result

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
            "name": f"所有 {device_code} 设备",
            "project_id": "__aggregate_all__",
            "project_name": "跨项目汇总",
            "project_code_name": None,
            "tg": None,
            "match_type": "aggregate_all_option",
            "match_score": None,
            "match_reason": "汇总该设备在所有匹配项目下的数据",
            "matched_fields": ["aggregate_scope"],
            "scope_mode": "aggregate_all",
            "retrieval_source": "aggregate_all",
            "confidence_level": "high",
            "is_recommended": False,
            "recommendation_rank": None,
            "decision_mode": "clarify_required",
        }

    def _build_clarification_message(self, groups: list[dict], resolved_rows: list[dict]) -> str:
        lines: list[str] = []
        for row in resolved_rows:
            device = str(row.get("device") or "").strip()
            name = str(row.get("name") or "").strip()
            project_name = str(row.get("project_name") or "").strip()
            if device and name and project_name:
                lines.append(f"已确认：{device} -> {device}（{name}，项目：{project_name}）")
            elif device and name:
                lines.append(f"已确认：{device} -> {device}（{name}）")

        for group in groups:
            candidates = group.get("candidates") or []
            preview_items = []
            for candidate in candidates[:3]:
                device = str(candidate.get("device") or "").strip()
                name = str(candidate.get("name") or "").strip()
                if device and name:
                    preview_items.append(f"{device}（{name}）")
                elif device:
                    preview_items.append(device)
                elif name:
                    preview_items.append(name)
            preview = "、".join(preview_items)
            keyword = str(group.get("keyword") or "").strip()
            if preview:
                lines.append(f"“{keyword}”匹配到多个候选设备：{preview}。请确认你指的是哪一个。")
            else:
                lines.append(f"“{keyword}”匹配到多个候选设备，请先确认。")
        return "\n".join(line for line in lines if line).strip() or "匹配到多个候选设备，请先确认。"

    def _resolve_exact_code_candidates(self, device_code: str, rows: list[dict], project_hints: list[str]) -> tuple[list[dict], dict | None]:
        normalized_code = self._normalize_scope_text(device_code)
        exact_rows = [row for row in rows if self._normalize_scope_text(row.get("device")) == normalized_code]
        if not exact_rows:
            return rows, None

        scopes = {
            (
                str(row.get("project_id") or ""),
                str(row.get("project_name") or ""),
                str(row.get("project_code_name") or ""),
                str(row.get("tg") or ""),
            )
            for row in exact_rows
        }
        if len(exact_rows) <= 1 or len(scopes) <= 1:
            return exact_rows[:1], None

        if project_hints:
            scored = []
            for row in exact_rows:
                best_score = max((self._project_hint_score(row, hint) for hint in project_hints), default=0)
                if best_score > 0:
                    scored.append((best_score, row))
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                top_score = scored[0][0]
                top_rows = [row for score, row in scored if score == top_score]
                if len(top_rows) == 1:
                    return [top_rows[0]], None

        clarification_rows = []
        for index, row in enumerate(exact_rows, start=1):
            clarification_rows.append({
                **row,
                "match_type": "exact_code_conflict",
                "match_reason": "精确码命中",
                "match_score": row.get("match_score"),
                "recommendation_rank": index,
                "is_recommended": False,
                "decision_mode": "clarify_required",
            })
        clarification_candidates = self._select_clarification_candidates(clarification_rows)
        aggregate_candidate = self._build_aggregate_scope_candidate(device_code, clarification_candidates)
        if aggregate_candidate:
            clarification_candidates.append(aggregate_candidate)
        return [], {"keyword": device_code, "candidates": clarification_candidates, "decision_mode": "clarify_required"}

    def _find_project_match(self, project_hints: list[str]) -> dict | None:
        if not project_hints:
            return None
        try:
            projects = self.metadata_engine.list_projects()
        except Exception:
            return None
        best_project = None
        best_score = 0
        for hint in project_hints:
            normalized_hint = self._normalize_scope_text(hint)
            if not normalized_hint:
                continue
            for project in projects or []:
                project_name = self._normalize_scope_text(project.get("project_name"))
                project_code_name = self._normalize_scope_text(project.get("project_code_name"))
                score = 0
                if normalized_hint == project_name or normalized_hint == project_code_name:
                    score = 100
                elif normalized_hint and (normalized_hint in project_name or normalized_hint in project_code_name):
                    score = 80
                if score > best_score:
                    best_score = score
                    best_project = project
        return best_project if best_score > 0 else None

    def _get_project_devices(self, project_hints: list[str]) -> tuple[list[dict], str]:
        project = self._find_project_match(project_hints)
        if not project:
            return [], ""
        devices = self.metadata_engine.get_devices_by_project(str(project.get("id") or ""))
        rows = [self._device_to_row(device) for device in devices]
        for index, row in enumerate(rows, start=1):
            row.setdefault("retrieval_source", "project_scope")
            row.setdefault("confidence_level", "high")
            row.setdefault("decision_mode", "auto_resolve")
            row.setdefault("recommendation_rank", index)
            row.setdefault("is_recommended", False)
        label = project.get("project_name") or project.get("project_code_name") or project.get("id")
        return rows, f"project_devices:{label}"

    def _narrow_rows_by_project_hints(self, rows: list[dict], project_hints: list[str]) -> list[dict]:
        if len(rows) <= 1 or not project_hints:
            return rows
        scored = []
        for row in rows:
            best_score = max((self._project_hint_score(row, hint) for hint in project_hints), default=0)
            if best_score > 0:
                scored.append((best_score, row))
        if not scored:
            return rows
        scored.sort(key=lambda item: item[0], reverse=True)
        top_score = scored[0][0]
        top_rows = [row for score, row in scored if score == top_score]
        return top_rows or rows

    def _resolve_rows_for_target(self, state: GraphState, target: str, project_hints: list[str]) -> tuple[list[dict], str]:
        alias_row = self._get_alias_memory_row(state, target)
        if alias_row:
            return [alias_row], "session_alias"

        result = self.resolver.resolve(target)
        rows = result.rows
        if project_hints and rows:
            hinted_rows = self._narrow_rows_by_project_hints(rows, project_hints)
            if hinted_rows:
                rows = hinted_rows
        return rows, result.decision_mode

    def __call__(self, state: GraphState) -> GraphState:
        history = list(state.get("history", []))
        intent = build_compat_intent_from_state(state)
        comparison_targets = get_comparison_targets_from_state(state)
        is_comparison = is_comparison_query(state) and len(comparison_targets) > 1
        target = get_primary_target_from_state(state) or str(intent.get("target") or "").strip()

        if is_comparison:
            return self._handle_comparison_query(state, comparison_targets, history)

        if not target:
            return {
                **state,
                "error": "缺少查询目标，无法解析设备范围。",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{"node": NODE_METADATA_MAPPER, "result": "错误: 缺少查询目标"}],
            }

        intent.setdefault("target", target)
        project_hints = get_project_hints_from_state(state)

        try:
            device_rows, decision_mode = self._resolve_rows_for_target(state, target, project_hints)
            sql = "hybrid_resolver"
            if not device_rows and has_device_listing_intent_from_state(state) and project_hints:
                device_rows, sql = self._get_project_devices(project_hints)

            if not device_rows:
                return {
                    **state,
                    "error": f"未找到与“{target}”匹配的设备，请检查设备名称、代号或换一种说法。",
                    "error_node": NODE_METADATA_MAPPER,
                    "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"未找到与“{target}”匹配的设备"}],
                }

            if self._is_explicit_device_code(target):
                device_rows, clarification = self._resolve_exact_code_candidates(target, device_rows, project_hints)
                if clarification:
                    return {
                        **state,
                        "intent": intent,
                        "clarification_required": True,
                        "clarification_candidates": [clarification],
                        "final_response": self._build_clarification_message([clarification], []),
                        "show_table": False,
                        "table_type": None,
                        "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"找到 {len(clarification.get('candidates') or [])} 个候选设备，等待用户确认"}],
                    }

            if has_device_listing_intent_from_state(state):
                device_codes = [row.get("device") for row in device_rows if row.get("device")]
                device_names = {row.get("device"): row.get("name") for row in device_rows if row.get("device")}
                tg_values = [str(row.get("tg")).strip() for row in device_rows if row.get("tg")]
                response = f"已找到 {len(device_rows)} 个相关设备，请查看下表。"
                return {
                    **state,
                    "intent": intent,
                    "device_codes": device_codes,
                    "device_names": device_names,
                    "tg_values": tg_values,
                    "devices": device_rows,
                    "final_response": response,
                    "show_table": True,
                    "table_type": "devices",
                    "query_info": {"devices": device_rows, "target": target, "sql": sql},
                    "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"找到 {len(device_codes)} 个设备候选"}],
                }

            if len(device_rows) > 1:
                clarification = {
                    "keyword": target,
                    "candidates": self._select_clarification_candidates(device_rows),
                    "decision_mode": decision_mode,
                }
                return {
                    **state,
                    "intent": intent,
                    "clarification_required": True,
                    "clarification_candidates": [clarification],
                    "final_response": self._build_clarification_message([clarification], []),
                    "show_table": False,
                    "table_type": None,
                    "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"找到 {len(clarification.get('candidates') or [])} 个候选设备，等待用户确认"}],
                }

            device_codes = [row.get("device") for row in device_rows if row.get("device")]
            device_names = {row.get("device"): row.get("name") for row in device_rows if row.get("device")}
            tg_values = [str(row.get("tg")).strip() for row in device_rows if row.get("tg")]
            logger.info("设备映射完成 [hybrid]: target='%s', 命中 %s 个设备", target, len(device_codes))
            return {
                **state,
                "intent": intent,
                "device_codes": device_codes,
                "device_names": device_names,
                "tg_values": tg_values,
                "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"解析出 {len(device_codes)} 个设备"}],
            }
        except DatabaseConnectionError as exc:
            logger.error("设备映射数据库连接失败: %s", exc)
            return {
                **state,
                "error": f"设备映射数据库连接失败: {str(exc)}",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{"node": NODE_METADATA_MAPPER, "result": "错误: 数据库连接失败"}],
            }
        except MetadataEngineError as exc:
            logger.error("设备映射元数据查询失败: %s", exc)
            return {
                **state,
                "error": f"设备映射元数据查询失败: {str(exc)}",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"错误: {str(exc)}"}],
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("设备映射执行异常: %s", exc)
            return {
                **state,
                "error": f"设备映射执行异常: {str(exc)}",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"错误: {str(exc)}"}],
            }

    def _handle_comparison_query(self, state: GraphState, targets: list[str], history: list) -> GraphState:
        all_device_codes = []
        all_device_names = {}
        all_tg_values = []
        comparison_device_groups = {}
        comparison_scope_groups = {}
        resolved_rows = []
        project_hints = get_project_hints_from_state(state)
        query_text = str(getattr(state.get("query_plan"), "current_question", None) or state.get("query") or "")

        for target in targets:
            device_rows, decision_mode = self._resolve_rows_for_target(state, target, project_hints)
            if not device_rows:
                continue

            if self._is_explicit_device_code(target):
                allow_multi_scope_aggregation = allows_explicit_multi_scope_aggregation(query_text, target)
                device_rows, clarification = self._resolve_exact_code_candidates(target, device_rows, project_hints)
                if clarification and not allow_multi_scope_aggregation:
                    return {
                        **state,
                        "clarification_required": True,
                        "clarification_candidates": [clarification],
                        "device_codes": list(dict.fromkeys(all_device_codes)),
                        "device_names": all_device_names,
                        "tg_values": list(dict.fromkeys(all_tg_values)),
                        "resolved_devices": resolved_rows,
                        "final_response": self._build_clarification_message([clarification], resolved_rows),
                        "show_table": False,
                        "table_type": None,
                        "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"找到 {len(clarification.get('candidates') or [])} 个候选设备，等待用户确认"}],
                    }
            elif len(device_rows) > 1:
                clarification = {
                    "keyword": target,
                    "candidates": self._select_clarification_candidates(device_rows),
                    "decision_mode": decision_mode,
                }
                return {
                    **state,
                    "clarification_required": True,
                    "clarification_candidates": [clarification],
                    "device_codes": list(dict.fromkeys(all_device_codes)),
                    "device_names": all_device_names,
                    "tg_values": list(dict.fromkeys(all_tg_values)),
                    "resolved_devices": resolved_rows,
                    "final_response": self._build_clarification_message([clarification], resolved_rows),
                    "show_table": False,
                    "table_type": None,
                    "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"找到 {len(clarification.get('candidates') or [])} 个候选设备，等待用户确认"}],
                }

            device_codes = [row.get("device") for row in device_rows if row.get("device")]
            comparison_device_groups[target] = device_codes
            comparison_scope_groups[target] = [dict(row) for row in device_rows if isinstance(row, dict)]
            all_device_codes.extend(device_codes)
            resolved_rows.extend(device_rows)
            for row in device_rows:
                if row.get("device"):
                    all_device_names[row.get("device")] = row.get("name")
                if row.get("tg"):
                    all_tg_values.append(str(row.get("tg")).strip())

        if not all_device_codes:
            return {
                **state,
                "error": f"未找到与这些目标匹配的设备: {targets}",
                "error_node": NODE_METADATA_MAPPER,
                "history": history + [{"node": NODE_METADATA_MAPPER, "result": "错误: 未找到匹配设备"}],
            }

        all_device_codes = list(dict.fromkeys(all_device_codes))
        return {
            **state,
            "intent": build_compat_intent_from_state({**state, "comparison_targets": targets, "is_comparison": True}),
            "device_codes": all_device_codes,
            "device_names": all_device_names,
            "tg_values": all_tg_values,
            "is_comparison": True,
            "comparison_targets": list(targets),
            "comparison_device_groups": comparison_device_groups,
            "comparison_scope_groups": comparison_scope_groups,
            "history": history + [{"node": NODE_METADATA_MAPPER, "result": f"对比解析: {len(targets)} 个目标，共 {len(all_device_codes)} 个设备"}],
        }


__all__ = ["MetadataMapperNode"]
