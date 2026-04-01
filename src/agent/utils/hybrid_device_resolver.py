from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional

from src.metadata.metadata_engine import MetadataEngine
from src.semantic_layer.device_search import DeviceSemanticSearch


logger = logging.getLogger(__name__)


@dataclass
class HybridResolveResult:
    rows: List[Dict[str, Any]]
    decision_mode: str
    retrieval_summary: str
    used_semantic: bool = False


class HybridDeviceResolver:
    def __init__(
        self,
        metadata_engine: MetadataEngine,
        device_search: Optional[DeviceSemanticSearch] = None,
        enable_semantic_fallback: bool = True,
        semantic_top_k: int = 12,
        semantic_min_score: float = 0.35,
    ):
        self.metadata_engine = metadata_engine
        self.enable_semantic_fallback = enable_semantic_fallback
        self.semantic_top_k = semantic_top_k
        self.semantic_min_score = semantic_min_score
        self._device_search = device_search
        self._semantic_init_attempted = device_search is not None

    def _normalize_text(self, value: object) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def _compact_text(self, value: object) -> str:
        return re.sub(r"\s+", "", self._normalize_text(value))

    def _is_short_or_elliptical(self, target: str) -> bool:
        compact = self._compact_text(target)
        if len(compact) <= 6:
            return True
        has_code = bool(re.search(r"[a-z]+\d+|\d+[a-z]+", compact, re.IGNORECASE))
        return not has_code and len(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]+", compact)) <= 2

    def _lexical_score_to_unit(self, score: float) -> float:
        return max(0.0, min(float(score or 0.0) / 120.0, 1.0))

    def _should_use_semantic(self, target: str, lexical_rows: List[Dict[str, Any]]) -> bool:
        if not self.enable_semantic_fallback:
            return False
        if not lexical_rows:
            return True
        if self._is_short_or_elliptical(target):
            return True
        if len(lexical_rows) >= 6:
            return True
        if len(lexical_rows) >= 2:
            top_score = self._lexical_score_to_unit(lexical_rows[0].get("match_score") or lexical_rows[0].get("score") or 0.0)
            next_score = self._lexical_score_to_unit(lexical_rows[1].get("match_score") or lexical_rows[1].get("score") or 0.0)
            if top_score - next_score < 0.18:
                return True
        return False

    def _ensure_device_search(self) -> Optional[DeviceSemanticSearch]:
        if self._device_search is not None:
            return self._device_search
        if self._semantic_init_attempted:
            return None
        self._semantic_init_attempted = True
        try:
            search = DeviceSemanticSearch()
            if search.initialize():
                self._device_search = search
                return self._device_search
        except Exception as exc:  # noqa: BLE001
            logger.warning("hybrid.resolver.semantic_init_failed error=%s", exc)
        return None

    def _device_to_row(self, device) -> Dict[str, Any]:
        if hasattr(device, "to_dict"):
            payload = device.to_dict()
            payload.setdefault("device", getattr(device, "device", ""))
            payload.setdefault("name", getattr(device, "name", ""))
            payload.setdefault("device_type", getattr(device, "device_type", ""))
            payload.setdefault("project_id", getattr(device, "project_id", ""))
            payload.setdefault("project_name", getattr(device, "project_name", None))
            payload.setdefault("project_code_name", getattr(device, "project_code_name", None))
            payload.setdefault("tg", getattr(device, "tg", None))
            payload.setdefault("match_score", payload.get("match_score", getattr(device, "score", 0.0)))
            payload.setdefault("retrieval_source", getattr(device, "retrieval_source", "lexical"))
            return payload
        return {
            "device": getattr(device, "device", ""),
            "name": getattr(device, "name", ""),
            "device_type": getattr(device, "device_type", ""),
            "project_id": getattr(device, "project_id", ""),
            "project_name": getattr(device, "project_name", None),
            "project_code_name": getattr(device, "project_code_name", None),
            "tg": getattr(device, "tg", None),
            "match_score": getattr(device, "score", 0.0),
            "matched_fields": list(getattr(device, "matched_fields", []) or []),
            "match_reason": getattr(device, "match_reason", None),
            "retrieval_source": getattr(device, "retrieval_source", "lexical"),
        }

    def _semantic_row(self, item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "device": item.get("device_id"),
            "name": item.get("device_name"),
            "device_type": item.get("device_type") or "",
            "project_id": str(item.get("project_id") or ""),
            "project_name": item.get("project_name"),
            "project_code_name": item.get("project_code_name"),
            "tg": item.get("tg"),
            "match_score": round(float(item.get("score") or 0.0), 4),
            "matched_fields": list(item.get("matched_fields") or []),
            "match_reason": item.get("match_reason") or "语义匹配",
            "retrieval_source": "semantic",
            "match_type": "semantic",
        }

    def _merge_rows(self, lexical_rows: List[Dict[str, Any]], semantic_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[tuple, Dict[str, Any]] = {}
        for row in lexical_rows + semantic_rows:
            key = (
                str(row.get("device") or "").strip(),
                str(row.get("project_id") or "").strip(),
                str(row.get("name") or "").strip(),
                str(row.get("tg") or "").strip(),
            )
            if not key[0]:
                continue
            current = merged.get(key)
            if current is None:
                merged[key] = dict(row)
                continue

            current_source = str(current.get("retrieval_source") or "lexical")
            next_source = str(row.get("retrieval_source") or "lexical")
            current_sources = {part.strip() for part in current_source.split(",") if part.strip()}
            current_sources.update({part.strip() for part in next_source.split(",") if part.strip()})
            current["retrieval_source"] = "merged" if len(current_sources) > 1 else next(iter(current_sources), "lexical")
            current["match_score"] = max(float(current.get("match_score") or 0.0), float(row.get("match_score") or 0.0))
            current["matched_fields"] = list(dict.fromkeys(list(current.get("matched_fields") or []) + list(row.get("matched_fields") or [])))
            current_reason = [str(current.get("match_reason") or "").strip(), str(row.get("match_reason") or "").strip()]
            current["match_reason"] = "；".join([item for item in dict.fromkeys(current_reason) if item])
            if current.get("match_type") != "semantic" and row.get("match_type") == "semantic":
                current["match_type"] = "merged"
        return list(merged.values())

    def _apply_rank_metadata(self, rows: List[Dict[str, Any]]) -> HybridResolveResult:
        if not rows:
            return HybridResolveResult(rows=[], decision_mode="clarify_required", retrieval_summary="no_match", used_semantic=False)

        def rank_score(row: Dict[str, Any]) -> float:
            source = str(row.get("retrieval_source") or "lexical")
            raw = float(row.get("match_score") or 0.0)
            if source == "semantic":
                return raw
            if source == "merged":
                return min(max(raw, 0.0), 1.0) if raw <= 1.0 else min(raw / 100.0, 1.0)
            return self._lexical_score_to_unit(raw)

        ranked = sorted(
            rows,
            key=lambda row: (
                -rank_score(row),
                str(row.get("device") or ""),
                str(row.get("name") or ""),
                str(row.get("project_id") or ""),
            ),
        )

        top = rank_score(ranked[0])
        second = rank_score(ranked[1]) if len(ranked) > 1 else 0.0
        gap = top - second
        if len(ranked) == 1 and top >= 0.75:
            decision_mode = "auto_resolve"
        elif len(ranked) > 1 and top >= 0.7 and gap >= 0.18:
            decision_mode = "recommend_confirm"
        else:
            decision_mode = "clarify_required"

        used_semantic = any(str(row.get("retrieval_source") or "") in {"semantic", "merged"} for row in ranked)
        for index, row in enumerate(ranked, start=1):
            score = rank_score(row)
            row["confidence_level"] = "high" if score >= 0.75 else ("medium" if score >= 0.5 else "low")
            row["decision_mode"] = decision_mode
            row["recommendation_rank"] = index
            row["is_recommended"] = bool(index == 1 and len(ranked) > 1 and decision_mode == "recommend_confirm")
            if row.get("retrieval_source") == "lexical":
                row.setdefault("match_type", "lexical")

        summary = f"lexical={sum(1 for row in ranked if row.get('retrieval_source') == 'lexical')} semantic={sum(1 for row in ranked if row.get('retrieval_source') == 'semantic')} merged={sum(1 for row in ranked if row.get('retrieval_source') == 'merged')}"
        return HybridResolveResult(rows=ranked, decision_mode=decision_mode, retrieval_summary=summary, used_semantic=used_semantic)

    def resolve(self, target: str) -> HybridResolveResult:
        lexical_devices, _ = self.metadata_engine.search_devices(target)
        lexical_rows = [self._device_to_row(device) for device in lexical_devices]

        semantic_rows: List[Dict[str, Any]] = []
        if self._should_use_semantic(target, lexical_rows):
            device_search = self._ensure_device_search()
            if device_search is not None:
                try:
                    semantic_rows = [
                        self._semantic_row(item)
                        for item in device_search.search_devices(
                            query=target,
                            top_k=self.semantic_top_k,
                            min_score=self.semantic_min_score,
                        )
                    ]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("hybrid.resolver.semantic_search_failed target=%s error=%s", target, exc)

        merged_rows = self._merge_rows(lexical_rows, semantic_rows)
        return self._apply_rank_metadata(merged_rows)

