from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional
from uuid import uuid4


SEMANTIC_RULES_PATH = Path("data/semantic_alias_rules.json")
SEMANTIC_RULES_HISTORY_PATH = Path("data/semantic_alias_rules_history.json")
_ALLOWED_SCOPE_TYPES = {"system", "project", "user"}
_ALLOWED_STATUS = {"enabled", "disabled"}
_SCOPE_PRIORITY = {"user": 300, "project": 200, "system": 100}


@dataclass
class SemanticRuleTestResult:
    query: str
    normalized_query: str
    matched_rules: List[Dict[str, Any]]
    skipped_rules: List[Dict[str, Any]]
    skip_reasons: List[str]
    applied_rule: Optional[Dict[str, Any]]
    device_candidates: List[Dict[str, Any]]
    next_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "normalized_query": self.normalized_query,
            "matched_rules": self.matched_rules,
            "skipped_rules": self.skipped_rules,
            "skip_reasons": self.skip_reasons,
            "applied_rule": self.applied_rule,
            "device_candidates": self.device_candidates,
            "next_action": self.next_action,
        }



class SemanticRuleStore:
    def __init__(
        self,
        rules_path: Path = SEMANTIC_RULES_PATH,
        history_path: Path = SEMANTIC_RULES_HISTORY_PATH,
    ) -> None:
        self.rules_path = Path(rules_path)
        self.history_path = Path(history_path)
        self._lock = Lock()
        self._ensure_store()

    def _ensure_store(self) -> None:
        self.rules_path.parent.mkdir(parents=True, exist_ok=True)
        sample_rules = [
            {
                "id": uuid4().hex,
                "canonical_term": "空调",
                "alias_terms": ["冷气", "制冷", "冷风机"],
                "entity_type": "device_type",
                "target_value": "空调设备",
                "scope_type": "system",
                "scope_value": "global",
                "priority": 90,
                "status": "enabled",
                "notes": "系统默认将冷气归一化到空调设备",
                "created_by": "system",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
            {
                "id": uuid4().hex,
                "canonical_term": "火锅店",
                "alias_terms": ["百年渝府火锅店", "渝府火锅"],
                "entity_type": "device",
                "target_value": "a1_b5",
                "scope_type": "project",
                "scope_value": "智慧物联网能效平台",
                "priority": 95,
                "status": "enabled",
                "notes": "项目专属店铺别名",
                "created_by": "system",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
            {
                "id": uuid4().hex,
                "canonical_term": "冷气",
                "alias_terms": ["空调"],
                "entity_type": "preference",
                "target_value": "空调",
                "scope_type": "user",
                "scope_value": "S001",
                "priority": 100,
                "status": "enabled",
                "notes": "用户个人偏好叫法",
                "created_by": "S001",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            },
        ]
        if not self.rules_path.exists():
            self.rules_path.write_text(json.dumps(sample_rules, ensure_ascii=False, indent=2), encoding="utf-8")
        if not self.history_path.exists():
            self.history_path.write_text("[]", encoding="utf-8")
        self._bootstrap_history_if_empty()

    def _load_rules(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.rules_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_rules(self, rules: List[Dict[str, Any]]) -> None:
        self.rules_path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_history(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_history(self, history: List[Dict[str, Any]]) -> None:
        self.history_path.write_text(json.dumps(history[-500:], ensure_ascii=False, indent=2), encoding="utf-8")

    def _bootstrap_history_if_empty(self) -> None:
        history = self._load_history()
        if history:
            return
        rules = self._load_rules()
        if not rules:
            return

        seeded_history: List[Dict[str, Any]] = []
        for rule in rules:
            timestamp = str(rule.get("created_at") or rule.get("updated_at") or datetime.now().isoformat())
            actor = str(rule.get("created_by") or rule.get("updated_by") or "system")
            seeded_history.append(
                {
                    "id": uuid4().hex,
                    "action": "create",
                    "actor": actor,
                    "timestamp": timestamp,
                    "rule_id": rule.get("id"),
                    "canonical_term": rule.get("canonical_term"),
                    "scope_type": rule.get("scope_type"),
                    "scope_value": rule.get("scope_value"),
                    "status": rule.get("status"),
                    "source": "bootstrap",
                }
            )

        seeded_history.sort(key=lambda item: str(item.get("timestamp") or ""))
        self._save_history(seeded_history)

    def _append_history(self, action: str, rule: Dict[str, Any], actor: str = "system") -> None:
        history = self._load_history()
        history.append(
            {
                "id": uuid4().hex,
                "action": action,
                "actor": actor,
                "timestamp": datetime.now().isoformat(),
                "rule_id": rule.get("id"),
                "canonical_term": rule.get("canonical_term"),
                "scope_type": rule.get("scope_type"),
                "scope_value": rule.get("scope_value"),
                "status": rule.get("status"),
            }
        )
        self._save_history(history)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return re.sub(r"\s+", "", str(value or "").strip().lower())

    @staticmethod
    def _normalize_alias_terms(alias_terms: Any) -> List[str]:
        values = alias_terms if isinstance(alias_terms, list) else [alias_terms]
        result: List[str] = []
        seen = set()
        for item in values:
            text = str(item or "").strip()
            normalized = SemanticRuleStore._normalize_text(text)
            if not text or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(text)
        return result

    def _sanitize_rule(self, payload: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        base = deepcopy(existing) if existing else {}
        canonical_term = str(payload.get("canonical_term") or base.get("canonical_term") or "").strip()
        alias_terms = self._normalize_alias_terms(payload.get("alias_terms") if "alias_terms" in payload else base.get("alias_terms", []))
        entity_type = str(payload.get("entity_type") or base.get("entity_type") or "device").strip() or "device"
        target_value = str(payload.get("target_value") or base.get("target_value") or canonical_term).strip()
        scope_type = str(payload.get("scope_type") or base.get("scope_type") or "system").strip().lower()
        scope_value = str(payload.get("scope_value") or base.get("scope_value") or ("global" if scope_type == "system" else "")).strip()
        priority = int(payload.get("priority") if payload.get("priority") is not None else base.get("priority", 50))
        status = str(payload.get("status") or base.get("status") or "enabled").strip().lower()
        notes = str(payload.get("notes") or base.get("notes") or "").strip()
        created_by = str(payload.get("created_by") or base.get("created_by") or payload.get("updated_by") or "system").strip() or "system"

        if not canonical_term:
            raise ValueError("canonical_term 不能为空")
        if not alias_terms:
            raise ValueError("alias_terms 至少需要一个别名")
        if scope_type not in _ALLOWED_SCOPE_TYPES:
            raise ValueError("scope_type 仅支持 system / project / user")
        if status not in _ALLOWED_STATUS:
                    raise ValueError("status ??? enabled / disabled")

        return {
            "id": base.get("id") or uuid4().hex,
            "canonical_term": canonical_term,
            "alias_terms": alias_terms,
            "entity_type": entity_type,
            "target_value": target_value,
            "scope_type": scope_type,
            "scope_value": scope_value or ("global" if scope_type == "system" else ""),
            "priority": priority,
            "status": status,
            "notes": notes,
            "created_by": base.get("created_by") or created_by,
            "created_at": base.get("created_at") or now,
            "updated_at": now,
        }

    def list_rules(
        self,
        keyword: str = "",
        scope_type: str = "",
        entity_type: str = "",
        status: str = "",
        scope_value: str = "",
    ) -> List[Dict[str, Any]]:
        rules = self._load_rules()
        keyword_norm = self._normalize_text(keyword)
        scope_type = str(scope_type or "").strip().lower()
        entity_type = str(entity_type or "").strip().lower()
        status = str(status or "").strip().lower()
        scope_value_norm = self._normalize_text(scope_value)

        filtered: List[Dict[str, Any]] = []
        for rule in rules:
            haystack = [
                rule.get("canonical_term", ""),
                rule.get("target_value", ""),
                rule.get("scope_value", ""),
                *list(rule.get("alias_terms") or []),
            ]
            if keyword_norm and not any(keyword_norm in self._normalize_text(item) for item in haystack):
                continue
            if scope_type and str(rule.get("scope_type") or "").strip().lower() != scope_type:
                continue
            if entity_type and str(rule.get("entity_type") or "").strip().lower() != entity_type:
                continue
            if status and str(rule.get("status") or "").strip().lower() != status:
                continue
            if scope_value_norm and scope_value_norm not in self._normalize_text(rule.get("scope_value")):
                continue
            filtered.append(rule)

        filtered.sort(key=lambda item: (-int(item.get("priority") or 0), str(item.get("canonical_term") or ""), str(item.get("scope_value") or "")))
        return filtered

    def create_rule(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            rules = self._load_rules()
            rule = self._sanitize_rule(payload)
            rules.append(rule)
            self._save_rules(rules)
            self._append_history("create", rule, actor=str(payload.get("created_by") or "system"))
            return rule

    def update_rule(self, rule_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            rules = self._load_rules()
            for idx, rule in enumerate(rules):
                if str(rule.get("id")) != str(rule_id):
                    continue
                updated = self._sanitize_rule(payload, existing=rule)
                rules[idx] = updated
                self._save_rules(rules)
                self._append_history("update", updated, actor=str(payload.get("updated_by") or payload.get("created_by") or "system"))
                return updated
        raise KeyError("rule_not_found")

    def toggle_rule(self, rule_id: str, actor: str = "system", status: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            rules = self._load_rules()
            for idx, rule in enumerate(rules):
                if str(rule.get("id")) != str(rule_id):
                    continue
                next_status = str(status or ("disabled" if rule.get("status") == "enabled" else "enabled")).strip().lower()
                if next_status not in _ALLOWED_STATUS:
                    raise ValueError("status ??? enabled / disabled")
                updated = dict(rule)
                updated["status"] = next_status
                updated["updated_at"] = datetime.now().isoformat()
                rules[idx] = updated
                self._save_rules(rules)
                self._append_history("toggle", updated, actor=actor)
                return updated
        raise KeyError("rule_not_found")

    def delete_rule(self, rule_id: str, actor: str = "system") -> Dict[str, Any]:
        with self._lock:
            rules = self._load_rules()
            for idx, rule in enumerate(rules):
                if str(rule.get("id")) != str(rule_id):
                    continue
                removed = rules.pop(idx)
                removed["updated_at"] = datetime.now().isoformat()
                self._save_rules(rules)
                self._append_history("delete", removed, actor=actor)
                return removed
        raise KeyError("rule_not_found")

    def history(self, limit: int = 100) -> List[Dict[str, Any]]:
        self._bootstrap_history_if_empty()
        history = self._load_history()
        history.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)
        return history[: max(1, min(int(limit or 100), 500))]

    def test_match(
        self,
        query: str,
        metadata_engine,
        user_id: str = "",
        project_name: str = "",
        limit: int = 10,
    ) -> SemanticRuleTestResult:
        query_text = str(query or "").strip()
        query_norm = self._normalize_text(query_text)
        if not query_text:
            raise ValueError("query 不能为空")

        rules = [rule for rule in self._load_rules() if str(rule.get("status") or "enabled").lower() == "enabled"]
        matched: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        skip_reasons: List[str] = []
        user_norm = self._normalize_text(user_id)
        project_norm = self._normalize_text(project_name)

        def add_skip_reason(reason: str) -> None:
            if reason and reason not in skip_reasons:
                skip_reasons.append(reason)

        for rule in rules:
            scope_type = str(rule.get("scope_type") or "system").lower()
            scope_value = str(rule.get("scope_value") or "").strip()
            scope_value_norm = self._normalize_text(scope_value)
            aliases = [str(rule.get("canonical_term") or "").strip(), *list(rule.get("alias_terms") or [])]
            hit_terms = [alias for alias in aliases if alias and self._normalize_text(alias) in query_norm]
            if not hit_terms:
                continue

            skip_reason = ""
            if scope_type == "user" and scope_value_norm and scope_value_norm != user_norm:
                if not user_norm:
                    skip_reason = "命中了用户级规则，但当前没有填写用户 ID"
                else:
                    expected_scope = scope_value or "未设置"
                    skip_reason = f"命中了用户级规则，但当前用户与规则作用域不一致（期望：{expected_scope}）"
            elif scope_type == "project" and scope_value_norm and scope_value_norm != project_norm:
                if not project_norm:
                    skip_reason = "命中了项目级规则，但当前没有填写项目名称"
                else:
                    expected_scope = scope_value or "未设置"
                    skip_reason = f"命中了项目级规则，但当前项目与规则作用域不一致（期望：{expected_scope}）"

            if skip_reason:
                skipped_rule = dict(rule)
                skipped_rule["matched_terms"] = sorted(hit_terms, key=len, reverse=True)
                skipped_rule["skip_reason"] = skip_reason
                skipped.append(skipped_rule)
                add_skip_reason(skip_reason)
                continue

            scored = dict(rule)
            scored["matched_terms"] = sorted(hit_terms, key=len, reverse=True)
            scored["scope_score"] = _SCOPE_PRIORITY.get(scope_type, 0)
            scored["match_score"] = scored["scope_score"] + int(rule.get("priority") or 0) + max(len(hit_terms[0]), 1)
            matched.append(scored)

        matched.sort(key=lambda item: (-int(item.get("match_score") or 0), -int(item.get("priority") or 0), str(item.get("canonical_term") or "")))
        matched = matched[: max(1, min(int(limit or 10), 20))]
        skipped = skipped[: max(1, min(int(limit or 10), 20))] if skipped else []

        normalized_query = query_text
        for rule in matched:
            canonical = str(rule.get("canonical_term") or "").strip()
            for term in sorted(rule.get("matched_terms") or [], key=len, reverse=True):
                if term and canonical and term in normalized_query:
                    normalized_query = normalized_query.replace(term, canonical)

        applied_rule = matched[0] if matched else None
        device_candidates: List[Dict[str, Any]] = []
        next_action = "no_match"
        if applied_rule:
            target_query = str(applied_rule.get("target_value") or applied_rule.get("canonical_term") or "").strip()
            try:
                devices, _ = metadata_engine.search_devices(target_query)
                for device in devices[:5]:
                    payload = device.to_dict() if hasattr(device, "to_dict") else {
                        "device": getattr(device, "device", ""),
                        "name": getattr(device, "name", ""),
                        "project_name": getattr(device, "project_name", None),
                        "project_code_name": getattr(device, "project_code_name", None),
                        "tg": getattr(device, "tg", None),
                    }
                    device_candidates.append(payload)
            except Exception:
                device_candidates = []

            if len(device_candidates) == 1:
                next_action = "can_auto_filter"
            elif len(device_candidates) > 1:
                next_action = "needs_confirmation"
            else:
                next_action = "normalize_only"
        elif skipped:
            next_action = "rule_skipped"

        return SemanticRuleTestResult(
            query=query_text,
            normalized_query=normalized_query,
            matched_rules=matched,
            skipped_rules=skipped,
            skip_reasons=skip_reasons,
            applied_rule=applied_rule,
            device_candidates=device_candidates,
            next_action=next_action,
        )
