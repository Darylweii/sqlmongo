from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from src.memory_rewrite import MemoryCommandRewriter


class ChatMemoryService:
    MEMORY_COMMAND_FILLERS = [
        "请记住以后查询",
        "请记住以后",
        "请记住",
        "以后记住",
        "以后查询",
        "查询",
        "以后",
        "请",
    ]
    MEMORY_TARGET_FILLERS = [
        "的意思",
        "这个意思",
        "这个叫法",
        "这个名称",
        "这个说法",
    ]

    def _normalize_memory_phrase(self, value: Any, *, is_target: bool = False) -> str:
        text = self.clean_memory_text(value)
        if not text:
            return ""
        changed = True
        while changed:
            changed = False
            for filler in self.MEMORY_COMMAND_FILLERS:
                if text.startswith(filler) and len(text) > len(filler):
                    text = text[len(filler):].strip()
                    changed = True
        if is_target:
            for filler in self.MEMORY_TARGET_FILLERS:
                if text.endswith(filler) and len(text) > len(filler):
                    text = text[:-len(filler)].strip()
        return text.strip("\uFF0C\u3002,.\uFF1A:?\uFF1B;\"\'\u201c\u201d")

    def __init__(
        self,
        *,
        memory_store,
        semantic_rule_store,
        normalize_alias_key: Callable[[str], str],
        lookup_device_by_code: Callable[[str], Optional[dict]],
        build_device_snapshot: Callable[..., Optional[dict]],
        logger,
        llm_memory_rewrite: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self.memory_store = memory_store
        self.semantic_rule_store = semantic_rule_store
        self.normalize_alias_key = normalize_alias_key
        self.lookup_device_by_code = lookup_device_by_code
        self.build_device_snapshot = build_device_snapshot
        self.logger = logger
        self.memory_rewriter = MemoryCommandRewriter(
            normalize_alias_key=normalize_alias_key,
            llm_rewrite=llm_memory_rewrite,
        )

    def resolve_chat_user_id(self, request_user_id: Optional[str], session_id: str) -> str:
        user_id = str(request_user_id or "").strip()
        if not user_id:
            return "anonymous-user"
        if self.LEGACY_ANONYMOUS_USER_PATTERN.fullmatch(user_id):
            return "anonymous-user"
        return user_id

    def infer_current_project_scope(self, alias_memory: Optional[Dict[str, Any]] = None) -> str:
        alias_map = alias_memory if isinstance(alias_memory, dict) else {}
        project_values = set()
        for item in alias_map.values():
            if not isinstance(item, dict):
                continue
            project_name = str(item.get("project_name") or item.get("project_code_name") or "").strip()
            if project_name:
                project_values.add(project_name)
        return next(iter(project_values)) if len(project_values) == 1 else ""

    def memory_scope_label(self, scope_type: str, scope_value: str) -> str:
        if str(scope_type or "").strip().lower() == "project":
            return f"仅当前项目（{scope_value or '未命名项目'}）"
        return "所有项目"

    def clean_memory_text(self, value: Any) -> str:
        return str(value or "").strip().strip('"').strip("'").replace("“", "").replace("”", "")

    def memory_view(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        scope_type = str(raw.get("scope_type") or "global").strip().lower()
        scope_value = str(raw.get("scope_value") or "global").strip()
        return {
            "id": raw.get("id"),
            "alias_text": raw.get("alias_text"),
            "canonical_text": raw.get("canonical_text"),
            "target_type": raw.get("target_type"),
            "target_value": raw.get("target_value"),
            "scope_type": scope_type,
            "scope_value": scope_value,
            "scope_label": self.memory_scope_label(scope_type, scope_value),
            "status": raw.get("status") or "enabled",
            "source": raw.get("source") or "chat_command",
            "updated_at": raw.get("updated_at"),
            "created_at": raw.get("created_at"),
        }

    def extract_memory_command(self, message: str) -> Optional[dict]:
        content = str(message or "").strip()
        if not content:
            return None

        create_command = self.memory_rewriter.rewrite_create_command(content)
        if create_command:
            return create_command

        list_patterns = [
            re.compile(r"^(?P<target>.+?)有哪些常用叫法[？?]?$", re.IGNORECASE),
            re.compile(r"^(?:查看|列出)(?:我的)?(?:长期记忆|叫法)[？?]?$", re.IGNORECASE),
        ]
        delete_patterns = [
            re.compile(r"^(?:删除|忘掉)(?P<alias>.+?)=(?P<target>.+?)(?:这个叫法)?[？?]?$", re.IGNORECASE),
            re.compile(r"^(?:删除|忘掉)(?P<alias>.+?)(?:这个叫法)?[？?]?$", re.IGNORECASE),
        ]
        for pattern in delete_patterns:
            match = pattern.search(content)
            if match:
                return {
                    "intent": "delete_alias_memory",
                    "alias_text": self.clean_memory_text(match.group("alias")),
                    "target_text": self.clean_memory_text(match.groupdict().get("target")),
                }
        for pattern in list_patterns:
            match = pattern.search(content)
            if match:
                return {"intent": "list_alias_memories", "target_text": self.clean_memory_text(match.groupdict().get("target"))}
        return None

    def classify_chat_intent(self, message: str) -> str:
        if self.extract_memory_command(message):
            return "memory_command"
        normalized = str(message or "").strip()
        query_keywords = ["查询", "多少", "对比", "电压", "电流", "电量", "功率", "耗电", "今天", "昨日", "本月"]
        return "data_query" if any(keyword in normalized for keyword in query_keywords) else "chat"

    def validate_memory_target(self, target_text: str) -> Dict[str, Any]:
        target = str(target_text or "").strip()
        if not target:
            raise ValueError("系统理解不能为空")
        if re.fullmatch(r"[a-zA-Z]\d*_[a-zA-Z0-9_]+", target):
            device_payload = self.lookup_device_by_code(target)
            if not isinstance(device_payload, dict):
                raise ValueError(f"未找到设备 {target}，无法保存为长期记忆")
            return {"canonical_text": target, "target_type": "device_code", "target_value": target, "device_payload": device_payload}
        return {"canonical_text": target, "target_type": "canonical_term", "target_value": target, "device_payload": None}

    def replace_alias_in_text(self, text: str, alias_text: str, canonical_text: str) -> str:
        source = str(text or "")
        alias = str(alias_text or "").strip()
        canonical = str(canonical_text or "").strip()
        if not source or not alias or not canonical:
            return source
        return source.replace(alias, canonical) if alias in source else source

    def build_user_memory_snapshot(self, memory_item: Dict[str, Any]) -> Optional[dict]:
        if not isinstance(memory_item, dict):
            return None
        device_code = str(memory_item.get("target_value") or "").strip()
        if not device_code:
            return None
        device_payload = self.lookup_device_by_code(device_code)
        if not isinstance(device_payload, dict):
            return None
        return self.build_device_snapshot(device_payload, alias=memory_item.get("alias_text"), source="user_memory")

    def apply_user_memories(self, message: str, *, user_id: str, alias_memory: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        effective_message = str(message or "")
        merged_alias_memory = dict(alias_memory or {})
        applied_items: List[dict] = []
        invalid_items: List[dict] = []
        project_scope = self.infer_current_project_scope(alias_memory)
        try:
            memories = self.memory_store.resolve_message_memories(user_id=user_id, message=effective_message, project_scope_value=project_scope)
        except Exception as exc:
            self.logger.warning("memory.resolve.failed user_id=%s error=%s", user_id, exc)
            memories = []
        for memory in memories:
            target_type = str(memory.get("target_type") or "canonical_term").strip()
            alias_text = str(memory.get("alias_text") or "").strip()
            canonical_text = str(memory.get("canonical_text") or memory.get("target_value") or "").strip()
            if target_type == "canonical_term":
                updated_message = self.replace_alias_in_text(effective_message, alias_text, canonical_text)
                if updated_message != effective_message:
                    effective_message = updated_message
                    applied_items.append(self.memory_view(memory))
                continue
            if target_type == "device_code":
                snapshot = self.build_user_memory_snapshot(memory)
                if snapshot is None:
                    invalid_items.append(self.memory_view(memory))
                    continue
                merged_alias_memory[self.normalize_alias_key(alias_text)] = snapshot
                applied_items.append(self.memory_view(memory))
        return {"effective_message": effective_message, "alias_memory": merged_alias_memory, "applied_items": applied_items, "invalid_items": invalid_items, "project_scope": project_scope}

    def extract_memory_suggestion_alias(self, message: str) -> str:
        content = str(message or "").strip()
        patterns = [
            re.compile(r"^(?:查询|查一下|查|看看|获取)?(?P<alias>.+?)(?:设备)?(?:今天|昨日|本月|的数据|数据|电流|电压|电量|用电量).*$"),
            re.compile(r"^(?P<alias>.+?)(?:设备)?(?:今天|昨日|本月|的数据|数据|电流|电压|电量|用电量).*$"),
        ]
        for pattern in patterns:
            match = pattern.search(content)
            if match:
                alias = self.clean_memory_text(match.group("alias"))
                if alias and len(alias) <= 24:
                    return alias
        return ""

    def build_memory_suggestion(self, message: str, *, user_id: str, alias_memory: Optional[Dict[str, Any]] = None, query_params: Optional[dict] = None, clarification_required: bool = False) -> Optional[dict]:
        if isinstance(query_params, dict) or clarification_required:
            return None
        alias_text = self.extract_memory_suggestion_alias(message)
        if not alias_text:
            return None
        alias_key = self.normalize_alias_key(alias_text)
        if alias_key in {self.normalize_alias_key(key) for key in (alias_memory or {}).keys()}:
            return None
        try:
            existing = self.memory_store.list_alias_memories(user_id=user_id, keyword=alias_text, limit=20)
        except Exception:
            existing = []
        for item in existing:
            if self.normalize_alias_key(item.get("alias_text")) == alias_key:
                return None
        rules = self.semantic_rule_store.list_rules(keyword=alias_text, status="enabled")
        recommended = None
        for rule in rules:
            aliases = [str(rule.get("canonical_term") or "").strip(), *list(rule.get("alias_terms") or [])]
            if any(alias_key == self.normalize_alias_key(term) for term in aliases if str(term or "").strip()):
                recommended = str(rule.get("canonical_term") or rule.get("target_value") or "").strip()
                break
        if not recommended:
            return None
        project_scope = self.infer_current_project_scope(alias_memory)
        scope_options = []
        if project_scope:
            scope_options.append({"scope_type": "project", "scope_value": project_scope, "label": self.memory_scope_label("project", project_scope), "recommended": True})
        scope_options.append({"scope_type": "global", "scope_value": "global", "label": self.memory_scope_label("global", "global"), "recommended": not bool(project_scope)})
        return {"alias_text": alias_text, "recommended_canonical": recommended, "scope_options": scope_options, "actions": ["remember", "query_once", "dismiss"]}

    def record_chat_message(self, *, session_id: str, user_id: str, role: str, message: str, intent_type: str, message_meta: Optional[Dict[str, Any]] = None) -> None:
        try:
            self.memory_store.record_chat_message(
                session_id=session_id,
                user_id=user_id,
                role=role,
                message=message,
                intent_type=intent_type,
                message_meta=message_meta,
            )
        except Exception as exc:
            self.logger.warning("chat.message.persist.failed session_id=%s user_id=%s error=%s", session_id, user_id, exc)

    def resolve_memory_scope_options(self, alias_memory: Optional[Dict[str, Any]] = None) -> List[dict]:
        project_scope = self.infer_current_project_scope(alias_memory)
        options = []
        if project_scope:
            options.append({"scope_type": "project", "scope_value": project_scope, "label": self.memory_scope_label("project", project_scope), "recommended": True})
        options.append({"scope_type": "global", "scope_value": "global", "label": self.memory_scope_label("global", "global"), "recommended": not bool(project_scope)})
        return options

    def handle_memory_command(self, *, request_id: str, session_id: str, user_id: str, message: str, alias_memory: Optional[Dict[str, Any]] = None, alias_confirmation: Optional[dict] = None) -> dict:
        command = self.extract_memory_command(message)
        action_payload = alias_confirmation if isinstance(alias_confirmation, dict) else {}
        if action_payload.get("action") == "create_memory":
            alias_text = str(action_payload.get("alias_text") or "").strip()
            canonical_text = str(action_payload.get("canonical_text") or "").strip()
            scope_type = str(action_payload.get("scope_type") or "global").strip().lower()
            scope_value = str(action_payload.get("scope_value") or ("global" if scope_type == "global" else "")).strip()
            target = self.validate_memory_target(canonical_text)
            stored = self.memory_store.upsert_alias_memory(user_id=user_id, alias_text=alias_text, canonical_text=target.get("canonical_text"), target_type=target.get("target_type"), target_value=target.get("target_value"), scope_type=scope_type, scope_value=scope_value, source=str(action_payload.get("source") or "chat_command"))
            continue_query = bool(action_payload.get("continue_query"))
            original_question = str(action_payload.get("original_question") or "").strip()
            response_text = f'已记住：以后“{alias_text}”默认按“{stored.get("canonical_text") or canonical_text}”理解\n生效范围：{self.memory_scope_label(scope_type, scope_value)}'
            payload = {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "created", "message": response_text, "items": [self.memory_view(stored)], "continue_query": continue_query, "original_question": original_question}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
            if continue_query and original_question:
                payload["memory_action_result"]["follow_up"] = {"type": "query_once", "alias_text": alias_text, "canonical_text": stored.get("canonical_text") or canonical_text, "original_question": original_question}
            return payload
        if action_payload.get("action") == "delete_memory":
            deleted = self.memory_store.delete_alias_memory(memory_id=int(action_payload.get("memory_id")), user_id=user_id)
            response_text = f'已删除常用叫法：{deleted.get("alias_text")} = {deleted.get("canonical_text")}'
            return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "deleted", "items": [self.memory_view(deleted)], "message": response_text}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
        if not command:
            return {}
        intent = command.get("intent")
        if intent == "create_alias_memory":
            alias_text = self.clean_memory_text(command.get("alias_text"))
            target_text = self.clean_memory_text(command.get("target_text"))
            target = self.validate_memory_target(target_text)
            scope_options = self.resolve_memory_scope_options(alias_memory)
            if len(scope_options) == 1 and scope_options[0].get("scope_type") == "global":
                stored = self.memory_store.upsert_alias_memory(user_id=user_id, alias_text=alias_text, canonical_text=target.get("canonical_text"), target_type=target.get("target_type"), target_value=target.get("target_value"), scope_type="global", scope_value="global", source="chat_command")
                response_text = f'已记住：以后“{alias_text}”默认按“{stored.get("canonical_text") or target_text}”理解\n生效范围：所有项目'
                return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "created", "items": [self.memory_view(stored)], "message": response_text}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
            response_text = f'我可以记住“{alias_text} = {target.get("canonical_text") or target_text}”。请选择生效范围。'
            return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "scope_confirmation", "alias_text": alias_text, "canonical_text": target.get("canonical_text") or target_text, "target_type": target.get("target_type"), "target_value": target.get("target_value"), "scope_options": scope_options, "message": response_text}, "memory_scope_confirmation_required": True, "memory_scope_options": scope_options, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
        if intent == "list_alias_memories":
            target_text = self.clean_memory_text(command.get("target_text"))
            items = self.memory_store.list_alias_memories(user_id=user_id, canonical_text=target_text, limit=50) if target_text else self.memory_store.list_alias_memories(user_id=user_id, limit=50)
            display_items = [self.memory_view(item) for item in items]
            response_text = f'“{target_text}”当前有 {len(display_items)} 个常用叫法。' if target_text else f'当前共记住 {len(display_items)} 个常用叫法。'
            return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "listed", "items": display_items, "message": response_text}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
        if intent == "delete_alias_memory":
            alias_text = self.clean_memory_text(command.get("alias_text"))
            target_text = self.clean_memory_text(command.get("target_text"))
            items = self.memory_store.list_alias_memories(user_id=user_id, keyword=alias_text, limit=50)
            matched = []
            for item in items:
                alias_match = self.normalize_alias_key(item.get("alias_text")) == self.normalize_alias_key(alias_text)
                target_match = (not target_text) or self.normalize_alias_key(item.get("canonical_text")) == self.normalize_alias_key(target_text) or self.normalize_alias_key(item.get("target_value")) == self.normalize_alias_key(target_text)
                if alias_match and target_match:
                    matched.append(item)
            if not matched:
                response_text = f'没有找到“{alias_text}”相关的长期记忆。'
                return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "not_found", "items": [], "message": response_text}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
            if len(matched) == 1:
                deleted = self.memory_store.delete_alias_memory(memory_id=int(matched[0].get("id")), user_id=user_id)
                response_text = f'已删除常用叫法：{deleted.get("alias_text")} = {deleted.get("canonical_text")}'
                return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "deleted", "items": [self.memory_view(deleted)], "message": response_text}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
            response_text = f'找到了 {len(matched)} 条同名常用叫法，请确认删除哪一条。'
            return {"success": True, "request_id": request_id, "session_id": session_id, "response": response_text, "intent_type": "memory_command", "memory_action_result": {"action": "delete_confirmation", "items": [self.memory_view(item) for item in matched], "message": response_text}, "query_params": None, "resolved_scope": None, "projects": None, "devices": None, "clarification_required": False, "clarification_candidates": None}
        return {}

    def apply_query_once_alias(self, alias_confirmation: Optional[dict], fallback_message: str) -> str:
        payload = alias_confirmation if isinstance(alias_confirmation, dict) else {}
        if str(payload.get("action") or "").strip().lower() != "query_once":
            return str(fallback_message or "")
        original_question = str(payload.get("original_question") or fallback_message or "").strip()
        alias_text = str(payload.get("alias_text") or "").strip()
        canonical_text = str(payload.get("canonical_text") or "").strip()
        return self.replace_alias_in_text(original_question, alias_text, canonical_text)

    def build_memory_effect_payload(self, applied_items: List[dict], invalid_items: List[dict]) -> Optional[dict]:
        if not applied_items and not invalid_items:
            return None
        return {"action": "applied", "applied_items": applied_items, "invalid_items": invalid_items, "message": "已结合长期记忆理解本轮问题" if applied_items else "检测到部分长期记忆已失效，已回退到默认匹配"}
    LEGACY_ANONYMOUS_USER_PATTERN = re.compile(r"^user-[0-9a-f-]+$", re.IGNORECASE)
