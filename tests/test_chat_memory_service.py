from pathlib import Path

from src.chat_memory_service import ChatMemoryService
from src.semantic_rules import SemanticRuleStore
from src.user_memory_store import UserMemoryStore


class _Logger:
    def warning(self, *args, **kwargs):
        return None


def _build_service(tmp_path: Path) -> ChatMemoryService:
    store = UserMemoryStore(tmp_path / "app_memory.sqlite3")
    return ChatMemoryService(
        memory_store=store,
        semantic_rule_store=SemanticRuleStore(),
        normalize_alias_key=lambda value: str(value or "").strip().lower(),
        lookup_device_by_code=lambda code: {"device": code, "name": code, "project_name": "P1"},
        build_device_snapshot=lambda device, alias=None, source="user_memory": {"device": device["device"], "name": device.get("name"), "project_name": device.get("project_name"), "alias": alias, "source": source},
        logger=_Logger(),
    )


def test_chat_memory_service_command_roundtrip(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    create_message = "\u628a\u51b7\u6c14\u8bb0\u6210\u7a7a\u8c03"
    create_result = service.handle_memory_command(
        request_id="r1",
        session_id="s1",
        user_id="u1",
        message=create_message,
        alias_memory={"x": {"project_name": "P1"}},
        alias_confirmation={
            "action": "create_memory",
            "alias_text": "\u51b7\u6c14",
            "canonical_text": "\u7a7a\u8c03",
            "scope_type": "global",
            "scope_value": "global",
        },
    )
    assert create_result["memory_action_result"]["action"] == "created"

    list_result = service.handle_memory_command(
        request_id="r2",
        session_id="s1",
        user_id="u1",
        message="\u7a7a\u8c03\u6709\u54ea\u4e9b\u5e38\u7528\u53eb\u6cd5\uff1f",
        alias_memory=None,
        alias_confirmation=None,
    )
    assert list_result["memory_action_result"]["action"] == "listed"
    assert len(list_result["memory_action_result"]["items"]) == 1

    delete_result = service.handle_memory_command(
        request_id="r3",
        session_id="s1",
        user_id="u1",
        message="\u5220\u9664\u201c\u51b7\u6c14=\u7a7a\u8c03\u201d\u8fd9\u4e2a\u53eb\u6cd5",
        alias_memory=None,
        alias_confirmation=None,
    )
    assert delete_result["memory_action_result"]["action"] == "deleted"



def test_chat_memory_service_understands_colloquial_memory_commands(tmp_path: Path) -> None:
    service = _build_service(tmp_path)

    cases = [
        ("\u8bf7\u8bb0\u4f4f\u4ee5\u540e\u67e5\u8be2\u51b7\u6c14\u4ee3\u8868\u7a7a\u8c03\u7684\u610f\u601d", "\u51b7\u6c14", "\u7a7a\u8c03"),
        ("\u4ee5\u540e\u8bb0\u4f4f\u4e03\u5c42\u52a8\u529b\u4ee3\u8868\u52a8\u529b\u8868", "\u4e03\u5c42\u52a8\u529b", "\u52a8\u529b\u8868"),
        ("\u4ee5\u540e\u8bb0\u4f4f\u4e03\u5c42\u52a8\u529b\u53ef\u4ee5\u53eb\u7269\u8054\u7f51\u7814\u53d1\u90e8\u95e8\u52a8\u529b\u8868", "\u4e03\u5c42\u52a8\u529b", "\u7269\u8054\u7f51\u7814\u53d1\u90e8\u95e8\u52a8\u529b\u8868"),
    ]

    for message, expected_alias, expected_target in cases:
        command = service.extract_memory_command(message)
        assert command is not None
        assert command["intent"] == "create_alias_memory"
        assert command["alias_text"] == expected_alias
        assert command["target_text"] == expected_target


def test_chat_memory_service_defaults_user_id_to_anonymous(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    assert service.resolve_chat_user_id("", "session-1") == "anonymous-user"
    assert service.resolve_chat_user_id(None, "session-1") == "anonymous-user"
    assert service.resolve_chat_user_id("user-1234abcd", "session-1") == "anonymous-user"
    assert service.resolve_chat_user_id("u1", "session-1") == "u1"
