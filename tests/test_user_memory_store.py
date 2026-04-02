from pathlib import Path

from src.user_memory_store import UserMemoryStore


def test_user_memory_store_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "app_memory.sqlite3"
    store = UserMemoryStore(db_path)

    created = store.upsert_alias_memory(
        user_id="u1",
        alias_text="lengqi",
        canonical_text="aircon",
        target_type="canonical_term",
        target_value="aircon",
        scope_type="global",
        scope_value="global",
    )
    assert created["alias_text"] == "lengqi"

    items = store.list_alias_memories(user_id="u1")
    assert len(items) == 1
    assert items[0]["canonical_text"] == "aircon"

    updated = store.update_alias_memory(memory_id=created["id"], user_id="u1", canonical_text="cooling")
    assert updated["canonical_text"] == "cooling"

    deleted = store.delete_alias_memory(memory_id=created["id"], user_id="u1")
    assert deleted["alias_text"] == "lengqi"
    assert store.list_alias_memories(user_id="u1") == []


def test_list_chat_sessions_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "app_memory.sqlite3"
    store = UserMemoryStore(db_path)

    store.record_chat_message(session_id="s1", user_id="u1", role="user", message="查询 a1_b9 设备今天的电流数据", intent_type="data_query")
    store.record_chat_message(session_id="s1", user_id="u1", role="assistant", message="已返回查询结果", intent_type="data_query")
    store.record_chat_message(session_id="s2", user_id="u1", role="user", message="新建对话", intent_type="chat")

    sessions = store.list_chat_sessions(user_id="u1")
    assert len(sessions) == 2
    assert sessions[0]["session_id"] == "s2"
    assert sessions[1]["title"].startswith("查询 a1_b9")
    assert sessions[1]["message_count"] == 2



def test_chat_history_preserves_message_meta(tmp_path: Path) -> None:
    db_path = tmp_path / "app_memory.sqlite3"
    store = UserMemoryStore(db_path)

    store.record_chat_message(
        session_id="s-meta",
        user_id="u-meta",
        role="assistant",
        message="已找到 2 个相关设备，请查看下表。",
        intent_type="data_query",
        message_meta={
            "response": "已找到 2 个相关设备，请查看下表。",
            "show_table": True,
            "table_type": "devices",
            "devices": [{"device": "a1_b9", "name": "B2柜", "project_name": "项目A"}],
        },
    )

    items = store.list_chat_history(session_id="s-meta", user_id="u-meta", limit=10)
    assert len(items) == 1
    assert items[0]["message_meta"]["show_table"] is True
    assert items[0]["message_meta"]["table_type"] == "devices"
    assert items[0]["message_meta"]["devices"][0]["device"] == "a1_b9"
