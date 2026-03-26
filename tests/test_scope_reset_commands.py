import importlib


def _load_app_module():
    module = importlib.import_module("web.app")
    module = importlib.reload(module)
    module.SESSION_STATE.clear()
    return module


def test_prepare_chat_context_reset_alias_only_clears_one_alias() -> None:
    app_module = _load_app_module()
    session_id = "scope-reset-one"
    state = app_module._get_session_state(session_id)
    state["device_aliases"] = {
        "a1_b9": {"device": "a1_b9", "name": "设备A", "project_name": "项目A", "alias": "a1_b9"},
        "b1_b14": {"device": "b1_b14", "name": "设备B", "project_name": "项目B", "alias": "b1_b14"},
    }
    state["last_user_query"] = "查询 a1_b9 设备今天的电流数据"

    context = app_module._prepare_chat_context(
        app_module.ChatRequest(
            message="重新确认 a1_b9",
            history=[],
            session_id=session_id,
        )
    )

    assert context["effective_message"] == "查询 a1_b9 设备今天的电流数据"
    assert "a1_b9" not in context["alias_memory"]
    assert "b1_b14" in context["alias_memory"]


def test_prepare_chat_context_clear_scope_reuses_last_question() -> None:
    app_module = _load_app_module()
    session_id = "scope-clear-all"
    state = app_module._get_session_state(session_id)
    state["device_aliases"] = {
        "a1_b9": {"device": "a1_b9", "name": "设备A", "project_name": "项目A", "alias": "a1_b9"},
        "b1_b14": {"device": "b1_b14", "name": "设备B", "project_name": "项目B", "alias": "b1_b14"},
    }
    state["last_user_query"] = "查询 a1_b9 设备今天的电流数据"

    context = app_module._prepare_chat_context(
        app_module.ChatRequest(
            message="清除当前确认范围",
            history=[],
            session_id=session_id,
        )
    )

    assert context["effective_message"] == "查询 a1_b9 设备今天的电流数据"
    assert context["alias_memory"] == {}


def test_prepare_chat_context_switch_project_rebinds_alias(monkeypatch) -> None:
    app_module = _load_app_module()
    session_id = "scope-switch-project"
    state = app_module._get_session_state(session_id)
    state["device_aliases"] = {
        "a1_b9": {"device": "a1_b9", "name": "旧设备", "project_name": "旧项目", "alias": "a1_b9"},
    }
    state["last_user_query"] = "查询 a1_b9 设备今天的电流数据"

    class _Device:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return dict(self.payload)

    def fake_search_devices(keyword):
        assert keyword == "a1_b9"
        return [
            _Device({"device": "a1_b9", "name": "B2柜", "project_id": "p1", "project_name": "中国能建集团数据机房监控项目", "project_code_name": "ceec", "tg": "TG1"}),
            _Device({"device": "a1_b9", "name": "601-612", "project_id": "p2", "project_name": "平陆运河项目", "project_code_name": "plyh", "tg": "TG2"}),
        ], False

    monkeypatch.setattr(app_module.metadata_engine, "search_devices", fake_search_devices)

    context = app_module._prepare_chat_context(
        app_module.ChatRequest(
            message="把 a1_b9 改成中国能建集团数据机房监控项目那个",
            history=[],
            session_id=session_id,
        )
    )

    assert context["effective_message"] == "查询 a1_b9 设备今天的电流数据"
    assert context["alias_memory"]["a1_b9"]["project_name"] == "中国能建集团数据机房监控项目"
    assert context["learned_aliases"][0]["project_name"] == "中国能建集团数据机房监控项目"
