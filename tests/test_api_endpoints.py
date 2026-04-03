import importlib

from fastapi.testclient import TestClient


def _load_app_module():
    module = importlib.import_module("web.app")
    module.SESSION_STATE.clear()
    return module


class _FakeResolutionResult:
    def __init__(self, devices, query_info):
        self._devices = devices
        self.query_info = query_info

    def to_dict_list(self):
        return list(self._devices)


def test_chat_endpoint_returns_steps_and_request_id(monkeypatch) -> None:
    app_module = _load_app_module()
    scripted_events = [
        {"type": "step_start", "step": "thinking", "timestamp_ms": 1700000000000},
        {
            "type": "step_done",
            "step": "thinking",
            "info": "decide get_sensor_data",
            "duration_ms": 11,
            "query_info": {"type": "fast_path", "device_code": "a1_b9"},
        },
        {
            "type": "final_answer",
            "response": "query finished",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9"],
                "start_time": "2026-03-17",
                "end_time": "2026-03-17",
                "data_type": "ep",
            },
            "projects": None,
            "devices": None,
            "analysis": {"mode": "single"},
            "chart_specs": [{"id": "trend-line"}],
            "chart_context": {
                "chartable": True,
                "query_kind": "single_series",
                "recommended_chart_type": "line",
                "follow_up_suggestions": [{"label": "趋势图", "chart_type": "line"}],
            },
            "show_charts": True,
            "total_duration_ms": 31,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    class FakeAgent:
        def run_with_progress(self, _message_with_history):
            yield from scripted_events

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: FakeAgent())

    with TestClient(app_module.app) as client:
        response = client.post("/api/chat", json={"message": "a1_b9 query", "history": []})

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["request_id"]
    assert payload["session_id"]
    assert payload["response"] == "query finished"
    assert payload["query_params"]["device_codes"] == ["a1_b9"]
    assert payload["chart_context"]["query_kind"] == "single_series"
    assert payload["total_duration_ms"] == 31
    assert payload["clarification_required"] is False
    assert payload["steps"][0]["status"] == "done"
    assert payload["steps"][0]["timestamp_ms"] == 1700000000000
    assert payload["steps"][0]["duration_ms"] == 11
    assert payload["steps"][0]["query_info"]["device_code"] == "a1_b9"



def test_chat_endpoint_error_uses_standard_error_payload(monkeypatch) -> None:
    app_module = _load_app_module()

    class ErrorAgent:
        def run_with_progress(self, _message_with_history):
            raise RuntimeError("chat failed")
            yield  # pragma: no cover

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: ErrorAgent())

    with TestClient(app_module.app) as client:
        response = client.post("/api/chat", json={"message": "broken", "history": []})

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert payload["error"] == "chat failed"
    assert payload["error_code"] == "CHAT_ERROR"
    assert payload["request_id"]



def test_query_endpoint_returns_transformed_sensor_rows(monkeypatch) -> None:
    app_module = _load_app_module()

    def fake_fetch(**_kwargs):
        return {
            "success": True,
            "data": [
                {"logTime": "2026-03-17 00:00:01", "device": "a1_b9", "tag": "ep", "val": 123.4}
            ],
            "total_count": 1,
            "page": 1,
            "page_size": 50,
            "total_pages": 1,
            "has_more": False,
            "statistics": {"avg": 123.4},
            "analysis": {"mode": "single"},
            "chart_specs": [{"id": "trend-line"}],
            "chart_context": {
                "chartable": True,
                "query_kind": "single_series",
                "recommended_chart_type": "line",
                "follow_up_suggestions": [{"label": "趋势图", "chart_type": "line"}],
            },
            "show_charts": True,
            "is_sampled": False,
            "aggregation_type": None,
        }

    monkeypatch.setattr(app_module, "fetch_sensor_data_with_components", fake_fetch)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/query",
            json={
                "device_codes": ["a1_b9"],
                "start_time": "2026-03-17",
                "end_time": "2026-03-17",
                "data_type": "ep",
                "page": 1,
                "page_size": 50,
                "user_query": "a1_b9 today",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["request_id"]
    assert payload["empty_result"] is False
    assert payload["total_count"] == 1
    assert payload["data"] == [
        {"time": "2026-03-17 00:00:01", "device": "a1_b9", "tag": "ep", "value": 123.4}
    ]
    assert payload["analysis"]["mode"] == "single"
    assert payload["chart_specs"][0]["id"] == "trend-line"
    assert payload["chart_context"]["recommended_chart_type"] == "line"


def test_query_endpoint_returns_focused_table_when_query_needs_direct_ranking(monkeypatch) -> None:
    app_module = _load_app_module()

    def fake_fetch(**_kwargs):
        return {
            "success": True,
            "data": [
                {"logTime": "2024-01-31 23:00:05", "device": "a1_b9", "tag": "ep", "val": 208484.0},
                {"logTime": "2024-01-31 22:00:04", "device": "a1_b9", "tag": "ep", "val": 208470.0},
            ],
            "total_count": 744,
            "page": 1,
            "page_size": 50,
            "total_pages": 15,
            "has_more": True,
            "statistics": {"avg": 207021.03},
            "analysis": {"mode": "single", "metric": "用电量", "unit": "kWh"},
            "focused_table": {
                "headers": ["排名", "时间", "设备", "标签", "数值"],
                "rows": [
                    {"排名": 1, "时间": "2024-01-31 23:00:05", "设备": "a1_b9", "标签": "ep", "数值": 208484},
                    {"排名": 2, "时间": "2024-01-31 22:00:04", "设备": "a1_b9", "标签": "ep", "数值": 208470},
                ],
                "page_size": 2,
                "total_count": 2,
                "has_more": False,
                "view_label": "问题直答",
            },
            "chart_specs": [],
            "show_charts": False,
            "is_sampled": False,
            "aggregation_type": None,
        }

    monkeypatch.setattr(app_module, "fetch_sensor_data_with_components", fake_fetch)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/query",
            json={
                "device_codes": ["a1_b9"],
                "start_time": "2024-01-01",
                "end_time": "2024-01-31",
                "data_type": "ep",
                "page": 1,
                "page_size": 50,
                "user_query": "找出 a1_b9 设备2024年1月用电量最高的前5个时间点",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["focused_table"]["view_label"] == "问题直答"
    assert payload["focused_table"]["rows"][0]["排名"] == 1
    assert payload["focused_table"]["rows"][0]["时间"] == "2024-01-31 23:00:05"



def test_query_endpoint_empty_result_uses_standard_payload(monkeypatch) -> None:
    app_module = _load_app_module()

    def fake_fetch(**_kwargs):
        return {
            "success": True,
            "data": [],
            "total_count": 0,
            "page": 1,
            "page_size": 50,
            "total_pages": 0,
            "has_more": False,
            "statistics": None,
            "analysis": None,
            "chart_specs": [],
            "show_charts": False,
            "is_sampled": False,
            "aggregation_type": None,
        }

    monkeypatch.setattr(app_module, "fetch_sensor_data_with_components", fake_fetch)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/query",
            json={
                "device_codes": ["a1_b9"],
                "start_time": "2026-03-17",
                "end_time": "2026-03-17",
                "data_type": "ep",
                "page": 1,
                "page_size": 50,
                "user_query": "a1_b9 empty",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["request_id"]
    assert payload["empty_result"] is True
    assert payload["message"] == app_module.EMPTY_RESULT_MESSAGE
    assert payload["error"] is None
    assert payload["error_code"] is None
    assert payload["data"] == []



def test_query_endpoint_failed_result_uses_query_failed_error(monkeypatch) -> None:
    app_module = _load_app_module()

    def fake_fetch(**_kwargs):
        return {"success": False, "error": "upstream failed"}

    monkeypatch.setattr(app_module, "fetch_sensor_data_with_components", fake_fetch)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/query",
            json={
                "device_codes": ["a1_b9"],
                "start_time": "2026-03-17",
                "end_time": "2026-03-17",
                "data_type": "ep",
                "page": 1,
                "page_size": 50,
                "user_query": "a1_b9 failed",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert payload["error"] == "upstream failed"
    assert payload["error_code"] == "QUERY_FAILED"
    assert payload["request_id"]



def test_devices_endpoint_returns_query_info_from_entity_resolver(monkeypatch) -> None:
    app_module = _load_app_module()
    resolution = _FakeResolutionResult(
        devices=[{"device": "a1_b9", "name": "meter a1_b9"}],
        query_info={"resolver": "chroma", "candidate_count": 1},
    )

    monkeypatch.setattr(app_module.entity_resolver, "search_device_candidates", lambda keyword, top_k=50: resolution)

    with TestClient(app_module.app) as client:
        response = client.get("/api/devices", params={"keyword": "a1_b9"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["request_id"]
    assert payload["devices"][0]["device"] == "a1_b9"
    assert payload["query_info"]["resolver"] == "chroma"



def test_devices_endpoint_falls_back_when_entity_resolver_fails(monkeypatch) -> None:
    app_module = _load_app_module()
    device_tool = importlib.import_module("src.tools.device_tool")

    def raise_resolver_error(_keyword, top_k=50):
        raise RuntimeError("resolver unavailable")

    monkeypatch.setattr(app_module.entity_resolver, "search_device_candidates", raise_resolver_error)
    monkeypatch.setattr(
        device_tool,
        "find_device_metadata_with_engine",
        lambda keyword, metadata_engine: [{"device": "fallback-1", "name": f"fallback for {keyword}"}],
    )

    with TestClient(app_module.app) as client:
        response = client.get("/api/devices", params={"keyword": "fallback-keyword"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["request_id"]
    assert payload["devices"][0]["device"] == "fallback-1"
    assert payload.get("query_info") is None



def test_projects_endpoint_success_includes_request_id(monkeypatch) -> None:
    app_module = _load_app_module()
    monkeypatch.setattr(app_module.metadata_engine, "list_projects", lambda: [{"id": "p1", "project_name": "Demo"}])

    with TestClient(app_module.app) as client:
        response = client.get("/api/projects")

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["request_id"]
    assert payload["projects"] == [{"id": "p1", "project_name": "Demo"}]



def test_query_endpoint_passes_tg_values_to_fetcher(monkeypatch) -> None:
    app_module = _load_app_module()
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "data": [],
            "total_count": 0,
            "page": kwargs.get("page", 1),
            "page_size": kwargs.get("page_size", 50),
            "total_pages": 0,
            "has_more": False,
            "statistics": None,
            "analysis": None,
            "chart_specs": [],
            "show_charts": False,
            "is_sampled": False,
            "aggregation_type": None,
        }

    monkeypatch.setattr(app_module, "fetch_sensor_data_with_components", fake_fetch)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/query",
            json={
                "device_codes": ["a1_b9"],
                "tg_values": ["TG232"],
                "start_time": "2026-03-17",
                "end_time": "2026-03-17",
                "data_type": "u_line",
                "page": 1,
                "page_size": 50,
                "user_query": "a1_b9 今天的线电压数据",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["device_codes"] == ["a1_b9"]
    assert captured["tg_values"] == ["TG232"]
    assert captured["data_type"] == "u_line"



def test_query_endpoint_passes_query_plan_to_fetcher_for_phase_tags(monkeypatch) -> None:
    app_module = _load_app_module()
    captured = {}

    def fake_fetch(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "data": [],
            "total_count": 0,
            "page": kwargs.get("page", 1),
            "page_size": kwargs.get("page_size", 50),
            "total_pages": 0,
            "has_more": False,
            "statistics": None,
            "analysis": None,
            "chart_specs": [],
            "show_charts": False,
            "is_sampled": False,
            "aggregation_type": None,
        }

    monkeypatch.setattr(app_module, "fetch_sensor_data_with_components", fake_fetch)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/query",
            json={
                "device_codes": ["a2_b1"],
                "start_time": "2024-01-01",
                "end_time": "2024-01-01",
                "data_type": "u_line",
                "page": 1,
                "page_size": 50,
                "user_query": "a2_b1在2024年1月1日的ua是多少",
                "query_plan": {
                    "current_question": "a2_b1在2024年1月1日的ua是多少",
                    "query_mode": "sensor_query",
                    "inferred_data_type": "u_line",
                    "explicit_device_codes": ["a2_b1"],
                    "search_targets": ["a2_b1"],
                    "has_sensor_intent": True,
                    "raw_plan": {"requested_tags": ["ua"]}
                },
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert captured["query_plan"]["raw_plan"]["requested_tags"] == ["ua"]


def test_delete_chat_history_endpoint_clears_current_user_messages() -> None:
    app_module = _load_app_module()
    app_module.user_memory_store.clear_chat_history(user_id="history-clean-user") if app_module.user_memory_store.list_chat_history(user_id="history-clean-user", limit=1) else None
    app_module.user_memory_store.clear_chat_history(user_id="history-clean-keep") if app_module.user_memory_store.list_chat_history(user_id="history-clean-keep", limit=1) else None
    app_module.user_memory_store.record_chat_message(
        session_id="session-a",
        user_id="history-clean-user",
        role="user",
        message="hello",
        intent_type="chat",
    )
    app_module.user_memory_store.record_chat_message(
        session_id="session-b",
        user_id="history-clean-user",
        role="assistant",
        message="world",
        intent_type="chat",
    )
    app_module.user_memory_store.record_chat_message(
        session_id="session-c",
        user_id="history-clean-keep",
        role="user",
        message="keep",
        intent_type="chat",
    )

    with TestClient(app_module.app) as client:
        response = client.delete("/api/chat/history", params={"user_id": "history-clean-user"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["deleted_count"] >= 2
    assert app_module.user_memory_store.list_chat_history(user_id="history-clean-user", limit=20) == []
    assert len(app_module.user_memory_store.list_chat_history(user_id="history-clean-keep", limit=20)) == 1
