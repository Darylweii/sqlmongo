import importlib
import json
from copy import deepcopy

from fastapi.testclient import TestClient


def _load_app_module():
    module = importlib.import_module("web.app")
    module.SESSION_STATE.clear()
    return module


def _parse_sse_events(response_text: str):
    events = []
    for line in str(response_text or "").splitlines():
        if not line.startswith("data: "):
            continue
        events.append(json.loads(line[6:]))
    return events


def test_chat_stream_emits_complete_event_with_step_metrics(monkeypatch) -> None:
    app_module = _load_app_module()
    scripted_events = [
        {"type": "step_start", "step": "思考中 (第1轮)", "timestamp_ms": 1700000000000},
        {
            "type": "step_done",
            "step": "思考中 (第1轮)",
            "info": "决定: get_sensor_data",
            "duration_ms": 12,
            "query_info": {"type": "fast_path", "device_code": "a1_b9"},
        },
        {
            "type": "final_answer",
            "response": "已完成",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9"],
                "start_time": "2026-03-17",
                "end_time": "2026-03-17",
                "data_type": "ep",
            },
            "analysis": {"mode": "single"},
            "chart_specs": [{"id": "trend-line"}],
            "show_charts": True,
            "total_duration_ms": 34,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    class FakeAgent:
        def run_with_progress(self, _message_with_history):
            yield from scripted_events

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: FakeAgent())

    with TestClient(app_module.app) as client:
        response = client.post("/api/chat/stream", json={"message": "a1_b9 设备今天的用电量", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert [event["type"] for event in events] == ["step_start", "step_done", "answer_delta", "complete"]
    assert events[0]["timestamp_ms"] == 1700000000000
    assert events[1]["duration_ms"] == 12
    assert events[1]["query_info"]["device_code"] == "a1_b9"
    assert events[2]["delta"] == "已完成"
    assert events[3]["success"] is True
    assert events[3]["request_id"]
    assert events[3]["table_type"] == "sensor_data"
    assert events[3]["total_duration_ms"] == 34


def test_chat_stream_error_event_uses_standard_error_payload(monkeypatch) -> None:
    app_module = _load_app_module()

    class ErrorAgent:
        def run_with_progress(self, _message_with_history):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: ErrorAgent())

    with TestClient(app_module.app) as client:
        response = client.post("/api/chat/stream", json={"message": "测试异常", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert len(events) == 1
    assert events[0]["type"] == "error"
    assert events[0]["success"] is False
    assert events[0]["error"] == "boom"
    assert events[0]["error_code"] == "CHAT_STREAM_ERROR"
    assert events[0]["request_id"]
    assert events[0]["session_id"]


def test_chat_stream_alias_confirmation_reuses_original_question_and_tg(monkeypatch) -> None:
    app_module = _load_app_module()
    call_records = []
    candidate = {
        "device": "a1_b9",
        "name": "1#变加联络 AA5-1 稻盛和夫",
        "project_id": "1",
        "project_name": "智慧物联网能效平台",
        "project_code_name": "bjdlzdh",
        "device_type": "meter",
        "tg": "TG232",
        "match_score": 100.0,
        "matched_fields": ["device"],
        "match_reason": "exact_device_code",
    }
    scripted_batches = [
        [
            {
                "type": "final_answer",
                "response": "请确认设备",
                "show_table": False,
                "table_type": "",
                "query_params": None,
                "analysis": None,
                "chart_specs": None,
                "show_charts": False,
                "total_duration_ms": 8,
                "clarification_required": True,
                "clarification_candidates": [{"keyword": "a1_b9", "candidates": [candidate]}],
            }
        ],
        [
            {
                "type": "final_answer",
                "response": "继续查询完成",
                "show_table": False,
                "table_type": "",
                "query_params": {"device_codes": ["a1_b9"]},
                "analysis": None,
                "chart_specs": None,
                "show_charts": False,
                "total_duration_ms": 6,
                "clarification_required": False,
                "clarification_candidates": None,
            }
        ],
    ]

    def fake_factory(**kwargs):
        record = {"alias_memory": deepcopy(kwargs.get("alias_memory") or {})}
        call_records.append(record)
        scripted_events = scripted_batches[len(call_records) - 1]

        class FakeAgent:
            def run_with_progress(self, message_with_history):
                record["message_with_history"] = message_with_history
                yield from scripted_events

        return FakeAgent()

    monkeypatch.setattr(app_module, "_create_chat_agent", fake_factory)

    with TestClient(app_module.app) as client:
        first_response = client.post(
            "/api/chat/stream",
            json={"message": "a1_b9 设备今天的用电量", "history": [], "session_id": "session-1"},
        )
        second_response = client.post(
            "/api/chat/stream",
            json={
                "message": "选这个设备",
                "history": [],
                "session_id": "session-1",
                "alias_confirmation": {
                    "alias": "a1_b9",
                    "keyword": "a1_b9",
                    "device": candidate["device"],
                    "name": candidate["name"],
                    "project_id": candidate["project_id"],
                    "project_name": candidate["project_name"],
                    "project_code_name": candidate["project_code_name"],
                    "device_type": candidate["device_type"],
                    "tg": candidate["tg"],
                    "matched_fields": candidate["matched_fields"],
                    "match_reason": candidate["match_reason"],
                    "device_info": candidate,
                    "original_question": "a1_b9 设备今天的用电量",
                },
            },
        )

    first_events = _parse_sse_events(first_response.text)
    second_events = _parse_sse_events(second_response.text)

    assert first_events[-1]["clarification_required"] is True
    assert first_events[-1]["clarification_candidates"][0]["candidates"][0]["tg"] == "TG232"
    assert second_events[-1]["clarification_required"] is False
    assert second_events[-1]["resolved_scope"]["items"][0]["device"] == "a1_b9"
    assert second_events[-1]["resolved_scope"]["items"][0]["project_name"] == candidate["project_name"]
    assert second_events[-1]["resolved_scope"]["items"][0]["tg"] == "TG232"

    assert len(call_records) == 2
    assert call_records[0]["alias_memory"] == {}
    assert call_records[1]["alias_memory"]["a1_b9"]["tg"] == "TG232"
    assert call_records[1]["message_with_history"] == "a1_b9 设备今天的用电量"


def test_chat_stream_confirm_project_scope_returns_devices_without_replanning(monkeypatch) -> None:
    app_module = _load_app_module()
    create_calls = []

    class _Device:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return dict(self.payload)

    monkeypatch.setattr(
        app_module.metadata_engine,
        "get_devices_by_project",
        lambda project_id: [
            _Device(
                {
                    "device": "dev_001",
                    "name": "测试设备A",
                    "project_id": project_id,
                    "project_name": "测试项目",
                    "project_code_name": "111",
                    "tg": "TG-A",
                }
            ),
            _Device(
                {
                    "device": "dev_002",
                    "name": "测试设备B",
                    "project_id": project_id,
                    "project_name": "测试项目",
                    "project_code_name": "111",
                    "tg": "TG-B",
                }
            ),
        ],
    )

    def fake_factory(**kwargs):
        create_calls.append(kwargs)
        raise AssertionError("project confirmation should not invoke chat agent")

    monkeypatch.setattr(app_module, "_create_chat_agent", fake_factory)

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "选这个项目",
                "history": [],
                "session_id": "project-confirm-1",
                "alias_confirmation": {
                    "action": "confirm_project_scope",
                    "alias": "测试项目",
                    "keyword": "测试项目",
                    "project_id": "51",
                    "project_name": "测试项目",
                    "project_code_name": "111",
                    "original_question": "测试项目有哪些设备",
                },
            },
        )

    events = _parse_sse_events(response.text)
    assert response.status_code == 200
    assert [event["type"] for event in events] == ["step_start", "step_done", "complete"]
    assert events[0]["step"] == "确认项目并列出设备"
    assert events[1]["step"] == "确认项目并列出设备"
    complete = events[-1]
    assert complete["type"] == "complete"
    assert complete["show_table"] is True
    assert complete["table_type"] == "devices"
    assert complete["clarification_required"] is False
    assert [item["device"] for item in complete["devices"]] == ["dev_001", "dev_002"]
    assert complete["resolved_scope"]["device_count"] == 2
    assert complete["original_question"] == "测试项目有哪些设备"
    assert create_calls == []


def test_projects_endpoint_error_uses_standard_error_payload(monkeypatch) -> None:
    app_module = _load_app_module()

    def raise_error():
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(app_module.metadata_engine, "list_projects", raise_error)

    with TestClient(app_module.app) as client:
        response = client.get("/api/projects")

    payload = response.json()
    assert response.status_code == 200
    assert payload["success"] is False
    assert payload["error"] == "metadata unavailable"
    assert payload["error_code"] == "PROJECTS_ERROR"
    assert payload["request_id"]



def test_build_resolved_scope_marks_cross_project_aggregate_scope(monkeypatch) -> None:
    app_module = _load_app_module()

    class _Device:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return dict(self.payload)

    def fake_list_all_devices():
        return [
            _Device({"device": "a1_b9", "name": "B2柜", "project_id": "p1", "project_name": "中国能建集团数据机房监控项目", "project_code_name": "ceec", "tg": "TG1"}),
            _Device({"device": "a1_b9", "name": "601-612", "project_id": "p2", "project_name": "平陆运河项目", "project_code_name": "plyh", "tg": "TG2"}),
            _Device({"device": "b1_b14", "name": "电子楼 AA3-1 电源进线", "project_id": "p3", "project_name": "智慧物联网能效平台", "project_code_name": "iot-energy", "tg": "TG233"}),
        ]

    monkeypatch.setattr(app_module.metadata_engine, "list_all_devices", fake_list_all_devices)

    resolved_scope = app_module._build_resolved_scope(
        {
            "device_codes": ["a1_b9", "a1_b9", "b1_b14"],
            "tg_values": [],
            "user_query": "汇总所有 a1_b9 与 b1_b14 哪个用电更多？",
        },
        alias_memory={},
        learned_aliases=[],
    )

    assert resolved_scope is not None
    assert resolved_scope["aggregation_scope_codes"] == ["a1_b9"]


def test_build_resolved_scope_fills_missing_devices_from_metadata_catalog(monkeypatch) -> None:
    app_module = _load_app_module()

    class _DummyDevice:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return dict(self.payload)

    monkeypatch.setattr(
        app_module.metadata_engine,
        "list_all_devices",
        lambda: [
            _DummyDevice(
                {
                    "device": "a1_b9",
                    "name": "B2柜",
                    "project_id": "p1",
                    "project_name": "中国能建集团数据机房监控项目",
                    "project_code_name": "cneec-room",
                    "tg": "TG232",
                }
            ),
            _DummyDevice(
                {
                    "device": "b1_b14",
                    "name": "电子楼 AA3-1 电源进线",
                    "project_id": "p2",
                    "project_name": "智慧物联网能效平台",
                    "project_code_name": "iot-energy",
                    "tg": "TG233",
                }
            ),
        ],
    )

    resolved_scope = app_module._build_resolved_scope(
        {
            "device_codes": ["a1_b9", "b1_b14"],
            "tg_values": ["TG232", "TG233"],
        },
        alias_memory={
            "a1_b9": {
                "device": "a1_b9",
                "name": "B2柜",
                "project_id": "p1",
                "project_name": "中国能建集团数据机房监控项目",
                "project_code_name": "cneec-room",
                "tg": "TG232",
                "alias": "a1_b9",
                "source": "session_alias",
            }
        },
        learned_aliases=[],
    )

    assert resolved_scope is not None
    assert resolved_scope["device_count"] == 2
    assert resolved_scope["tg_count"] == 2
    assert [item["device"] for item in resolved_scope["items"]] == ["a1_b9", "b1_b14"]
    assert {item["tg"] for item in resolved_scope["items"]} == {"TG232", "TG233"}
    assert resolved_scope["aggregation_scope_codes"] == []



def test_prepare_chat_context_learns_device_code_scope_variant() -> None:
    app_module = _load_app_module()

    request = app_module.ChatRequest(
        message="确认这个范围",
        history=[],
        session_id="session-scope-1",
        alias_confirmation={
            "alias": "B2柜",
            "keyword": "B2柜",
            "device": "a1_b9",
            "name": "B2柜",
            "project_id": "p1",
            "project_name": "中国能建集团数据机房监控项目",
            "project_code_name": "ceec-dc",
            "device_type": "meter",
            "tg": "TG232",
            "original_question": "a1_b9 在2024年1月的电压是多少",
        },
    )

    context = app_module._prepare_chat_context(request)
    alias_memory = context["alias_memory"]

    assert app_module._normalize_alias_key("B2柜") in alias_memory
    assert app_module._normalize_alias_key("a1_b9") in alias_memory
    assert alias_memory[app_module._normalize_alias_key("a1_b9")]["project_name"] == "中国能建集团数据机房监控项目"
    assert alias_memory[app_module._normalize_alias_key("a1_b9")]["tg"] == "TG232"
    assert context["effective_message"] == "a1_b9 在2024年1月的电压是多少"



def test_chat_stream_projects_complete_event_populates_projects_payload(monkeypatch) -> None:
    app_module = _load_app_module()
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已找到 2 个项目，请查看下表。",
            "show_table": True,
            "table_type": "projects",
            "query_params": {},
            "analysis": None,
            "chart_specs": None,
            "show_charts": False,
            "total_duration_ms": 9,
            "clarification_required": False,
            "clarification_candidates": None,
        }
    ]

    class FakeAgent:
        def run_with_progress(self, _message_with_history):
            yield from scripted_events

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: FakeAgent())
    monkeypatch.setattr(
        app_module.metadata_engine,
        "list_projects",
        lambda: [
            {"id": "p1", "project_name": "项目A"},
            {"id": "p2", "project_name": "项目B"},
        ],
    )

    with TestClient(app_module.app) as client:
        response = client.post("/api/chat/stream", json={"message": "有哪些项目可用", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events[-1]["type"] == "complete"
    assert events[-1]["table_type"] == "projects"
    assert events[-1]["projects"] == [
        {"id": "p1", "project_name": "项目A"},
        {"id": "p2", "project_name": "项目B"},
    ]


def test_chat_stream_complete_event_includes_table_preview(monkeypatch) -> None:
    app_module = _load_app_module()
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已返回结果",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a2_b1"],
                "start_time": "2024-01-01",
                "end_time": "2024-01-01",
                "data_type": "ua",
            },
            "analysis": {"mode": "single", "analysis_scope_label": "a2_b1"},
            "chart_specs": [],
            "show_charts": False,
            "table_preview": {
                "success": True,
                "data": [{"time": "2024-01-01 00:00:00", "device": "a2_b1", "tag": "ua", "value": 230.5}],
                "total_count": 864,
                "page": 1,
                "page_size": 50,
                "total_pages": 18,
                "has_more": True,
                "focused_table": {
                    "headers": ["指标", "结果"],
                    "rows": [{"指标": "A相电压 ua 平均值", "结果": "230.50 V"}],
                    "view_label": "问题直答",
                },
            },
            "total_duration_ms": 22,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    class FakeAgent:
        def run_with_progress(self, _message_with_history):
            yield from scripted_events

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: FakeAgent())

    with TestClient(app_module.app) as client:
        response = client.post("/api/chat/stream", json={"message": "a2_b1在2024年1月1日的ua是多少", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    complete_event = events[-1]
    assert complete_event["type"] == "complete"
    assert complete_event["table_preview"] is not None
    assert complete_event["table_preview"]["data"][0]["tag"] == "ua"
    assert complete_event["table_preview"]["focused_table"]["view_label"] == "问题直答"


def test_chat_stream_persists_last_sensor_query_context_for_chart_follow_up(monkeypatch) -> None:
    app_module = _load_app_module()
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已返回结果",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9"],
                "start_time": "2026-04-02",
                "end_time": "2026-04-02",
                "data_type": "i",
                "user_query": "查询 a1_b9 设备今天的电流数据",
            },
            "analysis": {"mode": "single", "analysis_scope_label": "a1_b9"},
            "chart_specs": [{"id": "trend-line"}],
            "chart_context": {
                "chartable": True,
                "query_kind": "single_series",
                "recommended_chart_type": "line",
                "follow_up_suggestions": [{"label": "趋势图", "chart_type": "line"}],
            },
            "show_charts": False,
            "table_preview": None,
            "_chart_cache": {
                "raw_data": [
                    {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ia", "val": 3.2},
                    {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "ia", "val": 5.8},
                ],
                "statistics": {"avg": 4.5, "count": 2, "trend": "平稳", "change_rate": 0.0, "cv": 0.2, "anomaly_count": 0, "anomaly_ratio": 0.0},
                "chart_specs": [{"id": "trend-line"}],
            },
            "total_duration_ms": 12,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    class FakeAgent:
        def run_with_progress(self, _message_with_history):
            yield from scripted_events

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: FakeAgent())

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "查询 a1_b9 设备今天的电流数据", "history": [], "session_id": "chart-session-1"},
        )

    assert response.status_code == 200
    state = app_module._get_session_state("chart-session-1")
    assert state["last_sensor_query_context"]["base_query"] == "查询 a1_b9 设备今天的电流数据"
    assert state["last_sensor_query_context"]["analysis_mode"] == "single"
    assert state["last_sensor_query_context"]["chart_count"] == 1
    assert state["last_sensor_result_cache"]["chart_context"]["recommended_chart_type"] == "line"


def test_chat_stream_chart_follow_up_reuses_session_sensor_cache_without_requery(monkeypatch) -> None:
    app_module = _load_app_module()
    call_count = {"value": 0}
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已返回结果",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9"],
                "tg_values": ["TG232"],
                "start_time": "2026-04-02",
                "end_time": "2026-04-02",
                "data_type": "u_line",
                "user_query": "查询 a1_b9 设备今天的电压数据",
            },
            "analysis": {"mode": "single", "metric": "电压", "unit": "V", "analysis_scope_label": "按三相联合分析"},
            "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            "chart_context": {
                "chartable": True,
                "query_kind": "single_series",
                "recommended_chart_type": "line",
                "follow_up_suggestions": [{"label": "趋势图", "chart_type": "line"}],
            },
            "show_charts": False,
            "table_preview": None,
            "_chart_cache": {
                "raw_data": [
                    {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ua", "val": 233},
                    {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ub", "val": 234},
                    {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "uc", "val": 232},
                    {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "ua", "val": 231},
                    {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "ub", "val": 232},
                    {"logTime": "2026-04-02 01:00:00", "device": "a1_b9", "tag": "uc", "val": 230},
                ],
                "statistics": {"avg": 232.0, "count": 6, "trend": "平稳", "change_rate": -0.5, "cv": 0.8, "anomaly_count": 0, "anomaly_ratio": 0.0},
                "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            },
            "total_duration_ms": 12,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    def fake_factory(**_kwargs):
        call_count["value"] += 1
        if call_count["value"] > 1:
            raise AssertionError("chart follow-up should reuse session cache instead of requerying agent")

        class FakeAgent:
            def run_with_progress(self, _message_with_history):
                yield from scripted_events

        return FakeAgent()

    monkeypatch.setattr(app_module, "_create_chat_agent", fake_factory)

    with TestClient(app_module.app) as client:
        first_response = client.post(
            "/api/chat/stream",
            json={"message": "查询 a1_b9 设备今天的电压数据", "history": [], "session_id": "chart-cache-session"},
        )
        second_response = client.post(
            "/api/chat/stream",
            json={"message": "画一张热力图", "history": [], "session_id": "chart-cache-session"},
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    second_events = _parse_sse_events(second_response.text)
    complete_event = second_events[-1]
    assert call_count["value"] == 1
    assert complete_event["type"] == "complete"
    assert complete_event["show_charts"] is True
    assert complete_event["chart_specs"][0]["chart_type"] == "heatmap"
    assert complete_event["chart_context"]["cache_hit"] is True
    assert "无需重新查询数据库" in complete_event["response"]


def test_chat_stream_comparison_chart_context_preserves_duplicate_slots(monkeypatch) -> None:
    app_module = _load_app_module()
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已完成四项对比",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9", "a2_b1", "a3_b2"],
                "start_time": "2026-04-02",
                "end_time": "2026-04-02",
                "data_type": "u_line",
                "user_query": "对比一下 a1_b9 和 a2_b1 和 a3_b2 以及 a3_b2 的电压数据",
                "query_plan": {
                    "comparison_targets": ["a1_b9", "a2_b1", "a3_b2", "a3_b2"],
                    "search_targets": ["a1_b9", "a2_b1", "a3_b2", "a3_b2"],
                },
                "comparison_scope_groups": {
                    "a1_b9": [{"device": "a1_b9", "name": "设备1", "project_name": "项目A", "tg": "TG1"}],
                    "a2_b1": [{"device": "a2_b1", "name": "设备2", "project_name": "项目B", "tg": "TG2"}],
                    "a3_b2": [{"device": "a3_b2", "name": "设备3", "project_name": "项目C", "tg": "TG3"}],
                },
            },
            "analysis": {"mode": "comparison"},
            "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            "show_charts": False,
            "table_preview": None,
            "_chart_cache": {
                "raw_data": [
                    {"logTime": "2026-04-02 00:00:00", "device": "a1_b9", "tag": "ua", "val": 233, "tg": "TG1"},
                    {"logTime": "2026-04-02 00:00:00", "device": "a2_b1", "tag": "ua", "val": 231, "tg": "TG2"},
                    {"logTime": "2026-04-02 00:00:00", "device": "a3_b2", "tag": "ua", "val": 229, "tg": "TG3"},
                ],
                "statistics": {"avg": 231.0, "count": 3, "trend": "平稳", "change_rate": 0.0, "cv": 0.5, "anomaly_count": 0, "anomaly_ratio": 0.0},
                "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            },
            "total_duration_ms": 18,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    class FakeAgent:
        def run_with_progress(self, _message_with_history):
            yield from scripted_events

    monkeypatch.setattr(app_module, "_create_chat_agent", lambda **_kwargs: FakeAgent())

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={"message": "对比一下 a1_b9 和 a2_b1 和 a3_b2 以及 a3_b2 的电压数据", "history": [], "session_id": "comparison-slot-session"},
        )

    complete_event = _parse_sse_events(response.text)[-1]
    assert complete_event["chart_context"]["comparison_slot_count"] == 4
    assert [slot["raw_target"] for slot in complete_event["chart_context"]["comparison_slots"]] == ["a1_b9", "a2_b1", "a3_b2", "a3_b2"]


def test_chat_stream_button_follow_up_uses_explicit_base_query_cache(monkeypatch) -> None:
    app_module = _load_app_module()
    call_count = {"value": 0}
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已完成对比",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9", "b1_b14", "a2_b3"],
                "tg_values": ["TG232", "TG233"],
                "start_time": "2026-04-01",
                "end_time": "2026-04-03",
                "data_type": "ep",
                "user_query": "对比一下 a1_b9 和 b1_b14 和 a2_b3 的用电量数据",
                "query_plan": {
                    "comparison_targets": ["a1_b9", "b1_b14", "a2_b3"],
                    "search_targets": ["a1_b9", "b1_b14", "a2_b3"],
                },
                "comparison_scope_groups": {
                    "a1_b9": [{"device": "a1_b9", "name": "设备1", "project_name": "项目A", "tg": "TG232"}],
                    "b1_b14": [{"device": "b1_b14", "name": "设备2", "project_name": "项目A", "tg": "TG233"}],
                    "a2_b3": [{"device": "a2_b3", "name": "设备3", "project_name": "项目A", "tg": "TG232"}],
                },
            },
            "analysis": {"mode": "comparison", "metric": "用电量", "unit": "kWh"},
            "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            "show_charts": False,
            "table_preview": None,
            "_chart_cache": {
                "raw_data": [
                    {"logTime": "2026-04-01 00:00:00", "device": "a1_b9", "tag": "ep", "val": 100, "tg": "TG232"},
                    {"logTime": "2026-04-01 01:00:00", "device": "a1_b9", "tag": "ep", "val": 110, "tg": "TG232"},
                    {"logTime": "2026-04-01 00:00:00", "device": "b1_b14", "tag": "ep", "val": 30, "tg": "TG233"},
                    {"logTime": "2026-04-01 01:00:00", "device": "b1_b14", "tag": "ep", "val": 35, "tg": "TG233"},
                    {"logTime": "2026-04-01 00:00:00", "device": "a2_b3", "tag": "ep", "val": 90, "tg": "TG232"},
                    {"logTime": "2026-04-01 01:00:00", "device": "a2_b3", "tag": "ep", "val": 92, "tg": "TG232"},
                ],
                "statistics": {"avg": 76.17, "count": 6, "trend": "平稳", "change_rate": 0.0, "cv": 0.2, "anomaly_count": 0, "anomaly_ratio": 0.0},
                "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            },
            "total_duration_ms": 15,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    def fake_factory(**_kwargs):
        call_count["value"] += 1
        if call_count["value"] > 1:
            raise AssertionError("button chart follow-up should reuse cache instead of requerying agent")

        class FakeAgent:
            def run_with_progress(self, _message_with_history):
                yield from scripted_events

        return FakeAgent()

    monkeypatch.setattr(app_module, "_create_chat_agent", fake_factory)

    with TestClient(app_module.app) as client:
        first_response = client.post(
            "/api/chat/stream",
            json={"message": "对比一下 a1_b9 和 b1_b14 和 a2_b3 的用电量数据", "history": [], "session_id": "button-chart-session"},
        )
        second_response = client.post(
            "/api/chat/stream",
            json={
                "message": "帮我画柱状对比图",
                "history": [],
                "session_id": "button-chart-session",
                "chart_follow_up_base_query": "对比一下 a1_b9 和 b1_b14 和 a2_b3 的用电量数据",
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    second_events = _parse_sse_events(second_response.text)
    complete_event = second_events[-1]
    assert call_count["value"] == 1
    assert complete_event["show_charts"] is True
    assert complete_event["chart_context"]["cache_hit"] is True
    assert complete_event["chart_specs"][0]["chart_type"] == "bar"


def test_chat_stream_button_follow_up_with_base_query_survives_missing_last_query_context() -> None:
    app_module = _load_app_module()
    session_id = "button-chart-missing-last-context"
    base_query = "比较 a1_b9、b1_b14、a2_b3 三个设备的用电情况"
    session_state = app_module._get_session_state(session_id)
    session_state["last_sensor_query_context"] = None
    session_state["last_sensor_result_cache"] = {
        "base_query": base_query,
        "query_params": {
            "device_codes": ["a1_b9", "b1_b14", "a2_b3"],
            "tg_values": ["TG232", "TG233", "TG232"],
            "start_time": "2026-04-01",
            "end_time": "2026-04-03",
            "data_type": "ep",
            "user_query": base_query,
            "comparison_scope_groups": {
                "a1_b9": [{"device": "a1_b9", "name": "设备1", "project_name": "项目A", "tg": "TG232"}],
                "b1_b14": [{"device": "b1_b14", "name": "设备2", "project_name": "项目A", "tg": "TG233"}],
                "a2_b3": [{"device": "a2_b3", "name": "设备3", "project_name": "项目A", "tg": "TG232"}],
            },
        },
        "resolved_scope": None,
        "analysis": {"mode": "comparison", "metric": "用电量", "unit": "kWh"},
        "table_type": "sensor_data",
        "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
        "statistics": {"avg": 76.17, "count": 6, "trend": "平稳", "change_rate": 0.0, "cv": 0.2, "anomaly_count": 0, "anomaly_ratio": 0.0},
        "raw_data": [
            {"logTime": "2026-04-01 00:00:00", "device": "a1_b9", "tag": "ep", "val": 100, "tg": "TG232"},
            {"logTime": "2026-04-01 01:00:00", "device": "a1_b9", "tag": "ep", "val": 110, "tg": "TG232"},
            {"logTime": "2026-04-01 00:00:00", "device": "b1_b14", "tag": "ep", "val": 30, "tg": "TG233"},
            {"logTime": "2026-04-01 01:00:00", "device": "b1_b14", "tag": "ep", "val": 35, "tg": "TG233"},
            {"logTime": "2026-04-01 00:00:00", "device": "a2_b3", "tag": "ep", "val": 90, "tg": "TG232"},
            {"logTime": "2026-04-01 01:00:00", "device": "a2_b3", "tag": "ep", "val": 92, "tg": "TG232"},
        ],
        "normalized_records": [
            {"timestamp": "2026-04-01 00:00:00", "device": "a1_b9", "tag": "ep", "value": 100.0, "tg": "TG232"},
            {"timestamp": "2026-04-01 01:00:00", "device": "a1_b9", "tag": "ep", "value": 110.0, "tg": "TG232"},
            {"timestamp": "2026-04-01 00:00:00", "device": "b1_b14", "tag": "ep", "value": 30.0, "tg": "TG233"},
            {"timestamp": "2026-04-01 01:00:00", "device": "b1_b14", "tag": "ep", "value": 35.0, "tg": "TG233"},
            {"timestamp": "2026-04-01 00:00:00", "device": "a2_b3", "tag": "ep", "value": 90.0, "tg": "TG232"},
            {"timestamp": "2026-04-01 01:00:00", "device": "a2_b3", "tag": "ep", "value": 92.0, "tg": "TG232"},
        ],
        "device_names": {"a1_b9": "设备1", "b1_b14": "设备2", "a2_b3": "设备3"},
        "chart_context": {
            "query_kind": "comparison_series",
            "comparison_slot_count": 3,
            "comparison_slots": [
                {"slot_id": "slot_1", "ordinal": 1, "raw_target": "a1_b9", "resolved_device_code": "a1_b9", "resolved_device_name": "设备1", "project_name": "项目A", "tg": "TG232", "status": "resolved"},
                {"slot_id": "slot_2", "ordinal": 2, "raw_target": "b1_b14", "resolved_device_code": "b1_b14", "resolved_device_name": "设备2", "project_name": "项目A", "tg": "TG233", "status": "resolved"},
                {"slot_id": "slot_3", "ordinal": 3, "raw_target": "a2_b3", "resolved_device_code": "a2_b3", "resolved_device_name": "设备3", "project_name": "项目A", "tg": "TG232", "status": "resolved"},
            ],
        },
        "updated_at": "2026-04-03T14:00:00",
    }

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "帮我画柱状对比图",
                "history": [],
                "session_id": session_id,
                "chart_follow_up_base_query": base_query,
            },
        )

    assert response.status_code == 200
    complete_event = _parse_sse_events(response.text)[-1]
    assert complete_event["success"] is True
    assert complete_event["show_charts"] is True
    assert complete_event["chart_context"]["cache_hit"] is True
    assert complete_event["chart_specs"][0]["chart_type"] == "bar"
    assert complete_event["chart_specs"][0]["option"]


def test_chat_stream_button_follow_up_restores_history_chart_cache_after_restart(monkeypatch) -> None:
    app_module = _load_app_module()
    session_id = "button-chart-history-cache-session"
    base_query = "对比一下 a1_b9 和 b1_b14 和 a2_b3 的用电量数据"
    call_count = {"value": 0}
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已完成对比",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9", "b1_b14", "a2_b3"],
                "tg_values": ["TG232", "TG233", "TG232"],
                "start_time": "2026-04-01",
                "end_time": "2026-04-03",
                "data_type": "ep",
                "user_query": base_query,
                "query_plan": {
                    "comparison_targets": ["a1_b9", "b1_b14", "a2_b3"],
                    "search_targets": ["a1_b9", "b1_b14", "a2_b3"],
                },
                "comparison_scope_groups": {
                    "a1_b9": [{"device": "a1_b9", "name": "设备1", "project_name": "项目A", "tg": "TG232"}],
                    "b1_b14": [{"device": "b1_b14", "name": "设备2", "project_name": "项目A", "tg": "TG233"}],
                    "a2_b3": [{"device": "a2_b3", "name": "设备3", "project_name": "项目A", "tg": "TG232"}],
                },
            },
            "analysis": {"mode": "comparison", "metric": "用电量", "unit": "kWh"},
            "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            "show_charts": False,
            "table_preview": None,
            "_chart_cache": {
                "raw_data": [
                    {"logTime": "2026-04-01 00:00:00", "device": "a1_b9", "tag": "ep", "val": 100, "tg": "TG232"},
                    {"logTime": "2026-04-01 01:00:00", "device": "a1_b9", "tag": "ep", "val": 110, "tg": "TG232"},
                    {"logTime": "2026-04-01 00:00:00", "device": "b1_b14", "tag": "ep", "val": 30, "tg": "TG233"},
                    {"logTime": "2026-04-01 01:00:00", "device": "b1_b14", "tag": "ep", "val": 35, "tg": "TG233"},
                    {"logTime": "2026-04-01 00:00:00", "device": "a2_b3", "tag": "ep", "val": 90, "tg": "TG232"},
                    {"logTime": "2026-04-01 01:00:00", "device": "a2_b3", "tag": "ep", "val": 92, "tg": "TG232"},
                ],
                "statistics": {"avg": 76.17, "count": 6, "trend": "平稳", "change_rate": 0.0, "cv": 0.2, "anomaly_count": 0, "anomaly_ratio": 0.0},
                "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            },
            "total_duration_ms": 15,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    def fake_factory(**_kwargs):
        call_count["value"] += 1
        if call_count["value"] > 1:
            raise AssertionError("history chart cache recovery should not requery agent after restart")

        class FakeAgent:
            def run_with_progress(self, _message_with_history):
                yield from scripted_events

        return FakeAgent()

    monkeypatch.setattr(app_module, "_create_chat_agent", fake_factory)

    with TestClient(app_module.app) as client:
        first_response = client.post(
            "/api/chat/stream",
            json={"message": base_query, "history": [], "session_id": session_id, "user_id": "anonymous-user"},
        )
        assert first_response.status_code == 200

        app_module.SESSION_STATE.clear()

        second_response = client.post(
            "/api/chat/stream",
            json={
                "message": "帮我画柱状对比图",
                "history": [],
                "session_id": session_id,
                "user_id": "anonymous-user",
                "chart_follow_up_base_query": base_query,
            },
        )

    complete_event = _parse_sse_events(second_response.text)[-1]
    assert call_count["value"] == 1
    assert complete_event["show_charts"] is True
    assert complete_event["chart_specs"][0]["chart_type"] == "bar"
    assert complete_event["chart_context"]["cache_hit"] is True
    assert complete_event["chart_context"]["cache_restore_mode"] == "history_cache"
    assert "聊天历史恢复" in complete_event["response"]


def test_chat_stream_button_follow_up_rebuilds_history_query_after_restart(monkeypatch) -> None:
    app_module = _load_app_module()
    session_id = "button-chart-history-query-session"
    base_query = "查询 a1_b9 设备今天的电流数据"
    user_id = "anonymous-user"

    app_module.chat_memory_service.record_chat_message(
        session_id=session_id,
        user_id=user_id,
        role="assistant",
        message="历史查询结果",
        intent_type="data_query",
        message_meta={
            "response": "历史查询结果",
            "original_question": base_query,
            "intent_type": "data_query",
            "query_params": {
                "device_codes": ["a1_b9"],
                "tg_values": ["TG19"],
                "start_time": "2026-04-03",
                "end_time": "2026-04-03",
                "data_type": "i",
                "page": 1,
                "page_size": 50,
                "user_query": base_query,
            },
            "analysis": {"mode": "single"},
            "chart_specs": [],
            "chart_context": None,
            "show_charts": False,
        },
    )
    app_module.SESSION_STATE.clear()

    class FakeSensorResult:
        data = [
            {"logTime": "2026-04-03 00:00:00", "device": "a1_b9", "tag": "ia", "val": 3.6, "tg": "TG19"},
            {"logTime": "2026-04-03 00:00:00", "device": "a1_b9", "tag": "ib", "val": 5.0, "tg": "TG19"},
            {"logTime": "2026-04-03 00:00:00", "device": "a1_b9", "tag": "ic", "val": 3.1, "tg": "TG19"},
            {"logTime": "2026-04-03 01:00:00", "device": "a1_b9", "tag": "ia", "val": 2.9, "tg": "TG19"},
            {"logTime": "2026-04-03 01:00:00", "device": "a1_b9", "tag": "ib", "val": 4.8, "tg": "TG19"},
            {"logTime": "2026-04-03 01:00:00", "device": "a1_b9", "tag": "ic", "val": 3.4, "tg": "TG19"},
        ]
        total_count = 6
        is_sampled = False
        failed_collections = []
        query_info = {}
        statistics = {"avg": 3.8, "count": 6, "trend": "平稳", "change_rate": 0.0, "cv": 0.2, "anomaly_count": 0, "anomaly_ratio": 0.0}
        page = 1
        page_size = 0
        total_pages = 1
        has_more = False

    monkeypatch.setattr(app_module.data_fetcher, "fetch_sync", lambda **_kwargs: FakeSensorResult())
    monkeypatch.setattr(app_module.metadata_engine, "list_all_devices", lambda: [])
    monkeypatch.setattr(
        app_module,
        "_create_chat_agent",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("history query rebuild should not invoke chat agent")),
    )

    with TestClient(app_module.app) as client:
        response = client.post(
            "/api/chat/stream",
            json={
                "message": "帮我画热力图",
                "history": [],
                "session_id": session_id,
                "user_id": user_id,
                "chart_follow_up_base_query": base_query,
            },
        )

    complete_event = _parse_sse_events(response.text)[-1]
    assert complete_event["show_charts"] is True
    assert complete_event["chart_specs"][0]["chart_type"] == "heatmap"
    assert complete_event["chart_context"]["cache_restore_mode"] == "history_query"
    assert "重新加载数据" in complete_event["response"]


def test_chat_stream_button_follow_up_reuses_cache_when_base_query_only_differs_in_punctuation(monkeypatch) -> None:
    app_module = _load_app_module()
    call_count = {"value": 0}
    scripted_events = [
        {
            "type": "final_answer",
            "response": "已完成对比分析",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a1_b9", "b1_b14", "a2_b3"],
                "data_type": "ep",
                "start_time": "2026-04-01 00:00:00",
                "end_time": "2026-04-01 23:59:59",
                "user_query": "比较 a1_b9、b1_b14、a2_b3 三个设备的用电情况",
            },
            "analysis": {"mode": "comparison", "metric": "用电量", "unit": "kWh"},
            "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            "show_charts": False,
            "table_preview": None,
            "_chart_cache": {
                "raw_data": [
                    {"logTime": "2026-04-01 00:00:00", "device": "a1_b9", "tag": "ep", "val": 100, "tg": "TG232"},
                    {"logTime": "2026-04-01 01:00:00", "device": "a1_b9", "tag": "ep", "val": 110, "tg": "TG232"},
                    {"logTime": "2026-04-01 00:00:00", "device": "b1_b14", "tag": "ep", "val": 30, "tg": "TG233"},
                    {"logTime": "2026-04-01 01:00:00", "device": "b1_b14", "tag": "ep", "val": 35, "tg": "TG233"},
                    {"logTime": "2026-04-01 00:00:00", "device": "a2_b3", "tag": "ep", "val": 90, "tg": "TG232"},
                    {"logTime": "2026-04-01 01:00:00", "device": "a2_b3", "tag": "ep", "val": 92, "tg": "TG232"},
                ],
                "statistics": {"avg": 76.17, "count": 6, "trend": "平稳", "change_rate": 0.0, "cv": 0.2, "anomaly_count": 0, "anomaly_ratio": 0.0},
                "chart_specs": [{"id": "trend-line", "chart_type": "line"}],
            },
            "total_duration_ms": 15,
            "clarification_required": False,
            "clarification_candidates": None,
        },
    ]

    def fake_factory(**_kwargs):
        call_count["value"] += 1
        if call_count["value"] > 1:
            raise AssertionError("chart follow-up should reuse cache even when base query punctuation differs")

        class FakeAgent:
            def run_with_progress(self, _message_with_history):
                yield from scripted_events

        return FakeAgent()

    monkeypatch.setattr(app_module, "_create_chat_agent", fake_factory)

    with TestClient(app_module.app) as client:
        first_response = client.post(
            "/api/chat/stream",
            json={"message": "比较 a1_b9、b1_b14、a2_b3 三个设备的用电情况", "history": [], "session_id": "button-chart-punctuation-session"},
        )
        second_response = client.post(
            "/api/chat/stream",
            json={
                "message": "帮我画柱状对比图",
                "history": [],
                "session_id": "button-chart-punctuation-session",
                "chart_follow_up_base_query": "比较a1_b9,b1_b14,a2_b3三个设备的用电情况",
            },
        )

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    second_events = _parse_sse_events(second_response.text)
    complete_event = second_events[-1]
    assert call_count["value"] == 1
    assert complete_event["show_charts"] is True
    assert complete_event["chart_context"]["cache_hit"] is True
    assert complete_event["chart_specs"][0]["chart_type"] == "bar"
