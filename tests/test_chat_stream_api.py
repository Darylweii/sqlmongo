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
            _Device({"device": "a1_b9", "name": "B2?", "project_id": "p1", "project_name": "??????????????", "project_code_name": "ceec", "tg": "TG1"}),
            _Device({"device": "a1_b9", "name": "601-612", "project_id": "p2", "project_name": "??????", "project_code_name": "plyh", "tg": "TG2"}),
            _Device({"device": "b1_b14", "name": "??? AA3-1 ????", "project_id": "p3", "project_name": "?????????", "project_code_name": "iot-energy", "tg": "TG233"}),
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
                    "name": "B2?",
                    "project_id": "p1",
                    "project_name": "??????????????",
                    "project_code_name": "cneec-room",
                    "tg": "TG232",
                }
            ),
            _DummyDevice(
                {
                    "device": "b1_b14",
                    "name": "??? AA3-1 ????",
                    "project_id": "p2",
                    "project_name": "?????????",
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
                "name": "B2?",
                "project_id": "p1",
                "project_name": "??????????????",
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
        message="????",
        history=[],
        session_id="session-scope-1",
        alias_confirmation={
            "alias": "B2?",
            "keyword": "B2?",
            "device": "a1_b9",
            "name": "B2?",
            "project_id": "p1",
            "project_name": "??????????????",
            "project_code_name": "ceec-dc",
            "device_type": "meter",
            "tg": "TG232",
            "original_question": "a1_b9 ??2024?1????????",
        },
    )

    context = app_module._prepare_chat_context(request)
    alias_memory = context["alias_memory"]

    assert app_module._normalize_alias_key("B2?") in alias_memory
    assert app_module._normalize_alias_key("a1_b9") in alias_memory
    assert alias_memory[app_module._normalize_alias_key("a1_b9")]["project_name"] == "??????????????"
    assert alias_memory[app_module._normalize_alias_key("a1_b9")]["tg"] == "TG232"
    assert context["effective_message"] == "a1_b9 ??2024?1????????"



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
            "response": "????",
            "show_table": True,
            "table_type": "sensor_data",
            "query_params": {
                "device_codes": ["a2_b1"],
                "start_time": "2024-01-01",
                "end_time": "2024-01-01",
                "data_type": "ua",
            },
            "analysis": {"mode": "single", "analysis_scope_label": "?????"},
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
                    "headers": ["??", "??"],
                    "rows": [{"??": "A????ua????", "??": "230.50 V"}],
                    "view_label": "????",
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
        response = client.post("/api/chat/stream", json={"message": "a2_b1?2024?1?1??ua???", "history": []})

    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    complete_event = events[-1]
    assert complete_event["type"] == "complete"
    assert complete_event["table_preview"] is not None
    assert complete_event["table_preview"]["data"][0]["tag"] == "ua"
    assert complete_event["table_preview"]["focused_table"]["view_label"] == "????"
