import importlib
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient


class _FakeDevice:
    def __init__(self, device, name, project_id, project_name, project_code_name, tg=None):
        self.device = device
        self.name = name
        self.project_id = project_id
        self.project_name = project_name
        self.project_code_name = project_code_name
        self.tg = tg
        self.device_type = "meter"

    def to_dict(self):
        return {
            "device": self.device,
            "name": self.name,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "project_code_name": self.project_code_name,
            "tg": self.tg,
            "device_type": self.device_type,
        }


class _FakeMetadataEngine:
    def __init__(self):
        self.search_calls = 0

    def search_devices(self, keyword):
        self.search_calls += 1
        if keyword == "a1_b9":
            return [
                _FakeDevice("a1_b9", "B2?", "p1", "??????????????", "ceec"),
                _FakeDevice("a1_b9", "601-612", "p2", "??????", "plyh"),
            ], False
        return [], False

    def list_projects(self):
        return []

    def list_all_devices(self):
        return []


class _FakeDataFetcher:
    def fetch_sync(self, **kwargs):
        return SimpleNamespace(
            data=[{"logTime": "2026-03-23 00:00:00", "device": "a1_b9", "tag": "i", "val": 3.2}],
            total_count=1,
            statistics={"count": 1, "avg": 3.2, "max": 3.2, "min": 3.2},
            query_info={"collections": ["source_data_i_202603"]},
            is_sampled=False,
            failed_collections=[],
        )


def _parse_events(response_text: str):
    events = []
    for line in str(response_text or "").splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


def test_alias_confirmation_does_not_loop_in_dag(monkeypatch):
    app_module = importlib.import_module("web.app")
    app_module = importlib.reload(app_module)
    app_module.SESSION_STATE.clear()

    fake_metadata = _FakeMetadataEngine()
    monkeypatch.setattr(app_module, "metadata_engine", fake_metadata)
    monkeypatch.setattr(app_module, "data_fetcher", _FakeDataFetcher())
    monkeypatch.setattr(app_module, "llm", None)
    monkeypatch.setattr(app_module.config.agent, "orchestrator_type", "dag", raising=False)

    query = "\u67e5\u8be2 a1_b9 \u8bbe\u5907\u4eca\u5929\u7684\u7535\u6d41\u6570\u636e"
    with TestClient(app_module.app) as client:
        first = client.post("/api/chat/stream", json={"message": query, "history": [], "session_id": "loop-reg"})
        first_events = _parse_events(first.text)
        first_complete = first_events[-1]
        candidate = first_complete["clarification_candidates"][0]["candidates"][0]

        second = client.post(
            "/api/chat/stream",
            json={
                "message": "\u9009\u8fd9\u4e2a\u8bbe\u5907",
                "history": [],
                "session_id": "loop-reg",
                "alias_confirmation": {
                    "alias": "a1_b9",
                    "keyword": "a1_b9",
                    "device": candidate["device"],
                    "name": candidate["name"],
                    "project_id": candidate["project_id"],
                    "project_name": candidate["project_name"],
                    "project_code_name": candidate["project_code_name"],
                    "tg": candidate.get("tg"),
                    "device_info": candidate,
                    "original_question": query,
                },
            },
        )
        second_events = _parse_events(second.text)
        second_complete = second_events[-1]

    assert first_complete["clarification_required"] is True
    assert second_complete["clarification_required"] is False
    assert second_complete["query_params"]["device_codes"] == ["a1_b9"]
    assert second_complete["resolved_scope"]["items"][0]["project_name"] == candidate["project_name"]
    assert fake_metadata.search_calls == 1
