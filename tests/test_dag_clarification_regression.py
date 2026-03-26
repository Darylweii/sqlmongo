from dataclasses import dataclass

from src.agent.dag_orchestrator import DAGOrchestrator


@dataclass
class _Device:
    device: str
    name: str
    project_id: str
    project_name: str
    project_code_name: str
    tg: str | None = None
    device_type: str = "meter"

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
    def search_devices(self, keyword):
        assert keyword == "a1_b9"
        return [
            _Device("a1_b9", "B2?", "1", "??????????????", "ceec"),
            _Device("a1_b9", "601-612", "2", "??????", "plyh"),
        ], False

    def list_projects(self):
        return []

    def list_all_devices(self):
        return []


class _FakeDataFetcher:
    def fetch_sync(self, **kwargs):
        raise AssertionError("clarification path should stop before database fetch")


class _FakeLLM:
    pass


def test_dag_clarification_stops_before_db_fetch() -> None:
    orchestrator = DAGOrchestrator(
        llm=_FakeLLM(),
        metadata_engine=_FakeMetadataEngine(),
        data_fetcher=_FakeDataFetcher(),
    )

    query = "\u67e5\u8be2 a1_b9 \u8bbe\u5907\u4eca\u5929\u7684\u7535\u538b\u6570\u636e"
    events = list(orchestrator.run_with_progress(query))

    assert [event["node_name"] for event in events if event["type"] == "step_done"] == [
        "intent_parser",
        "metadata_mapper",
    ]

    final_event = events[-1]
    assert final_event["type"] == "final_answer"
    assert final_event["clarification_required"] is True
    assert final_event["clarification_candidates"][0]["keyword"] == "a1_b9"
    assert len(final_event["clarification_candidates"][0]["candidates"]) == 2
