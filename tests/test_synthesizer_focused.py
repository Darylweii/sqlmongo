import json

from src.agent.nodes.synthesizer import SynthesizerNode


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return type("Resp", (), {"content": json.dumps({"ok": True}, ensure_ascii=False)})()


def test_synthesizer_prefers_focused_anomaly_response_without_llm() -> None:
    llm = FakeLLM()
    node = SynthesizerNode(llm)

    state = {
        "user_query": "a1_b9 的异常时间点有哪些",
        "history": [],
        "total_count": 6,
        "raw_data": [
            {"logTime": "2024-01-01 00:00:00", "val": 229, "device": "a1_b9", "tag": "ua"},
            {"logTime": "2024-01-01 01:00:00", "val": 230, "device": "a1_b9", "tag": "ua"},
            {"logTime": "2024-01-01 02:00:00", "val": 231, "device": "a1_b9", "tag": "ua"},
            {"logTime": "2024-01-01 03:00:00", "val": 260, "device": "a1_b9", "tag": "ua"},
            {"logTime": "2024-01-01 04:00:00", "val": 230, "device": "a1_b9", "tag": "ua"},
            {"logTime": "2024-01-01 05:00:00", "val": 229, "device": "a1_b9", "tag": "ua"},
        ],
        "query_plan": {
            "current_question": "a1_b9 的异常时间点有哪些",
            "query_mode": "anomaly_points",
            "inferred_data_type": "u_line",
            "search_targets": ["a1_b9"],
            "has_sensor_intent": True,
            "has_anomaly_point_intent": True,
            "response_style": "direct_answer",
            "time_start": "2024-01-01",
            "time_end": "2024-01-01",
        },
    }

    result = node(state)

    assert llm.calls == 0
    assert result["show_table"] is True
    assert "异常" in result["final_response"]
    assert "2024-01-01 03:00" in result["final_response"]
