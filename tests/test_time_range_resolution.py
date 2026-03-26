from datetime import datetime

from src.agent.orchestrator import LLMAgent


def make_agent() -> LLMAgent:
    return LLMAgent(llm=None)


def test_resolve_today_range_from_query() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "查询 a1_b9 设备今天的电压数据",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result == {"start_time": "2026-03-17", "end_time": "2026-03-17"}


def test_normalize_action_input_overrides_wrong_today_range() -> None:
    agent = make_agent()
    state = agent._init_state("查询 a1_b9 设备今天的电压数据")
    normalized = agent._normalize_action_input(
        {
            "device_codes": ["a1_b9"],
            "data_type": "u_line",
            "start_time": "2026-03-10",
            "end_time": "2026-03-17",
        },
        "get_sensor_data",
        state,
    )
    resolved = agent._resolve_time_range_from_query(
        state["user_query"],
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert resolved == {"start_time": "2026-03-17", "end_time": "2026-03-17"}
    assert normalized["start_time"] != "2026-03-10"


def test_resolve_this_week_range() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "a1_b9 设备本周的用电情况",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result == {"start_time": "2026-03-16", "end_time": "2026-03-17"}


def test_resolve_month_range() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "a1_b9 设备2024年1月的用电量是上升还是下降？",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result == {"start_time": "2024-01-01", "end_time": "2024-01-31"}


def test_resolve_month_range_with_daily_average_phrase() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "计算 a1_b9 和 b1_b14 在2024年1月的日均用电量对比",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result == {"start_time": "2024-01-01", "end_time": "2024-01-31"}


def test_ambiguous_dual_period_keeps_llm_control() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "对比 a1_b9 设备上周和本周的用电量变化",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result is None


def test_resolve_recent_numeric_days_range() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "a1_b9 设备最近30天的用电情况",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result == {"start_time": "2026-02-15", "end_time": "2026-03-17"}


def test_resolve_iso_em_dash_range() -> None:
    agent = make_agent()
    result = agent._resolve_time_range_from_query(
        "查询 a1_b9 设备 2024-01-01—2024-01-31 的电压数据",
        now=datetime(2026, 3, 17, 15, 30, 0),
    )
    assert result == {"start_time": "2024-01-01", "end_time": "2024-01-31"}
