from src.analysis.insight_engine import InsightEngine
from src.charts import chart_planner


def _single_phase_records():
    return [
        {"logTime": "2026-04-02 00:00:00", "val": 3.2, "device": "a1_b9", "tag": "ia"},
        {"logTime": "2026-04-02 00:00:00", "val": 4.1, "device": "a1_b9", "tag": "ib"},
        {"logTime": "2026-04-02 00:00:00", "val": 5.0, "device": "a1_b9", "tag": "ic"},
        {"logTime": "2026-04-02 01:00:00", "val": 6.8, "device": "a1_b9", "tag": "ia"},
        {"logTime": "2026-04-02 01:00:00", "val": 7.2, "device": "a1_b9", "tag": "ib"},
        {"logTime": "2026-04-02 01:00:00", "val": 8.5, "device": "a1_b9", "tag": "ic"},
    ]


def _comparison_records():
    return [
        {"logTime": "2026-04-02 00:00:00", "val": 3.2, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2026-04-02 01:00:00", "val": 6.8, "device": "a1_b9", "tag": "ep"},
        {"logTime": "2026-04-02 00:00:00", "val": 2.5, "device": "b1_b7", "tag": "ep"},
        {"logTime": "2026-04-02 01:00:00", "val": 4.4, "device": "b1_b7", "tag": "ep"},
    ]


def test_insight_engine_defaults_to_single_best_line_chart() -> None:
    analysis, chart_specs = InsightEngine.build(
        records=_single_phase_records(),
        statistics=None,
        data_type="i",
        device_codes=["a1_b9"],
        user_query="查询 a1_b9 今天的电流数据",
    )

    assert analysis is not None
    assert len(chart_specs) == 1
    assert chart_specs[0]["chart_type"] == "line"
    assert chart_specs[0]["planner_source"] == "rule"


def test_insight_engine_honors_explicit_bar_chart_request() -> None:
    _analysis, chart_specs = InsightEngine.build(
        records=_single_phase_records(),
        statistics=None,
        data_type="i",
        device_codes=["a1_b9"],
        user_query="查询 a1_b9 今天的电流数据，帮我画柱状图",
    )

    assert len(chart_specs) == 1
    assert chart_specs[0]["chart_type"] == "bar"


def test_insight_engine_honors_explicit_scatter_chart_request() -> None:
    _analysis, chart_specs = InsightEngine.build(
        records=_single_phase_records(),
        statistics=None,
        data_type="i",
        device_codes=["a1_b9"],
        user_query="查询 a1_b9 今天的电流数据，帮我画散点图",
    )

    assert len(chart_specs) == 1
    assert chart_specs[0]["chart_type"] == "scatter"


def test_insight_engine_honors_explicit_boxplot_request() -> None:
    _analysis, chart_specs = InsightEngine.build(
        records=_comparison_records(),
        statistics=None,
        data_type="ep",
        device_codes=["a1_b9", "b1_b7"],
        user_query="对比 a1_b9 和 b1_b7 的耗电量，帮我画箱线图",
    )

    assert len(chart_specs) == 1
    assert chart_specs[0]["chart_type"] == "boxplot"


def test_insight_engine_honors_explicit_heatmap_request() -> None:
    _analysis, chart_specs = InsightEngine.build(
        records=_single_phase_records(),
        statistics=None,
        data_type="i",
        device_codes=["a1_b9"],
        user_query="查询 a1_b9 今天的电流数据，帮我画热力图",
    )

    assert len(chart_specs) == 1
    assert chart_specs[0]["chart_type"] == "heatmap"


def test_extract_chart_request_supports_compound_chart_phrases() -> None:
    assert chart_planner.extract_chart_request("帮我画柱状对比图")["requested_chart_type"] == "bar"
    assert chart_planner.extract_chart_request("帮我画趋势对比图")["requested_chart_type"] == "line"
    assert chart_planner.extract_chart_request("帮我画箱线分布图")["requested_chart_type"] == "boxplot"
    assert chart_planner.extract_chart_request("帮我画异常散点图")["requested_chart_type"] == "scatter"


def test_insight_engine_uses_llm_fallback_when_rule_is_ambiguous(monkeypatch) -> None:
    monkeypatch.setattr(
        chart_planner,
        "_invoke_llm_chart_planner",
        lambda request, analysis: {"chart_type": "bar", "reason": "LLM 认为柱状图更适合", "confidence": 0.92},
    )

    _analysis, chart_specs = InsightEngine.build(
        records=_comparison_records(),
        statistics=None,
        data_type="ep",
        device_codes=["a1_b9", "b1_b7"],
        user_query="对比 a1_b9 和 b1_b7 的耗电量，帮我画图",
    )

    assert len(chart_specs) == 1
    assert chart_specs[0]["chart_type"] == "bar"
    assert chart_specs[0]["planner_source"] == "llm"
