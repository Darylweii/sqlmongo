from src.agent.orchestrator import LLMAgent
from src.agent.query_entities import parse_query_entities


def make_agent() -> LLMAgent:
    return LLMAgent(llm=None)


def test_parse_generic_sensor_query_entities() -> None:
    parsed = parse_query_entities("\u53d8\u538b\u5668\u4eca\u5929\u7684\u7528\u7535\u91cf")
    assert parsed.has_sensor_intent is True
    assert parsed.has_time_reference is True
    assert parsed.inferred_data_type == "ep"
    assert parsed.search_targets == ("\u53d8\u538b\u5668",)


def test_parse_comparison_targets_without_hardcoded_aliases() -> None:
    parsed = parse_query_entities("\u5473\u591a\u7f8e\u5bf9\u6bd4\u706b\u9505\u5e97\u7684\u7528\u7535\u91cf")
    assert parsed.has_comparison_intent is True
    assert parsed.search_targets == ("\u5473\u591a\u7f8e", "\u706b\u9505\u5e97")
    assert parsed.inferred_data_type == "ep"


def test_parse_detect_data_type_query_keeps_core_target() -> None:
    parsed = parse_query_entities("\u53d8\u538b\u5668\u6709\u54ea\u4e9b\u6570\u636e\u7c7b\u578b")
    assert parsed.has_detect_data_types_intent is True
    assert parsed.search_targets == ("\u53d8\u538b\u5668",)


def test_parse_contextual_project_scope_without_hardcoding_project_name() -> None:
    parsed = parse_query_entities("\u667a\u6167\u7269\u8054\u7f51\u80fd\u6548\u5e73\u53f0\u91cc\u6709\u54ea\u4e9b\u53d8\u538b\u5668")
    assert parsed.search_targets == ("\u53d8\u538b\u5668",)
    assert parsed.project_hints == ("\u667a\u6167\u7269\u8054\u7f51\u80fd\u6548\u5e73\u53f0",)


def test_parse_multiple_explicit_device_codes() -> None:
    parsed = parse_query_entities("\u6bd4\u8f83 a9_b6 a1_b7\u3001a2_b14 \u548c a1_b9 \u6700\u8fd1\u4e00\u5468\u7684\u7528\u7535\u60c5\u51b5")
    assert parsed.has_comparison_intent is True
    assert parsed.explicit_device_codes == ("a9_b6", "a1_b7", "a2_b14", "a1_b9")
    assert parsed.search_targets == parsed.explicit_device_codes


def test_parse_single_explicit_device_code_keeps_stable_target() -> None:
    parsed = parse_query_entities("a1_b9 \u8bbe\u59072024\u5e741\u6708\u7684\u7528\u7535\u91cf\u662f\u4e0a\u5347\u8fd8\u662f\u4e0b\u964d\uff1f")
    assert parsed.explicit_device_codes == ("a1_b9",)
    assert parsed.search_targets == ("a1_b9",)


def test_parse_ranked_timepoints_semantics_for_top_query() -> None:
    parsed = parse_query_entities("找出 a1_b9 设备2024年1月用电量最高的前5个时间点")
    assert parsed.query_mode == "ranked_timepoints"
    assert parsed.has_ranked_point_intent is True
    assert parsed.ranking_order == "desc"
    assert parsed.ranking_limit == 5


def test_parse_ranked_timepoints_semantics_for_record_paraphrase() -> None:
    parsed = parse_query_entities("列出 a1_b9 2024年1月电量最大的5条记录")
    assert parsed.query_mode == "ranked_timepoints"
    assert parsed.has_ranked_point_intent is True
    assert parsed.ranking_order == "desc"
    assert parsed.ranking_limit == 5


def test_parse_ranked_timepoints_semantics_for_lowest_query() -> None:
    parsed = parse_query_entities("a1_b9 在2024年1月用电量最低的前3个时刻是什么")
    assert parsed.query_mode == "ranked_timepoints"
    assert parsed.has_ranked_point_intent is True
    assert parsed.ranking_order == "asc"
    assert parsed.ranking_limit == 3


def test_parse_ranked_buckets_semantics_for_daily_top_query() -> None:
    parsed = parse_query_entities("找出 a1_b9 设备2024年1月按天用电量最高的前5个周期")
    assert parsed.query_mode == "ranked_buckets"
    assert parsed.has_ranked_point_intent is True
    assert parsed.ranking_order == "desc"
    assert parsed.ranking_limit == 5
    assert parsed.ranking_granularity == "day"


def test_parse_ranked_buckets_semantics_for_hourly_top_query() -> None:
    parsed = parse_query_entities("列出 a1_b9 今天按小时电压最高的前3个周期")
    assert parsed.query_mode == "ranked_buckets"
    assert parsed.has_ranked_point_intent is True
    assert parsed.ranking_order == "desc"
    assert parsed.ranking_limit == 3
    assert parsed.ranking_granularity == "hour"


def test_parse_trend_decision_semantics_for_rise_or_fall_query() -> None:
    parsed = parse_query_entities("a1_b9 设备2024年1月的用电量是上升还是下降？")
    assert parsed.query_mode == "trend_decision"
    assert parsed.has_trend_decision_intent is True
    assert parsed.explicit_device_codes == ("a1_b9",)


def test_parse_trend_decision_semantics_for_trend_query() -> None:
    parsed = parse_query_entities("a1_b9 最近一周用电量趋势如何")
    assert parsed.query_mode == "trend_decision"
    assert parsed.has_trend_decision_intent is True


def test_build_heuristic_action_fast_paths_generic_sensor_search() -> None:
    agent = make_agent()
    state = agent._init_state("\u53d8\u538b\u5668\u4eca\u5929\u7684\u7528\u7535\u91cf")
    action = agent._build_heuristic_action(state, fast_path=True)
    assert action is not None
    assert action["action"] == "search_devices"
    assert action["action_input"]["keywords"] == ["\u53d8\u538b\u5668"]
    assert action["action_input"]["comparison_mode"] is False
    assert action["_heuristic_reason"] == "fast_path_generic_sensor_search"


def test_build_heuristic_action_keeps_comparison_targets_generic() -> None:
    agent = make_agent()
    state = agent._init_state("\u5473\u591a\u7f8e\u5bf9\u6bd4\u706b\u9505\u5e97\u7684\u7528\u7535\u91cf")
    action = agent._build_heuristic_action(state, fast_path=True)
    assert action is not None
    assert action["action"] == "search_devices"
    assert action["action_input"]["keywords"] == ["\u5473\u591a\u7f8e", "\u706b\u9505\u5e97"]
    assert action["action_input"]["comparison_mode"] is True


def test_rerank_devices_with_project_context_prefers_project_match() -> None:
    agent = make_agent()
    devices = [
        {
            "device": "a10_b1",
            "name": "1#\u53d8\u538b\u5668\u96c6\u4e2d\u5668 \u5473\u591a\u7f8e",
            "project_id": "p1",
            "project_name": "\u667a\u6167\u7269\u8054\u7f51\u80fd\u6548\u5e73\u53f0",
            "match_score": 88.0,
            "matched_fields": ["name"],
        },
        {
            "device": "x1_y1",
            "name": "1#\u53d8\u538b\u5668",
            "project_id": "p2",
            "project_name": "\u5176\u4ed6\u9879\u76ee",
            "match_score": 88.0,
            "matched_fields": ["name"],
        },
    ]
    reranked = agent._rerank_devices_with_query_context(
        devices,
        "\u53d8\u538b\u5668",
        "\u667a\u6167\u7269\u8054\u7f51\u80fd\u6548\u5e73\u53f0\u91cc\u6709\u54ea\u4e9b\u53d8\u538b\u5668",
    )
    assert reranked[0]["device"] == "a10_b1"
    assert float(reranked[0].get("match_score") or 0.0) > float(reranked[1].get("match_score") or 0.0)


def test_parse_anomaly_points_semantics_for_anomaly_query() -> None:
    parsed = parse_query_entities("a1_b9 \u8bbe\u59072024\u5e741\u6708\u6709\u6ca1\u6709\u5f02\u5e38\u7528\u7535\u7684\u65f6\u95f4\u70b9\uff1f")
    assert parsed.query_mode == "anomaly_points"
    assert parsed.has_anomaly_point_intent is True
    assert parsed.inferred_data_type == "ep"
    assert parsed.explicit_device_codes == ("a1_b9",)
    assert parsed.search_targets == ("a1_b9",)


def test_parse_device_listing_query_with_intermediate_phrase() -> None:
    parsed = parse_query_entities("\u641c\u7d22\u5305\u542b b9 \u7684\u8bbe\u5907")
    assert parsed.has_device_listing_intent is True
    assert parsed.has_sensor_intent is False
    assert parsed.query_mode == "device_listing"
    assert parsed.search_targets == ("b9",)



def test_parse_ranked_buckets_semantics_for_natural_most_query() -> None:
    parsed = parse_query_entities("a1_b9 \u8bbe\u59072024\u5e741\u6708\u54ea\u5929\u7528\u7535\u6700\u591a\uff1f")
    assert parsed.query_mode == "ranked_buckets"
    assert parsed.has_ranked_point_intent is True
    assert parsed.ranking_order == "desc"
    assert parsed.ranking_limit == 1
    assert parsed.ranking_granularity == "day"


def test_parse_device_listing_query_for_elevator_devices() -> None:
    parsed = parse_query_entities("\u6709\u54ea\u4e9b\u7535\u68af\u8bbe\u5907\uff1f")
    assert parsed.has_device_listing_intent is True
    assert parsed.query_mode == "device_listing"
    assert parsed.search_targets == ("\u7535\u68af",)


def test_parse_project_scoped_device_listing_query() -> None:
    parsed = parse_query_entities("\u5317\u4eac\u7535\u529b\u9879\u76ee\u6709\u54ea\u4e9b\u8bbe\u5907\uff1f")
    assert parsed.has_device_listing_intent is True
    assert parsed.has_project_listing_intent is False
    assert parsed.query_mode == "device_listing"
    assert any("\u5317\u4eac\u7535\u529b\u9879\u76ee" == hint for hint in parsed.project_hints)


def test_parse_exact_phase_voltage_query_prefers_local_metric_tag() -> None:
    parsed = parse_query_entities("a2_b1?2024?1?1??ua???")
    assert parsed.has_sensor_intent is True
    assert parsed.inferred_data_type == "ua"
    assert parsed.explicit_device_codes == ("a2_b1",)
    assert parsed.search_targets == ("a2_b1",)


def test_parse_multi_phase_voltage_query_keeps_voltage_family() -> None:
    parsed = parse_query_entities("?? a2_b1 ?2024?1?1??ua?ub")
    assert parsed.has_sensor_intent is True
    assert parsed.inferred_data_type == "u_line"
    assert parsed.explicit_device_codes == ("a2_b1",)
