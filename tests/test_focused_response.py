from src.agent.focused_response import build_focused_sensor_response


def test_build_focused_sensor_response_answers_bucket_top1_directly() -> None:
    response = build_focused_sensor_response(
        {
            "mode": "ranked_buckets",
            "metric": "???",
            "unit": "kWh",
            "granularity": "day",
            "order": "desc",
            "aggregation_note": "???????????????????",
            "rows": [
                {"time": "2024-01-31", "value": 1024.0, "sample_count": 24},
            ],
        },
        total_count=744,
    )

    assert "2024-01-31" in response
    assert "?????????" in response
    assert "? 1 ???" not in response
