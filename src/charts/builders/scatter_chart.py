from __future__ import annotations

from typing import Any, Dict, List

from src.charts.builders.common import build_slot_groups, display_name, downsample_points, group_records, sorted_group_items


def build_scatter_chart(
    normalized_records: List[Dict[str, Any]],
    analysis: Dict[str, Any] | None,
    data_type: str,
    device_names: Dict[str, str] | None,
    planner_params: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if not normalized_records:
        return None

    planner_params = planner_params or {}
    unit = str((analysis or {}).get("unit") or "")
    grouped_by_device = group_records(normalized_records, "device")
    is_comparison = len(grouped_by_device) > 1 or str((analysis or {}).get("mode") or "") == "comparison"

    if planner_params.get("series_dimension") == "slot":
        grouped_items = build_slot_groups(normalized_records, planner_params.get("comparison_slots"))
    else:
        group_key = "device" if is_comparison else ("tag" if len(group_records(normalized_records, "tag")) > 1 else "device")
        grouped = group_records(normalized_records, group_key)
        grouped_items = [
            (display_name(key, device_names) if group_key == "device" else key, records)
            for key, records in sorted_group_items(grouped)
        ]

    series = []
    for label, records in grouped_items:
        sampled = downsample_points(records, max_points=300)
        points = sampled if planner_params.get("aggregation") != "anomaly_only" else [item for item in sampled if bool(item.get("is_anomaly"))]
        if not points:
            continue
        series.append(
            {
                "name": label,
                "type": "scatter",
                "symbolSize": 8,
                "data": [[item["time_value"], item["value"]] for item in points],
            }
        )

    if planner_params.get("highlight_anomalies"):
        anomaly_data = [
            [item["time_value"], item["value"]]
            for item in normalized_records
            if bool(item.get("is_anomaly"))
        ]
        if anomaly_data:
            series.append(
                {
                    "name": "异常点",
                    "type": "scatter",
                    "symbolSize": 10,
                    "itemStyle": {"color": "#ef4444"},
                    "data": anomaly_data[:160],
                }
            )

    if not series:
        return None

    return {
        "tooltip": {"trigger": "item"},
        "legend": {"data": [item["name"] for item in series]},
        "grid": {"left": 40, "right": 24, "top": 48, "bottom": 48},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": unit},
        "series": series,
    }
