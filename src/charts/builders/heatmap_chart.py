from __future__ import annotations

from typing import Any, Dict, List

from src.charts.builders.common import build_slot_groups, display_name, group_records, hourly_average_records


def build_heatmap_chart(
    normalized_records: List[Dict[str, Any]],
    analysis: Dict[str, Any] | None,
    data_type: str,
    device_names: Dict[str, str] | None,
    planner_params: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if not normalized_records:
        return None

    planner_params = planner_params or {}
    if planner_params.get("series_dimension") == "slot":
        slot_groups = build_slot_groups(normalized_records, planner_params.get("comparison_slots"))
        heatmap_records: List[Dict[str, Any]] = []
        for label, records in slot_groups:
            for item in records:
                copied = dict(item)
                copied["slot"] = label
                heatmap_records.append(copied)
        normalized_records = heatmap_records

    unit = str((analysis or {}).get("unit") or "")
    grouped_by_device = group_records(normalized_records, "device")
    if planner_params.get("series_dimension") == "slot":
        group_key = "slot"
        label_builder = lambda key: key
    elif len(grouped_by_device) > 1 or str((analysis or {}).get("mode") or "") == "comparison":
        group_key = "device"
        label_builder = lambda key: display_name(key, device_names)
    else:
        grouped_by_tag = group_records(normalized_records, "tag")
        group_key = "tag" if len(grouped_by_tag) > 1 else "device"
        label_builder = lambda key: key if group_key == "tag" else display_name(key, device_names)

    hourly = hourly_average_records(normalized_records, group_key=group_key)
    if not hourly:
        return None
    x_axis = list(hourly.keys())
    y_groups = sorted({group_name for row in hourly.values() for group_name in row.keys()})
    y_axis = [label_builder(group_name) for group_name in y_groups]
    label_index = {group_name: idx for idx, group_name in enumerate(y_groups)}
    data = []
    max_value = 0.0
    for x_index, hour in enumerate(x_axis):
        for group_name, avg_value in hourly.get(hour, {}).items():
            max_value = max(max_value, float(avg_value))
            data.append([x_index, label_index[group_name], float(avg_value)])
    if not data:
        return None
    return {
        "tooltip": {"position": "top"},
        "grid": {"left": 72, "right": 24, "top": 32, "bottom": 48},
        "xAxis": {"type": "category", "data": x_axis},
        "yAxis": {"type": "category", "data": y_axis},
        "visualMap": {
            "min": 0,
            "max": round(max_value, 2) if max_value else 1,
            "calculable": True,
            "orient": "horizontal",
            "left": "center",
            "bottom": 0,
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {"show": False},
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}},
            }
        ],
    }
