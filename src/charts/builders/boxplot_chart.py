from __future__ import annotations

from typing import Any, Dict, List

from src.charts.builders.common import build_slot_groups, compute_boxplot_values, display_name, group_records, sorted_group_items


def build_boxplot_chart(
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

    if planner_params.get("series_dimension") == "slot":
        grouped_items = build_slot_groups(normalized_records, planner_params.get("comparison_slots"))
    elif len(grouped_by_device) > 1 or str((analysis or {}).get("mode") or "") == "comparison":
        grouped_items = [
            (display_name(device, device_names), records)
            for device, records in sorted_group_items(grouped_by_device)
        ]
    else:
        grouped_by_tag = group_records(normalized_records, "tag")
        if len(grouped_by_tag) > 1:
            grouped_items = [(tag, records) for tag, records in sorted_group_items(grouped_by_tag)]
        else:
            grouped_items = [
                (display_name(device, device_names), records)
                for device, records in sorted_group_items(grouped_by_device)
            ]

    labels = []
    data = []
    for label, records in grouped_items:
        stats = compute_boxplot_values([float(item["value"]) for item in records if item.get("value") is not None])
        if not stats:
            continue
        labels.append(label)
        data.append(stats)
    if not data:
        return None
    return {
        "tooltip": {"trigger": "item"},
        "grid": {"left": 48, "right": 24, "top": 32, "bottom": 48},
        "xAxis": {"type": "category", "data": labels},
        "yAxis": {"type": "value", "name": unit},
        "series": [{"type": "boxplot", "data": data}],
    }
