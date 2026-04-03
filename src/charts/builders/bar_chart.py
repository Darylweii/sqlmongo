from __future__ import annotations

from typing import Any, Dict, List

from src.charts.builders.common import build_slot_groups, build_slot_label, display_name, group_records, hourly_average_records


def _summary_values(records: List[Dict[str, Any]]) -> Dict[str, float]:
    values = [float(item["value"]) for item in records if item.get("value") is not None]
    if not values:
        return {"avg": 0.0, "sum": 0.0, "max": 0.0, "min": 0.0}
    return {
        "avg": round(sum(values) / len(values), 2),
        "sum": round(sum(values), 2),
        "max": round(max(values), 2),
        "min": round(min(values), 2),
    }


def build_bar_chart(
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
    aggregation = str(planner_params.get("aggregation") or "")

    if is_comparison or aggregation == "device_summary":
        slot_groups = build_slot_groups(normalized_records, planner_params.get("comparison_slots")) if planner_params.get("series_dimension") == "slot" else []
        if slot_groups:
            grouped_items = slot_groups
            labels = [label for label, _ in grouped_items]
        else:
            grouped_items = list(grouped_by_device.items())
            labels = [display_name(device, device_names) for device, _ in grouped_items]

        summaries = [_summary_values(records) for _, records in grouped_items]
        use_total = str(data_type or "").strip().lower() in {"ep", "epzyz", "fz-ep", "gffddl", "dcdcdljl", "dcdfdljl"}
        legend = ["平均值", "总量"] if use_total else ["平均值", "最大值", "最小值"]
        series = [
            {
                "name": "平均值",
                "type": "bar",
                "data": [item["avg"] for item in summaries],
                "itemStyle": {"color": "#6366f1"},
            }
        ]
        if use_total:
            series.append(
                {
                    "name": "总量",
                    "type": "bar",
                    "data": [item["sum"] for item in summaries],
                    "itemStyle": {"color": "#10b981"},
                }
            )
        else:
            series.extend(
                [
                    {
                        "name": "最大值",
                        "type": "bar",
                        "data": [item["max"] for item in summaries],
                        "itemStyle": {"color": "#f59e0b"},
                    },
                    {
                        "name": "最小值",
                        "type": "bar",
                        "data": [item["min"] for item in summaries],
                        "itemStyle": {"color": "#14b8a6"},
                    },
                ]
            )
        return {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"data": legend},
            "grid": {"left": 48, "right": 24, "top": 48, "bottom": 36},
            "xAxis": {"type": "category", "data": labels},
            "yAxis": {"type": "value", "name": unit},
            "series": series,
        }

    group_key = "tag" if len(group_records(normalized_records, "tag")) > 1 else "device"
    hourly = hourly_average_records(normalized_records, group_key=group_key)
    if not hourly:
        return None
    x_axis = list(hourly.keys())
    group_names = sorted({group_name for values in hourly.values() for group_name in values.keys()})
    series = []
    for group_name in group_names:
        series.append(
            {
                "name": group_name if group_key == "tag" else display_name(group_name, device_names),
                "type": "bar",
                "data": [hourly.get(hour, {}).get(group_name, 0) for hour in x_axis],
            }
        )

    return {
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
        "legend": {"data": [item["name"] for item in series]},
        "grid": {"left": 40, "right": 24, "top": 32, "bottom": 36},
        "xAxis": {"type": "category", "data": x_axis},
        "yAxis": {"type": "value", "name": unit},
        "series": series,
    }
