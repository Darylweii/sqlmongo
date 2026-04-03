from __future__ import annotations

from typing import Any, Dict, List

from src.charts.builders.common import build_slot_groups, display_name, downsample_points, group_records, sorted_group_items


def build_line_chart(
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
    metric = str((analysis or {}).get("metric") or data_type or "数据")
    grouped_by_device = group_records(normalized_records, "device")
    is_comparison = len(grouped_by_device) > 1 or str((analysis or {}).get("mode") or "") == "comparison"

    if is_comparison:
        slot_groups = build_slot_groups(normalized_records, planner_params.get("comparison_slots")) if planner_params.get("series_dimension") == "slot" else []
        grouped_items = slot_groups or [
            (display_name(device, device_names), records)
            for device, records in sorted_group_items(grouped_by_device)
        ]
        series = []
        legends = []
        for label, records in grouped_items:
            legends.append(label)
            series.append(
                {
                    "name": label,
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "data": [[item["time_value"], item["value"]] for item in downsample_points(records)],
                }
            )
        if planner_params.get("highlight_anomalies"):
            anomaly_data = [
                [item["time_value"], item["value"]]
                for item in normalized_records
                if bool(item.get("is_anomaly"))
            ]
            if anomaly_data:
                legends.append("异常点")
                series.append(
                    {
                        "name": "异常点",
                        "type": "scatter",
                        "symbolSize": 10,
                        "itemStyle": {"color": "#ef4444"},
                        "data": anomaly_data[:120],
                    }
                )
        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": legends},
            "grid": {"left": 40, "right": 24, "top": 48, "bottom": 48},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": unit},
            "series": series,
        }

    grouped_by_tag = group_records(normalized_records, "tag")
    if len(grouped_by_tag) <= 1:
        only_records = downsample_points(normalized_records)
        series = [
            {
                "name": metric,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": [[item["time_value"], item["value"]] for item in only_records],
                "markPoint": {"data": [{"type": "max", "name": "最大值"}, {"type": "min", "name": "最小值"}]},
                "markLine": {"data": [{"type": "average", "name": "平均值"}]},
            }
        ]
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
                        "data": anomaly_data[:120],
                    }
                )
        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": [item["name"] for item in series]},
            "grid": {"left": 40, "right": 24, "top": 48, "bottom": 48},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": unit},
            "series": series,
        }

    series = []
    legends = []
    for tag, records in sorted_group_items(grouped_by_tag):
        tag_label = tag or metric
        legends.append(tag_label)
        series.append(
            {
                "name": tag_label,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": [[item["time_value"], item["value"]] for item in downsample_points(records)],
            }
        )
    return {
        "tooltip": {"trigger": "axis"},
        "legend": {"data": legends},
        "grid": {"left": 40, "right": 24, "top": 48, "bottom": 48},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": unit},
        "series": series,
    }
