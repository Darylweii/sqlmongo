from __future__ import annotations

from typing import Any, Dict, List

from src.charts.builders import (
    build_bar_chart,
    build_boxplot_chart,
    build_heatmap_chart,
    build_line_chart,
    build_scatter_chart,
)
from src.charts.chart_types import CHART_TYPE_LABELS, DEFAULT_CHART_TYPE


CHART_BUILDERS = {
    "line": build_line_chart,
    "bar": build_bar_chart,
    "scatter": build_scatter_chart,
    "boxplot": build_boxplot_chart,
    "heatmap": build_heatmap_chart,
}


def build_chart_specs_from_plan(
    normalized_records: List[Dict[str, Any]],
    analysis: Dict[str, Any] | None,
    data_type: str,
    device_names: Dict[str, str] | None,
    chart_plans: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for plan in chart_plans or []:
        chart_type = str(plan.get("chart_type") or DEFAULT_CHART_TYPE).strip().lower()
        builder = CHART_BUILDERS.get(chart_type)
        if builder is None:
            continue
        option = builder(
            normalized_records=normalized_records,
            analysis=analysis,
            data_type=data_type,
            device_names=device_names,
            planner_params=plan.get("params") if isinstance(plan.get("params"), dict) else {},
        )
        if not option:
            continue
        spec = {
            "id": str(plan.get("id") or f"{chart_type}-chart"),
            "chart_type": chart_type,
            "title": str(plan.get("title") or CHART_TYPE_LABELS.get(chart_type, "图表")),
            "height": int(plan.get("height") or 360),
            "reason": str(plan.get("reason") or ""),
            "planner_source": str(plan.get("planner_source") or "rule"),
            "option": option,
        }
        specs.append(spec)
    return specs
