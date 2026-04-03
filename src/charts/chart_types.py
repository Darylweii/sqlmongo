from __future__ import annotations

from typing import Dict


SUPPORTED_CHART_TYPES = ("line", "bar", "scatter", "boxplot", "heatmap")
DEFAULT_CHART_TYPE = "line"

CHART_TYPE_LABELS: Dict[str, str] = {
    "line": "折线图",
    "bar": "柱状图",
    "scatter": "散点图",
    "boxplot": "箱线图",
    "heatmap": "热力图",
}

CHART_TYPE_ALIASES: Dict[str, str] = {
    "折线图": "line",
    "趋势图": "line",
    "曲线图": "line",
    "line": "line",
    "柱状图": "bar",
    "柱形图": "bar",
    "bar": "bar",
    "散点图": "scatter",
    "scatter": "scatter",
    "箱线图": "boxplot",
    "箱型图": "boxplot",
    "boxplot": "boxplot",
    "热力图": "heatmap",
    "heatmap": "heatmap",
}
