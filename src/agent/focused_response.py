from __future__ import annotations

from typing import Any, Mapping


def format_metric_value(value: Any, unit: str = "") -> str:
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
        return f"{number:.2f}{(' ' + unit) if unit else ''}"
    except (TypeError, ValueError):
        return str(value)


def build_focused_sensor_response(focused: Mapping[str, Any], total_count: int | None = None) -> str:
    if not isinstance(focused, Mapping):
        return ""

    mode = str(focused.get("mode") or "")
    unit = str(focused.get("unit") or "")
    metric = str(focused.get("metric") or "数值")

    if mode == "trend_decision":
        direction_label = str(focused.get("direction_label") or "稳定")
        lines: list[str] = []
        lines.append("【结果】")
        lines.append(str(focused.get("headline") or f"{metric}整体呈{direction_label}趋势"))
        lines.append("")
        lines.append("【判断依据】")
        lines.append(f"- 判断粒度: {focused.get('basis_label') or '按原始时序'}")
        lines.append(f"- 起始均值: {format_metric_value(focused.get('start_mean'), unit)} （{focused.get('start_label') or '-'}）")
        lines.append(f"- 结束均值: {format_metric_value(focused.get('end_mean'), unit)} （{focused.get('end_label') or '-'}）")
        change_rate = focused.get("change_rate")
        if change_rate is not None:
            lines.append(f"- 变化幅度: {float(change_rate):.2f}%")
        aggregation_note = str(focused.get("aggregation_note") or "").strip()
        if aggregation_note:
            lines.append(f"- 统计口径: {aggregation_note}")
        return "\n".join(lines).strip()

    if mode == "anomaly_points":
        lines: list[str] = []
        basis_label = str(focused.get("basis_label") or "按原始时序")
        anomaly_count = int(focused.get("anomaly_count") or 0)
        sample_count = int(focused.get("sample_count") or 0)
        anomaly_ratio_pct = float(focused.get("anomaly_ratio_pct") or 0.0)
        lower_bound = focused.get("lower_bound")
        upper_bound = focused.get("upper_bound")
        detection_note = str(focused.get("aggregation_note") or focused.get("detection_note") or "").strip()
        rows = list(focused.get("rows") or [])

        lines.append("【结果】")
        if focused.get("insufficient_samples"):
            lines.append(str(focused.get("headline") or f"{metric}样本不足，暂不适合识别异常时间点"))
            lines.append("")
            lines.append("【说明】")
            lines.append(f"- 判定粒度: {basis_label}")
            if sample_count:
                lines.append(f"- 样本数: {sample_count}")
            if detection_note:
                lines.append(f"- 统计口径: {detection_note}")
            return "\n".join(lines).strip()

        if anomaly_count <= 0:
            lines.append(str(focused.get("headline") or f"在{basis_label}口径下，未识别到明显异常{metric}时间点"))
            lines.append("")
            lines.append("【判断依据】")
            lines.append(f"- 判定粒度: {basis_label}")
            if sample_count:
                lines.append(f"- 样本数: {sample_count}")
            if lower_bound is not None and upper_bound is not None:
                lines.append(f"- 正常区间: {format_metric_value(lower_bound, unit)} ~ {format_metric_value(upper_bound, unit)}")
            if detection_note:
                lines.append(f"- 统计口径: {detection_note}")
            return "\n".join(lines).strip()

        lines.append(str(focused.get("headline") or f"在{basis_label}口径下识别到 {anomaly_count} 个异常{metric}时间点"))
        if rows:
            lines.append("")
            lines.append("【异常时间点】")
            for index, row in enumerate(rows, start=1):
                parts = [str(row.get("time") or "-")]
                if row.get("device"):
                    parts.append(str(row.get("device")))
                if row.get("tag"):
                    parts.append(str(row.get("tag")))
                detail = f"{index}. {' / '.join(parts)} / {format_metric_value(row.get('value'), unit)}"
                severity = row.get("severity")
                if severity is not None:
                    detail += f" / 偏离 {format_metric_value(severity, unit)}"
                lines.append(detail)

        lines.append("")
        lines.append("【判断依据】")
        lines.append(f"- 判定粒度: {basis_label}")
        if sample_count:
            lines.append(f"- 样本数: {sample_count}")
        lines.append(f"- 异常点数: {anomaly_count}")
        lines.append(f"- 异常占比: {anomaly_ratio_pct:.2f}%")
        if lower_bound is not None and upper_bound is not None:
            lines.append(f"- 正常区间: {format_metric_value(lower_bound, unit)} ~ {format_metric_value(upper_bound, unit)}")
        if detection_note:
            lines.append(f"- 统计口径: {detection_note}")
        return "\n".join(lines).strip()

    rows = list(focused.get("rows") or [])
    if not rows:
        return ""

    order = str(focused.get("order") or "")
    direction_text = "从高到低" if order == "desc" else "从低到高"

    lines: list[str] = []
    lines.append("【结果】")

    if mode == "bucket_summary":
        granularity = str(focused.get("granularity") or "day")
        granularity_label_map = {
            "hour": "按小时",
            "day": "按天",
            "week": "按周",
            "month": "按月",
        }
        granularity_label = granularity_label_map.get(granularity, "按周期")
        aggregation_note = str(focused.get("aggregation_note") or "").strip()
        lines.append(str(focused.get("headline") or f"已按{granularity_label}汇总{metric}"))
        lines.append("")
        lines.append("【分周期结果】")
        for index, row in enumerate(rows, start=1):
            lines.append(f"{index}. {row.get('time') or '-'} / {format_metric_value(row.get('value'), unit)} / 样本 {row.get('sample_count', 0)}")
        lines.append("")
        lines.append("【说明】")
        lines.append(f"- 统计粒度: {granularity_label}")
        if aggregation_note:
            lines.append(f"- 统计口径: {aggregation_note}")
        return "\n".join(lines).strip()

    if mode == "ranked_buckets":
        granularity = str(focused.get("granularity") or "day")
        granularity_label_map = {"hour": "\u6309\u5c0f\u65f6", "day": "\u6309\u5929", "week": "\u6309\u5468", "month": "\u6309\u6708"}
        granularity_label = granularity_label_map.get(granularity, "\u6309\u5468\u671f")
        aggregation_note = str(focused.get("aggregation_note") or "").strip()
        if len(rows) == 1:
            top_row = rows[0]
            extreme_text = "最高" if order == "desc" else "最低"
            lines.append(f"按{granularity_label}统计后，{metric}{extreme_text}的周期是 {top_row.get('time') or '-'}，数值为 {format_metric_value(top_row.get('value'), unit)}。")
            lines.append("")
            lines.append("【说明】")
            lines.append(f"- 统计粒度: {granularity_label}")
            lines.append(f"- 样本数: {top_row.get('sample_count', 0)}")
            if total_count is not None:
                lines.append(f"- 排序范围: 共 {total_count} 条原始记录")
            if aggregation_note:
                lines.append(f"- 统计口径: {aggregation_note}")
            return "\n".join(lines).strip()

        lines.append(f"已按{granularity_label}{metric}{direction_text}排序，以下是前 {len(rows)} 个周期：")
        lines.append("")
        for index, row in enumerate(rows, start=1):
            lines.append(f"{index}. {row.get('time') or '-'} / {format_metric_value(row.get('value'), unit)} / 样本 {row.get('sample_count', 0)}")
        lines.append("")
        lines.append("【说明】")
        if total_count is not None:
            lines.append(f"- 排序范围: 共 {total_count} 条原始记录")
        if aggregation_note:
            lines.append(f"- 统计口径: {aggregation_note}")
        return "\n".join(lines).strip()

    extreme_text = "最高" if order == "desc" else "最低"
    if len(rows) == 1:
        top_row = rows[0]
        parts = [str(top_row.get("time") or "-")]
        if top_row.get("device"):
            parts.append(str(top_row.get("device")))
        if top_row.get("tag"):
            parts.append(str(top_row.get("tag")))
        lines.append(f"{metric}{extreme_text}的时间点是 {' / '.join(parts)}，数值为 {format_metric_value(top_row.get('value'), unit)}。")
        lines.append("")
        lines.append("【说明】")
        if total_count is not None:
            lines.append(f"- 排序范围: 共 {total_count} 条记录")
        return "\n".join(lines).strip()

    lines.append(f"已按{metric}{direction_text}排序，以下是{extreme_text}的前 {len(rows)} 个时间点：")
    lines.append("")
    for index, row in enumerate(rows, start=1):
        parts = [str(row.get("time") or "-")]
        if row.get("device"):
            parts.append(str(row.get("device")))
        if row.get("tag"):
            parts.append(str(row.get("tag")))
        lines.append(f"{index}. {' / '.join(parts)} / {format_metric_value(row.get('value'), unit)}")
    lines.append("")
    lines.append("【说明】")
    if total_count is not None:
        lines.append(f"- 排序范围: 共 {total_count} 条记录")
    lines.append(f"- 第 1 名: {rows[0].get('time', '-')} / {format_metric_value(rows[0].get('value'), unit)}")
    return "\n".join(lines).strip()
