from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.charts.chart_planner import plan_chart_specs
from src.charts.chart_registry import build_chart_specs_from_plan
from src.charts.chart_types import CHART_TYPE_LABELS


DATA_TYPE_NAMES = {
    "ep": "用电量",
    "i": "电流",
    "u_line": "电压",
    "p": "功率",
    "qf": "功率因数",
    "t": "温度",
    "sd": "湿度",
    "f": "频率",
    "soc": "荷电状态",
    "gffddl": "供方峰段电量",
    "loadrate": "负载率",
}

DATA_TYPE_UNITS = {
    "ep": "kWh",
    "i": "A",
    "ia": "A",
    "ib": "A",
    "ic": "A",
    "u_line": "V",
    "ua": "V",
    "ub": "V",
    "uc": "V",
    "u_phase": "V",
    "uab": "V",
    "ubc": "V",
    "uca": "V",
    "p": "kW",
    "qf": "",
    "t": "°C",
    "sd": "%",
    "f": "Hz",
    "soc": "%",
    "gffddl": "kWh",
    "loadrate": "%",
}

ANOMALY_HINTS = ("异常", "离群", "波动点", "散点")
DISTRIBUTION_HINTS = ("稳定", "波动", "分布", "箱线")


class InsightEngine:
    @classmethod
    def build(
        cls,
        records: Optional[List[Dict[str, Any]]],
        statistics: Optional[Dict[str, Any]],
        data_type: str = "ep",
        device_codes: Optional[List[str]] = None,
        device_names: Optional[Dict[str, str]] = None,
        user_query: str = "",
        comparison_slots: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized = cls._normalize_records(records or [])
        if not normalized:
            return None, []

        analysis = cls.build_analysis(
            normalized_records=normalized,
            statistics=statistics or cls._compute_stats(normalized),
            data_type=data_type,
            device_codes=device_codes,
            device_names=device_names,
            user_query=user_query,
            comparison_slots=comparison_slots,
        )
        chart_specs = cls.build_chart_specs(
            normalized_records=normalized,
            analysis=analysis,
            data_type=data_type,
            device_names=device_names,
            user_query=user_query,
            comparison_slots=comparison_slots,
        )
        return analysis, chart_specs

    @classmethod
    def build_analysis(
        cls,
        *,
        normalized_records: List[Dict[str, Any]],
        statistics: Optional[Dict[str, Any]],
        data_type: str = "ep",
        device_codes: Optional[List[str]] = None,
        device_names: Optional[Dict[str, str]] = None,
        user_query: str = "",
        comparison_slots: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        normalized = list(normalized_records or [])
        if not normalized:
            return None

        unique_devices = sorted({item["device"] for item in normalized if item.get("device")})
        devices = [str(code).strip() for code in (device_codes or unique_devices) if str(code).strip()]
        has_multi_slot = len(comparison_slots or []) > 1
        if has_multi_slot or len(set(devices or unique_devices)) > 1 or len(unique_devices) > 1:
            return cls._build_comparison_analysis(normalized, data_type, device_names, comparison_slots)
        return cls._build_single_analysis(normalized, statistics or cls._compute_stats(normalized), data_type)

    @classmethod
    def build_chart_specs(
        cls,
        *,
        normalized_records: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        data_type: str = "ep",
        device_names: Optional[Dict[str, str]] = None,
        user_query: str = "",
        comparison_slots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        if not normalized_records or not analysis:
            return []
        chart_plans = plan_chart_specs(
            normalized_records=normalized_records,
            analysis=analysis,
            data_type=data_type,
            device_names=device_names,
            user_query=user_query,
            comparison_slots=comparison_slots,
        )
        return build_chart_specs_from_plan(
            normalized_records=normalized_records,
            analysis=analysis,
            data_type=data_type,
            device_names=device_names,
            chart_plans=chart_plans,
        )

    @classmethod
    def build_chart_context(
        cls,
        *,
        normalized_records: List[Dict[str, Any]],
        analysis: Optional[Dict[str, Any]],
        chart_specs: Optional[List[Dict[str, Any]]],
        user_query: str = "",
        comparison_slots: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not normalized_records or not analysis:
            return None

        available_chart_types = ["line", "bar", "boxplot", "scatter", "heatmap"]
        query_kind = cls._infer_query_kind(analysis, user_query, comparison_slots)
        recommended_chart_types = cls._build_recommended_chart_types(query_kind)
        anomaly_count = len(analysis.get("anomalies") or [])
        total_count = max(len(normalized_records), 1)
        recommended_chart_type = str((chart_specs or [{}])[0].get("chart_type") or "").strip().lower()
        if recommended_chart_type not in available_chart_types:
            recommended_chart_type = recommended_chart_types[0]

        return {
            "chartable": True,
            "query_kind": query_kind,
            "data_signature": cls._build_data_signature(normalized_records),
            "supports_follow_up_chart": True,
            "recommended_chart_type": recommended_chart_type,
            "recommended_chart_types": recommended_chart_types,
            "available_chart_types": available_chart_types,
            "comparison_slot_count": len(comparison_slots or []),
            "comparison_slots": list(comparison_slots or []),
            "anomaly_summary": {
                "has_anomalies": anomaly_count > 0,
                "anomaly_count": anomaly_count,
                "anomaly_ratio": round(anomaly_count / total_count * 100, 2),
            },
            "follow_up_suggestions": cls._build_follow_up_suggestions(query_kind),
        }

    @classmethod
    def build_comparison_slots(
        cls,
        *,
        comparison_targets: Optional[List[str]],
        comparison_scope_groups: Optional[Dict[str, List[Dict[str, Any]]]],
        device_names: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        slots: List[Dict[str, Any]] = []
        targets = [str(target).strip() for target in (comparison_targets or []) if str(target).strip()]
        groups = comparison_scope_groups if isinstance(comparison_scope_groups, dict) else {}
        for index, raw_target in enumerate(targets, start=1):
            scope_rows = groups.get(raw_target) if isinstance(groups.get(raw_target), list) else []
            resolved_row = next((row for row in scope_rows if isinstance(row, dict) and row.get("device")), None)
            resolved_device_code = str((resolved_row or {}).get("device") or raw_target).strip()
            resolved_device_name = str((resolved_row or {}).get("name") or (device_names or {}).get(resolved_device_code) or resolved_device_code).strip()
            slots.append(
                {
                    "slot_id": f"slot_{index}",
                    "ordinal": index,
                    "raw_target": raw_target,
                    "resolved_device_code": resolved_device_code,
                    "resolved_device_name": resolved_device_name,
                    "project_name": str((resolved_row or {}).get("project_name") or "").strip() or None,
                    "tg": str((resolved_row or {}).get("tg") or "").strip() or None,
                    "status": "resolved" if resolved_device_code else "pending",
                }
            )
        return slots

    @classmethod
    def _infer_analysis_scope(cls, normalized: List[Dict[str, Any]], data_type: str) -> Tuple[Optional[str], Optional[str]]:
        normalized_type = str(data_type or "").strip().lower()
        tags = {str(item.get("tag") or "").strip().lower() for item in normalized if str(item.get("tag") or "").strip()}
        single_phase_tags = {"ua", "ub", "uc", "ia", "ib", "ic"}
        three_phase_voltage_tags = {"ua", "ub", "uc"}
        three_phase_current_tags = {"ia", "ib", "ic"}

        if normalized_type in single_phase_tags:
            return "single_phase", "按单相分析"
        if normalized_type == "u_line" and tags and tags <= three_phase_voltage_tags:
            return ("three_phase_joint", "按三相联合分析") if len(tags) >= 2 else ("single_phase", "按单相分析")
        if normalized_type == "i" and tags and tags <= three_phase_current_tags:
            return ("three_phase_joint", "按三相联合分析") if len(tags) >= 2 else ("single_phase", "按单相分析")
        return None, None

    @classmethod
    def _build_single_analysis(
        cls,
        normalized: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        data_type: str,
    ) -> Dict[str, Any]:
        values = [item["value"] for item in normalized]
        unit = DATA_TYPE_UNITS.get(data_type, "")
        metric_name = DATA_TYPE_NAMES.get(data_type, data_type)
        analysis_scope_mode, analysis_scope_label = cls._infer_analysis_scope(normalized, data_type)
        q1, _, q3 = cls._quartiles(values)
        anomaly_points = cls._detect_anomaly_points(normalized, q1, q3)
        peak_point = max(normalized, key=lambda item: item["value"])
        valley_point = min(normalized, key=lambda item: item["value"])
        trend = statistics.get("trend", "平稳")
        change_rate = statistics.get("change_rate", 0)
        volatility_level = cls._volatility_level(statistics.get("cv"))
        peak_gap = round(peak_point["value"] - valley_point["value"], 2)
        analysis = {
            "mode": "single",
            "metric": metric_name,
            "unit": unit,
            "analysis_scope_mode": analysis_scope_mode,
            "analysis_scope_label": analysis_scope_label,
            "headline": (
                f"{analysis_scope_label}的{metric_name}呈{trend}趋势，整体波动{volatility_level}"
                if analysis_scope_label
                else f"{metric_name}呈{trend}趋势，整体波动{volatility_level}"
            ),
            "summary_cards": [
                {"label": "均值", "value": cls._format_number(statistics.get("avg"), unit), "detail": f"共 {statistics.get('count', len(normalized))} 条记录"},
                {"label": "趋势", "value": trend, "detail": f"阶段变化 {change_rate}%"},
                {"label": "波动", "value": volatility_level, "detail": f"CV {statistics.get('cv', 0)}%"},
                {"label": "异常点", "value": str(len(anomaly_points)), "detail": f"占比 {statistics.get('anomaly_ratio', 0)}%"},
            ],
            "insights": [
                f"该时段{metric_name}整体呈{trend}趋势，阶段变化约 {change_rate}%。",
                f"峰值出现在 {peak_point['time_label']}，达到 {cls._format_number(peak_point['value'], unit)}；谷值出现在 {valley_point['time_label']}，为 {cls._format_number(valley_point['value'], unit)}。",
                f"峰谷差为 {cls._format_number(peak_gap, unit)}，说明整体波动水平为{volatility_level}。",
                f"90% 的数据主要集中在 {cls._format_number(statistics.get('p5'), unit)} 到 {cls._format_number(statistics.get('p95'), unit)} 之间。",
            ],
            "peak_valley": {
                "peak_time": peak_point["time_label"],
                "peak_value": peak_point["value"],
                "valley_time": valley_point["time_label"],
                "valley_value": valley_point["value"],
                "gap": peak_gap,
            },
            "anomalies": [
                {"time": item["time_label"], "value": item["value"], "device": item["device"]}
                for item in anomaly_points[:8]
            ],
        }
        time_distribution = cls._hourly_distribution(normalized)
        if time_distribution:
            peak_hour = max(time_distribution, key=lambda item: item["avg_value"])
            valley_hour = min(time_distribution, key=lambda item: item["avg_value"])
            analysis["time_distribution"] = {
                "peak_hour": peak_hour["hour"],
                "peak_value": peak_hour["avg_value"],
                "valley_hour": valley_hour["hour"],
                "valley_value": valley_hour["avg_value"],
            }
            analysis["insights"].append(f"从小时分布看，{peak_hour['hour']} 时附近更接近高峰，{valley_hour['hour']} 时附近更接近低谷。")
        if anomaly_points:
            analysis["insights"].append(f"共识别 {len(anomaly_points)} 个异常点，可优先复核峰值附近时段和现场运行状态。")
        return analysis

    @classmethod
    def _build_comparison_analysis(
        cls,
        normalized: List[Dict[str, Any]],
        data_type: str,
        device_names: Optional[Dict[str, str]],
        comparison_slots: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        unit = DATA_TYPE_UNITS.get(data_type, "")
        metric_name = DATA_TYPE_NAMES.get(data_type, data_type)

        if comparison_slots and len(comparison_slots) > 1:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for slot in comparison_slots:
                slot_id = str(slot.get("slot_id") or "").strip()
                if not slot_id:
                    continue
                resolved_device_code = str(slot.get("resolved_device_code") or slot.get("raw_target") or "").strip()
                resolved_tg = str(slot.get("tg") or "").strip()
                grouped[slot_id] = [
                    item for item in normalized
                    if str(item.get("device") or "").strip() == resolved_device_code
                    and (not resolved_tg or not str(item.get("tg") or "").strip() or str(item.get("tg") or "").strip() == resolved_tg)
                ]
            pretty_names = {str(slot.get("slot_id")): cls._slot_display_name(slot) for slot in comparison_slots if str(slot.get("slot_id") or "").strip()}
        else:
            grouped = {}
            for item in normalized:
                grouped.setdefault(item["device"], []).append(item)
            pretty_names = {device: cls._display_name(device, device_names) for device in grouped}

        device_stats: Dict[str, Dict[str, Any]] = {
            key: cls._compute_stats(items) if items else {"avg": 0, "sum": 0, "max": 0, "min": 0, "cv": 0, "trend": "无数据", "anomaly_ratio": 0}
            for key, items in grouped.items()
        }
        avg_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("avg", 0), reverse=True)
        sum_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("sum", 0), reverse=True)
        stable_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("cv", float("inf")))
        anomaly_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("anomaly_ratio", 0), reverse=True)

        def ranking_names(items: List[Tuple[str, Dict[str, Any]]]) -> str:
            return " > ".join(pretty_names[item[0]] for item in items[:8])

        top_avg_device, top_avg_stats = avg_ranking[0]
        top_sum_device, top_sum_stats = sum_ranking[0]
        stable_device, stable_stats = stable_ranking[0]
        analysis = {
            "mode": "comparison",
            "metric": metric_name,
            "unit": unit,
            "headline": f"{pretty_names[top_avg_device]} 的{metric_name}均值最高，{pretty_names[stable_device]} 波动最稳定",
            "summary_cards": [
                {"label": "均值最高", "value": pretty_names[top_avg_device], "detail": cls._format_number(top_avg_stats.get("avg"), unit)},
                {"label": "总量最高", "value": pretty_names[top_sum_device], "detail": cls._format_number(top_sum_stats.get("sum"), unit)},
                {"label": "最稳定", "value": pretty_names[stable_device], "detail": f"CV {stable_stats.get('cv', 0)}%"},
                {"label": "异常最多", "value": pretty_names[anomaly_ranking[0][0]], "detail": f"{anomaly_ranking[0][1].get('anomaly_ratio', 0)}%"},
            ],
            "insights": [
                f"均值排名：{ranking_names(avg_ranking)}",
                f"总量排名：{ranking_names(sum_ranking)}",
                f"稳定性排名：{ranking_names(stable_ranking)}（CV 越低越稳定）",
                f"异常占比最高的是 {pretty_names[anomaly_ranking[0][0]]}，约为 {anomaly_ranking[0][1].get('anomaly_ratio', 0)}%。",
            ],
            "rankings": {
                "avg": [{"name": pretty_names[item[0]], "value": item[1].get("avg", 0)} for item in avg_ranking],
                "sum": [{"name": pretty_names[item[0]], "value": item[1].get("sum", 0)} for item in sum_ranking],
                "stability": [{"name": pretty_names[item[0]], "value": item[1].get("cv", 0)} for item in stable_ranking],
            },
            "devices": [
                {
                    "name": pretty_names[key],
                    "avg": stats.get("avg", 0),
                    "sum": stats.get("sum", 0),
                    "max": stats.get("max", 0),
                    "min": stats.get("min", 0),
                    "cv": stats.get("cv", 0),
                    "trend": stats.get("trend", "平稳"),
                    "anomaly_ratio": stats.get("anomaly_ratio", 0),
                }
                for key, stats in device_stats.items()
            ],
        }
        return analysis

    @classmethod
    def _normalize_records(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            value = cls._extract_value(item)
            time_value = item.get("logTime") or item.get("time") or item.get("date") or item.get("month") or item.get("year")
            if value is None or time_value in (None, ""):
                continue
            dt = cls._parse_datetime(time_value)
            normalized.append(
                {
                    "time": dt,
                    "time_label": cls._format_time(dt),
                    "time_value": dt.isoformat(),
                    "device": str(item.get("device") or item.get("deviceName") or "未知设备"),
                    "tag": str(item.get("tag") or ""),
                    "tg": str(item.get("tg") or "").strip() or None,
                    "value": float(value),
                }
            )
        normalized.sort(key=lambda item: (item["time"], item["device"], item.get("tag") or ""))
        if normalized:
            values = [item["value"] for item in normalized]
            q1, _, q3 = cls._quartiles(values)
            anomaly_pairs = {
                (item["time_value"], item["device"], item.get("tag") or "", item["value"])
                for item in cls._detect_anomaly_points(normalized, q1, q3)
            }
            for item in normalized:
                item["is_anomaly"] = (
                    item["time_value"],
                    item["device"],
                    item.get("tag") or "",
                    item["value"],
                ) in anomaly_pairs
        return normalized

    @classmethod
    def _compute_stats(cls, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        values = [item["value"] for item in records]
        if not values:
            return {}
        sorted_values = sorted(values)
        count = len(values)
        total = sum(values)
        avg = total / count
        q1, median, q3 = cls._quartiles(sorted_values)
        p5 = cls._percentile(sorted_values, 5)
        p95 = cls._percentile(sorted_values, 95)
        variance = sum((value - avg) ** 2 for value in values) / count
        std = variance ** 0.5
        cv = (std / avg * 100) if avg else 0
        anomaly_points = cls._detect_anomaly_points(records, q1, q3)
        third = max(1, count // 3)
        first_avg = sum(values[:third]) / third
        last_avg = sum(values[-third:]) / third
        change_rate = round(((last_avg - first_avg) / first_avg * 100), 2) if first_avg else 0
        trend = "上升" if change_rate > 5 else "下降" if change_rate < -5 else "平稳"
        return {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "avg": round(avg, 2),
            "sum": round(total, 2),
            "count": count,
            "median": round(median, 2),
            "std": round(std, 2),
            "cv": round(cv, 2),
            "p5": round(p5, 2),
            "p95": round(p95, 2),
            "trend": trend,
            "change_rate": change_rate,
            "anomaly_count": len(anomaly_points),
            "anomaly_ratio": round(len(anomaly_points) / count * 100, 2),
        }

    @classmethod
    def _hourly_distribution(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[float]] = {}
        for item in records:
            hour = item["time"].strftime("%H")
            buckets.setdefault(hour, []).append(item["value"])
        return [
            {"hour": hour, "avg_value": round(sum(values) / len(values), 2)}
            for hour, values in sorted(buckets.items())
            if values
        ]

    @classmethod
    def _detect_anomaly_points(cls, records: List[Dict[str, Any]], q1: float, q3: float) -> List[Dict[str, Any]]:
        if not records:
            return []
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [item for item in records if item["value"] < lower or item["value"] > upper]

    @classmethod
    def _extract_value(cls, item: Dict[str, Any]) -> Optional[float]:
        for field in ("value", "val", "average", "total", "diff", "max", "min", "avg"):
            value = item.get(field)
            if value is None or value == "":
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @classmethod
    def _quartiles(cls, values: List[float]) -> Tuple[float, float, float]:
        ordered = sorted(values)
        return cls._percentile(ordered, 25), cls._percentile(ordered, 50), cls._percentile(ordered, 75)

    @classmethod
    def _percentile(cls, values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        index = (len(values) - 1) * percentile / 100
        low = int(index)
        high = min(low + 1, len(values) - 1)
        fraction = index - low
        return float(values[low] + (values[high] - values[low]) * fraction)

    @classmethod
    def _parse_datetime(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        text = str(value)
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return datetime.min

    @classmethod
    def _format_time(cls, dt: datetime) -> str:
        if dt == datetime.min:
            return "未知时间"
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def _display_name(cls, device: str, device_names: Optional[Dict[str, str]]) -> str:
        if device_names and device in device_names:
            return device_names[device]
        return device

    @classmethod
    def _slot_display_name(cls, slot: Dict[str, Any]) -> str:
        ordinal = int(slot.get("ordinal") or 0)
        raw_target = str(slot.get("raw_target") or slot.get("resolved_device_code") or "").strip()
        if ordinal > 0 and raw_target:
            return f"第{ordinal}项 {raw_target}"
        return raw_target or "未命名对象"

    @classmethod
    def _format_number(cls, value: Any, unit: str = "") -> str:
        if value is None:
            return "-"
        try:
            number = float(value)
            return f"{number:.2f}{(' ' + unit) if unit else ''}"
        except (TypeError, ValueError):
            return str(value)

    @classmethod
    def _volatility_level(cls, cv: Optional[float]) -> str:
        if cv is None:
            return "未知"
        if cv < 10:
            return "低"
        if cv < 30:
            return "中"
        return "高"

    @classmethod
    def _infer_query_kind(
        cls,
        analysis: Dict[str, Any],
        user_query: str,
        comparison_slots: Optional[List[Dict[str, Any]]],
    ) -> str:
        compact = str(user_query or "").replace(" ", "")
        if len(comparison_slots or []) > 1 or str(analysis.get("mode") or "") == "comparison":
            return "comparison_series"
        if any(keyword in compact for keyword in ANOMALY_HINTS):
            return "anomaly_analysis"
        if any(keyword in compact for keyword in DISTRIBUTION_HINTS):
            return "distribution_analysis"
        return "single_series"

    @classmethod
    def _build_recommended_chart_types(cls, query_kind: str) -> List[str]:
        mapping = {
            "single_series": ["line", "bar", "heatmap"],
            "comparison_series": ["line", "bar", "boxplot"],
            "anomaly_analysis": ["scatter", "line", "heatmap"],
            "distribution_analysis": ["boxplot", "heatmap", "line"],
        }
        return mapping.get(query_kind, ["line", "bar", "heatmap"])

    @classmethod
    def _build_follow_up_suggestions(cls, query_kind: str) -> List[Dict[str, str]]:
        mapping = {
            "single_series": [
                {"label": "趋势图", "chart_type": "line"},
                {"label": "柱状图", "chart_type": "bar"},
                {"label": "热力图", "chart_type": "heatmap"},
            ],
            "comparison_series": [
                {"label": "趋势对比图", "chart_type": "line"},
                {"label": "均值柱状图", "chart_type": "bar"},
                {"label": "箱线分布图", "chart_type": "boxplot"},
            ],
            "anomaly_analysis": [
                {"label": "异常散点图", "chart_type": "scatter"},
                {"label": "趋势图", "chart_type": "line"},
            ],
            "distribution_analysis": [
                {"label": "箱线图", "chart_type": "boxplot"},
                {"label": "热力图", "chart_type": "heatmap"},
            ],
        }
        return mapping.get(query_kind, [{"label": "趋势图", "chart_type": "line"}])

    @classmethod
    def _build_data_signature(cls, normalized_records: List[Dict[str, Any]]) -> str:
        digest = hashlib.sha1()
        for item in normalized_records[:2000]:
            digest.update(
                "|".join(
                    [
                        str(item.get("time_value") or ""),
                        str(item.get("device") or ""),
                        str(item.get("tag") or ""),
                        str(item.get("tg") or ""),
                        str(item.get("value") or ""),
                    ]
                ).encode("utf-8", errors="ignore")
            )
        return digest.hexdigest()
