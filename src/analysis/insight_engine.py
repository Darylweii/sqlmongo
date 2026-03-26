from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


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


class InsightEngine:
    @classmethod
    def _infer_analysis_scope(cls, normalized: List[Dict[str, Any]], data_type: str) -> Tuple[Optional[str], Optional[str]]:
        normalized_type = str(data_type or "").strip().lower()
        tags = {str(item.get("tag") or "").strip().lower() for item in (normalized or []) if str(item.get("tag") or "").strip()}

        single_phase_tags = {"ua", "ub", "uc", "ia", "ib", "ic"}
        three_phase_voltage_tags = {"ua", "ub", "uc"}
        three_phase_current_tags = {"ia", "ib", "ic"}

        if normalized_type in single_phase_tags:
            return "single_phase", "按单相分析"

        if normalized_type == "u_line" and tags and tags <= three_phase_voltage_tags:
            return (
                ("three_phase_joint", "按三相联合分析")
                if len(tags) >= 2
                else ("single_phase", "按单相分析")
            )

        if normalized_type == "i" and tags and tags <= three_phase_current_tags:
            return (
                ("three_phase_joint", "按三相联合分析")
                if len(tags) >= 2
                else ("single_phase", "按单相分析")
            )

        return None, None

    @classmethod
    def build(
        cls,
        records: Optional[List[Dict[str, Any]]],
        statistics: Optional[Dict[str, Any]],
        data_type: str = "ep",
        device_codes: Optional[List[str]] = None,
        device_names: Optional[Dict[str, str]] = None,
        user_query: str = "",
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        normalized = cls._normalize_records(records or [])
        if not normalized:
            return None, []

        unique_devices = sorted({item["device"] for item in normalized if item.get("device")})
        devices = device_codes or unique_devices
        if len(set(devices or unique_devices)) > 1 or len(unique_devices) > 1:
            return cls._build_comparison_analysis(normalized, data_type, device_names, user_query)
        return cls._build_single_analysis(normalized, statistics or cls._compute_stats(normalized), data_type, device_names, user_query)

    @classmethod
    def _build_single_analysis(
        cls,
        normalized: List[Dict[str, Any]],
        statistics: Dict[str, Any],
        data_type: str,
        device_names: Optional[Dict[str, str]],
        user_query: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        values = [item["value"] for item in normalized]
        unit = DATA_TYPE_UNITS.get(data_type, "")
        metric_name = DATA_TYPE_NAMES.get(data_type, data_type)
        analysis_scope_mode, analysis_scope_label = cls._infer_analysis_scope(normalized, data_type)
        q1, median, q3 = cls._quartiles(values)
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
            "headline": (f"{analysis_scope_label}?{metric_name}???{trend}????????{volatility_level}" if analysis_scope_label else f"{metric_name}???{trend}????????{volatility_level}"),
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
                {
                    "time": item["time_label"],
                    "value": item["value"],
                    "device": item["device"],
                }
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
            analysis["insights"].append(
                f"从小时分布看，{peak_hour['hour']} 时附近更接近高峰，{valley_hour['hour']} 时附近更接近低谷。"
            )
        if anomaly_points:
            analysis["insights"].append(
                f"共识别 {len(anomaly_points)} 个异常点，可优先复核峰值附近时段和现场运行状态。"
            )

        chart_specs = [
            {
                "id": "trend-line",
                "title": f"{metric_name}趋势图",
                "height": 360,
                "option": cls._build_single_line_option(normalized, data_type, anomaly_points),
            },
            {
                "id": "hourly-bar",
                "title": f"{metric_name}分时均值",
                "height": 320,
                "option": cls._build_hourly_bar_option(time_distribution, metric_name, unit),
            },
        ]
        return analysis, [spec for spec in chart_specs if spec.get("option")]

    @classmethod
    def _build_comparison_analysis(
        cls,
        normalized: List[Dict[str, Any]],
        data_type: str,
        device_names: Optional[Dict[str, str]],
        user_query: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        unit = DATA_TYPE_UNITS.get(data_type, "")
        metric_name = DATA_TYPE_NAMES.get(data_type, data_type)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in normalized:
            grouped.setdefault(item["device"], []).append(item)
        device_stats: Dict[str, Dict[str, Any]] = {device: cls._compute_stats(items) for device, items in grouped.items()}
        pretty_names = {device: cls._display_name(device, device_names) for device in grouped}
        avg_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("avg", 0), reverse=True)
        sum_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("sum", 0), reverse=True)
        stable_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("cv", float("inf")))
        anomaly_ranking = sorted(device_stats.items(), key=lambda item: item[1].get("anomaly_ratio", 0), reverse=True)
        rank_display_limit = 8

        def ranking_names(items: List[Tuple[str, Dict[str, Any]]]) -> str:
            visible_items = items if len(items) <= rank_display_limit else items[:rank_display_limit]
            content = " > ".join(pretty_names[item[0]] for item in visible_items)
            if len(items) > rank_display_limit:
                content += f" > ...（共 {len(items)} 个设备）"
            return content

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
                f"均值排名：" + ranking_names(avg_ranking),
                f"总量排名：" + ranking_names(sum_ranking),
                f"稳定性排名：" + ranking_names(stable_ranking) + "（CV 越低越稳定）",
                f"异常占比最高的是 {pretty_names[anomaly_ranking[0][0]]}，约为 {anomaly_ranking[0][1].get('anomaly_ratio', 0)}%。",
            ],
            "rankings": {
                "avg": [{"name": pretty_names[item[0]], "value": item[1].get("avg", 0)} for item in avg_ranking],
                "sum": [{"name": pretty_names[item[0]], "value": item[1].get("sum", 0)} for item in sum_ranking],
                "stability": [{"name": pretty_names[item[0]], "value": item[1].get("cv", 0)} for item in stable_ranking],
            },
            "devices": [
                {
                    "name": pretty_names[device],
                    "avg": stats.get("avg", 0),
                    "sum": stats.get("sum", 0),
                    "max": stats.get("max", 0),
                    "min": stats.get("min", 0),
                    "cv": stats.get("cv", 0),
                    "trend": stats.get("trend", "平稳"),
                    "anomaly_ratio": stats.get("anomaly_ratio", 0),
                }
                for device, stats in device_stats.items()
            ],
        }

        line_option = cls._build_comparison_line_option(grouped, data_type, device_names)
        bar_option = cls._build_comparison_bar_option(device_stats, metric_name, unit, device_names)
        radar_option = cls._build_radar_option(device_stats, device_names)
        boxplot_option = cls._build_boxplot_option(grouped, data_type, device_names)
        chart_specs = [
            {"id": "comparison-line", "title": f"{metric_name}趋势对比", "height": 380, "option": line_option},
            {"id": "comparison-bar", "title": f"{metric_name}均值/总量对比", "height": 360, "option": bar_option},
            {"id": "comparison-radar", "title": f"多维指标雷达图", "height": 360, "option": radar_option},
            {"id": "comparison-boxplot", "title": f"分布箱线图", "height": 360, "option": boxplot_option},
        ]
        return analysis, [spec for spec in chart_specs if spec.get("option")]

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
            normalized.append({
                "time": dt,
                "time_label": cls._format_time(dt),
                "time_value": dt.isoformat(),
                "device": str(item.get("device") or item.get("deviceName") or "未知设备"),
                "tag": str(item.get("tag") or ""),
                "value": float(value),
            })
        normalized.sort(key=lambda item: item["time"])
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
    def _build_single_line_option(cls, records: List[Dict[str, Any]], data_type: str, anomaly_points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        unit = DATA_TYPE_UNITS.get(data_type, "")
        metric_name = DATA_TYPE_NAMES.get(data_type, data_type)
        points = cls._downsample_points(records)
        anomaly_pairs = [[item["time_value"], item["value"]] for item in anomaly_points[:30]]
        series = [
            {
                "name": metric_name,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": [[item["time_value"], item["value"]] for item in points],
                "markPoint": {"data": [{"type": "max", "name": "最大值"}, {"type": "min", "name": "最小值"}]},
                "markLine": {"data": [{"type": "average", "name": "平均值"}]},
            }
        ]
        if anomaly_pairs:
            series.append(
                {
                    "name": "异常点",
                    "type": "scatter",
                    "symbolSize": 10,
                    "itemStyle": {"color": "#ef4444"},
                    "data": anomaly_pairs,
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

    @classmethod
    def _build_hourly_bar_option(cls, time_distribution: Optional[List[Dict[str, Any]]], metric_name: str, unit: str) -> Optional[Dict[str, Any]]:
        if not time_distribution:
            return None
        return {
            "tooltip": {"trigger": "axis"},
            "grid": {"left": 40, "right": 24, "top": 32, "bottom": 36},
            "xAxis": {"type": "category", "data": [item["hour"] for item in time_distribution]},
            "yAxis": {"type": "value", "name": unit},
            "series": [
                {
                    "name": f"平均{metric_name}",
                    "type": "bar",
                    "data": [item["avg_value"] for item in time_distribution],
                    "itemStyle": {"color": "#6366f1"},
                }
            ],
        }

    @classmethod
    def _build_comparison_line_option(
        cls,
        grouped: Dict[str, List[Dict[str, Any]]],
        data_type: str,
        device_names: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        if not grouped:
            return None
        unit = DATA_TYPE_UNITS.get(data_type, "")
        legend = []
        series = []
        for device, records in grouped.items():
            display_name = cls._display_name(device, device_names)
            legend.append(display_name)
            points = cls._downsample_points(records)
            series.append({
                "name": display_name,
                "type": "line",
                "smooth": True,
                "showSymbol": False,
                "data": [[item["time_value"], item["value"]] for item in points],
            })
        return {
            "tooltip": {"trigger": "axis"},
            "legend": {"data": legend},
            "grid": {"left": 40, "right": 24, "top": 48, "bottom": 48},
            "xAxis": {"type": "time"},
            "yAxis": {"type": "value", "name": unit},
            "series": series,
        }

    @classmethod
    def _build_comparison_bar_option(
        cls,
        device_stats: Dict[str, Dict[str, Any]],
        metric_name: str,
        unit: str,
        device_names: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        if not device_stats:
            return None
        labels = [cls._display_name(device, device_names) for device in device_stats]
        return {
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "legend": {"data": ["平均值", "总量"]},
            "grid": {"left": 48, "right": 24, "top": 48, "bottom": 36},
            "xAxis": {"type": "category", "data": labels},
            "yAxis": [
                {"type": "value", "name": f"平均值({unit})"},
                {"type": "value", "name": f"总量({unit})"},
            ],
            "series": [
                {
                    "name": "平均值",
                    "type": "bar",
                    "data": [device_stats[device].get("avg", 0) for device in device_stats],
                    "itemStyle": {"color": "#6366f1"},
                },
                {
                    "name": "总量",
                    "type": "bar",
                    "yAxisIndex": 1,
                    "data": [device_stats[device].get("sum", 0) for device in device_stats],
                    "itemStyle": {"color": "#10b981"},
                },
            ],
        }

    @classmethod
    def _build_radar_option(cls, device_stats: Dict[str, Dict[str, Any]], device_names: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        if len(device_stats) < 2:
            return None
        series_data = []
        max_avg = max((stats.get("avg", 0) for stats in device_stats.values()), default=1)
        max_max = max((stats.get("max", 0) for stats in device_stats.values()), default=1)
        max_sum = max((stats.get("sum", 0) for stats in device_stats.values()), default=1)
        indicators = [
            {"name": "平均值", "max": max_avg or 1},
            {"name": "峰值", "max": max_max or 1},
            {"name": "总量", "max": max_sum or 1},
            {"name": "稳定性", "max": 100},
            {"name": "异常率", "max": 100},
        ]
        for device, stats in device_stats.items():
            series_data.append({
                "name": cls._display_name(device, device_names),
                "value": [
                    stats.get("avg", 0),
                    stats.get("max", 0),
                    stats.get("sum", 0),
                    max(0, round(100 - stats.get("cv", 0), 2)),
                    max(0, round(100 - stats.get("anomaly_ratio", 0), 2)),
                ],
            })
        return {
            "tooltip": {},
            "legend": {"data": [item["name"] for item in series_data]},
            "radar": {"indicator": indicators, "radius": "60%"},
            "series": [{"type": "radar", "data": series_data}],
        }

    @classmethod
    def _build_boxplot_option(
        cls,
        grouped: Dict[str, List[Dict[str, Any]]],
        data_type: str,
        device_names: Optional[Dict[str, str]],
    ) -> Optional[Dict[str, Any]]:
        if len(grouped) < 2:
            return None
        labels = []
        data = []
        for device, records in grouped.items():
            values = sorted(item["value"] for item in records)
            if len(values) < 2:
                continue
            q1, median, q3 = cls._quartiles(values)
            labels.append(cls._display_name(device, device_names))
            data.append([
                round(min(values), 2),
                round(q1, 2),
                round(median, 2),
                round(q3, 2),
                round(max(values), 2),
            ])
        if not data:
            return None
        unit = DATA_TYPE_UNITS.get(data_type, "")
        return {
            "tooltip": {"trigger": "item"},
            "grid": {"left": 48, "right": 24, "top": 32, "bottom": 48},
            "xAxis": {"type": "category", "data": labels},
            "yAxis": {"type": "value", "name": unit},
            "series": [{"type": "boxplot", "data": data}],
        }

    @classmethod
    def _hourly_distribution(cls, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[float]] = {}
        for item in records:
            hour = item["time"].strftime("%H")
            buckets.setdefault(hour, []).append(item["value"])
        result = [
            {"hour": hour, "avg_value": round(sum(values) / len(values), 2)}
            for hour, values in sorted(buckets.items())
            if values
        ]
        return result

    @classmethod
    def _downsample_points(cls, records: List[Dict[str, Any]], max_points: int = 180) -> List[Dict[str, Any]]:
        if len(records) <= max_points:
            return records
        step = max(1, len(records) // max_points)
        sampled = records[::step]
        if sampled[-1] != records[-1]:
            sampled.append(records[-1])
        return sampled

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
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y-%m",
            "%Y",
        ):
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
