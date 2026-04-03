from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


def display_name(key: str, device_names: Dict[str, str] | None) -> str:
    if device_names and key in device_names:
        return str(device_names[key] or key)
    return str(key or "未知对象")


def build_slot_label(slot: Dict[str, Any]) -> str:
    ordinal = int(slot.get("ordinal") or 0)
    raw_target = str(
        slot.get("raw_target")
        or slot.get("resolved_device_code")
        or slot.get("resolved_device_name")
        or ""
    ).strip()
    base_label = raw_target or "未命名对象"
    if ordinal > 0:
        return f"第{ordinal}项 {base_label}"
    return base_label


def downsample_points(records: List[Dict[str, Any]], max_points: int = 240) -> List[Dict[str, Any]]:
    if len(records) <= max_points:
        return list(records)
    step = max(1, len(records) // max_points)
    sampled = list(records[::step])
    if sampled and sampled[-1] != records[-1]:
        sampled.append(records[-1])
    return sampled


def group_records(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[str(item.get(key) or "未分类")].append(item)
    return dict(grouped)


def filter_records_by_slot(records: Iterable[Dict[str, Any]], slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    resolved_device_code = str(slot.get("resolved_device_code") or slot.get("raw_target") or "").strip()
    resolved_tg = str(slot.get("tg") or "").strip()
    results: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        device_code = str(item.get("device") or "").strip()
        if resolved_device_code and device_code != resolved_device_code:
            continue
        item_tg = str(item.get("tg") or "").strip()
        if resolved_tg and item_tg and item_tg != resolved_tg:
            continue
        results.append(item)
    return results


def build_slot_groups(records: Iterable[Dict[str, Any]], comparison_slots: List[Dict[str, Any]] | None) -> List[Tuple[str, List[Dict[str, Any]]]]:
    slot_groups: List[Tuple[str, List[Dict[str, Any]]]] = []
    for slot in comparison_slots or []:
        if not isinstance(slot, dict):
            continue
        slot_records = filter_records_by_slot(records, slot)
        if not slot_records:
            continue
        slot_groups.append((build_slot_label(slot), slot_records))
    return slot_groups


def sorted_group_items(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, List[Dict[str, Any]]]]:
    return sorted(grouped.items(), key=lambda item: item[0])


def compute_boxplot_values(values: List[float]) -> List[float] | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    return [
        round(min(ordered), 2),
        round(percentile(ordered, 25), 2),
        round(percentile(ordered, 50), 2),
        round(percentile(ordered, 75), 2),
        round(max(ordered), 2),
    ]


def percentile(values: List[float], ratio: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * ratio / 100
    low = int(index)
    high = min(low + 1, len(values) - 1)
    fraction = index - low
    return float(values[low] + (values[high] - values[low]) * fraction)


def hourly_average_records(records: Iterable[Dict[str, Any]], *, group_key: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for item in records:
        dt = item.get("time")
        if dt is None:
            continue
        hour = dt.strftime("%H")
        group_value = str(item.get(group_key) or "默认")
        buckets[hour][group_value].append(float(item.get("value") or 0.0))

    result: Dict[str, Dict[str, float]] = {}
    for hour, values_by_group in buckets.items():
        result[hour] = {
            group_value: round(sum(values) / len(values), 2)
            for group_value, values in values_by_group.items()
            if values
        }
    return dict(sorted(result.items(), key=lambda item: item[0]))
