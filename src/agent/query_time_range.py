from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional
import re


DATE_SEPARATOR_PATTERN = r'(?:[-/\u5e74\u6708\u65e5\u53f7\u5230\u81f3~\u2014_.?\uff1f,\uff0c:\uff1a\s]+)'
ISO_RANGE_PATTERN = re.compile(
    r'(?P<start>\d{4}-\d{1,2}-\d{1,2})\s*(?:\u5230|\u81f3|~|-|\u2014)\s*(?P<end>\d{4}-\d{1,2}-\d{1,2})'
)
CN_DAY_RANGE_PATTERN = re.compile(
    r'(?P<y1>\d{4})\u5e74(?P<m1>\d{1,2})\u6708(?P<d1>\d{1,2})\u65e5?\s*(?:\u5230|\u81f3|~|-|\u2014)\s*(?:(?P<y2>\d{4})\u5e74)?(?P<m2>\d{1,2})\u6708(?P<d2>\d{1,2})\u65e5?'
)
EXACT_MONTH_PATTERN = re.compile(r'(?P<year>\d{4})[-/\u5e74](?P<month>\d{1,2})(?:\u6708)?')
RECENT_DAYS_PATTERN = re.compile(r'(?:\u6700\u8fd1|\u8fd1|\u8fc7\u53bb)(?P<days>\d{1,3})\u5929')
RELAXED_EXACT_DAY_PATTERN = re.compile(
    rf'(?P<year>20\d{{2}}){DATE_SEPARATOR_PATTERN}(?P<month>\d{{1,2}}){DATE_SEPARATOR_PATTERN}(?P<day>\d{{1,2}})(?!\d)'
)
RELAXED_EXACT_MONTH_PATTERN = re.compile(
    rf'(?P<year>20\d{{2}}){DATE_SEPARATOR_PATTERN}(?P<month>\d{{1,2}})(?!\d)'
)
AMBIGUOUS_PAIRS = (
    ('\u4e0a\u5468', '\u672c\u5468'),
    ('\u4eca\u5929', '\u6628\u5929'),
    ('\u4eca\u5929', '\u524d\u5929'),
    ('\u672c\u6708', '\u4e0a\u6708'),
)


def build_month_range(year: int, month: int) -> Optional[Dict[str, str]]:
    if month < 1 or month > 12:
        return None
    start_dt = datetime(year, month, 1)
    if month == 12:
        next_month = datetime(year + 1, 1, 1)
    else:
        next_month = datetime(year, month + 1, 1)
    end_dt = next_month - timedelta(days=1)
    return {
        'start_time': start_dt.strftime('%Y-%m-%d'),
        'end_time': end_dt.strftime('%Y-%m-%d'),
    }


def resolve_time_range_from_query(query_text: str, now: Optional[datetime] = None) -> Optional[Dict[str, str]]:
    text = re.sub(r'\s+', '', str(query_text or ''))
    if not text:
        return None

    if any(left in text and right in text for left, right in AMBIGUOUS_PAIRS):
        return None

    now = now or datetime.now()
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    iso_range = ISO_RANGE_PATTERN.search(text)
    if iso_range:
        start_dt = datetime(int(iso_range.group('start')[0:4]), int(iso_range.group('start')[5:7]), int(iso_range.group('start')[8:10]))
        end_dt = datetime(int(iso_range.group('end')[0:4]), int(iso_range.group('end')[5:7]), int(iso_range.group('end')[8:10]))
        return {
            'start_time': start_dt.strftime('%Y-%m-%d'),
            'end_time': end_dt.strftime('%Y-%m-%d'),
        }

    cn_day_range = CN_DAY_RANGE_PATTERN.search(text)
    if cn_day_range:
        end_year = int(cn_day_range.group('y2') or cn_day_range.group('y1'))
        start_dt = datetime(int(cn_day_range.group('y1')), int(cn_day_range.group('m1')), int(cn_day_range.group('d1')))
        end_dt = datetime(end_year, int(cn_day_range.group('m2')), int(cn_day_range.group('d2')))
        return {
            'start_time': start_dt.strftime('%Y-%m-%d'),
            'end_time': end_dt.strftime('%Y-%m-%d'),
        }

    exact_cn_day = re.search(r'(?P<year>\d{4})\u5e74(?P<month>\d{1,2})\u6708(?P<day>\d{1,2})\u65e5', text)
    if exact_cn_day:
        target_dt = datetime(int(exact_cn_day.group('year')), int(exact_cn_day.group('month')), int(exact_cn_day.group('day')))
        target = target_dt.strftime('%Y-%m-%d')
        return {'start_time': target, 'end_time': target}

    exact_iso_day = re.search(r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})', text)
    if exact_iso_day:
        target_dt = datetime(int(exact_iso_day.group('year')), int(exact_iso_day.group('month')), int(exact_iso_day.group('day')))
        target = target_dt.strftime('%Y-%m-%d')
        return {'start_time': target, 'end_time': target}

    exact_md_day = re.search(r'(?<!\d)(?P<month>\d{1,2})\u6708(?P<day>\d{1,2})\u65e5', text)
    if exact_md_day and '\u5e74' not in text:
        target_dt = datetime(now.year, int(exact_md_day.group('month')), int(exact_md_day.group('day')))
        target = target_dt.strftime('%Y-%m-%d')
        return {'start_time': target, 'end_time': target}

    relaxed_exact_day = RELAXED_EXACT_DAY_PATTERN.search(text)
    if relaxed_exact_day:
        try:
            target_dt = datetime(
                int(relaxed_exact_day.group('year')),
                int(relaxed_exact_day.group('month')),
                int(relaxed_exact_day.group('day')),
            )
        except ValueError:
            target_dt = None
        if target_dt is not None:
            target = target_dt.strftime('%Y-%m-%d')
            return {'start_time': target, 'end_time': target}

    exact_month = EXACT_MONTH_PATTERN.search(text)
    has_explicit_cn_day = bool(re.search(r'(?:\d{4}\u5e74)?\d{1,2}\u6708\d{1,2}\u65e5', text))
    has_explicit_iso_day = bool(re.search(r'\d{4}-\d{1,2}-\d{1,2}', text))
    if exact_month and not has_explicit_cn_day and not has_explicit_iso_day:
        return build_month_range(int(exact_month.group('year')), int(exact_month.group('month')))

    relaxed_exact_month = RELAXED_EXACT_MONTH_PATTERN.search(text)
    if relaxed_exact_month and not RELAXED_EXACT_DAY_PATTERN.search(text):
        month_range = build_month_range(int(relaxed_exact_month.group('year')), int(relaxed_exact_month.group('month')))
        if month_range:
            return month_range

    recent_days = RECENT_DAYS_PATTERN.search(text)
    if recent_days:
        days = max(int(recent_days.group('days')), 1)
        start_dt = today - timedelta(days=days)
        return {
            'start_time': start_dt.strftime('%Y-%m-%d'),
            'end_time': today.strftime('%Y-%m-%d'),
        }

    if any(token in text for token in ['\u6700\u8fd1\u4e00\u5468', '\u6700\u8fd17\u5929', '\u8fd1\u4e00\u5468', '\u8fd17\u5929', '\u8fc7\u53bb\u4e00\u5468', '\u8fc7\u53bb7\u5929', '\u6700\u8fd1\u4e03\u5929']):
        start_dt = today - timedelta(days=7)
        return {
            'start_time': start_dt.strftime('%Y-%m-%d'),
            'end_time': today.strftime('%Y-%m-%d'),
        }

    if any(token in text for token in ['\u4eca\u5929', '\u4eca\u65e5']):
        target = today.strftime('%Y-%m-%d')
        return {'start_time': target, 'end_time': target}

    if any(token in text for token in ['\u6628\u5929', '\u6628\u65e5']):
        target_dt = today - timedelta(days=1)
        target = target_dt.strftime('%Y-%m-%d')
        return {'start_time': target, 'end_time': target}

    if '\u524d\u5929' in text:
        target_dt = today - timedelta(days=2)
        target = target_dt.strftime('%Y-%m-%d')
        return {'start_time': target, 'end_time': target}

    if '\u672c\u5468' in text:
        monday = today - timedelta(days=today.weekday())
        return {
            'start_time': monday.strftime('%Y-%m-%d'),
            'end_time': today.strftime('%Y-%m-%d'),
        }

    if '\u4e0a\u5468' in text:
        current_monday = today - timedelta(days=today.weekday())
        last_monday = current_monday - timedelta(days=7)
        last_sunday = current_monday - timedelta(days=1)
        return {
            'start_time': last_monday.strftime('%Y-%m-%d'),
            'end_time': last_sunday.strftime('%Y-%m-%d'),
        }

    if '\u672c\u6708' in text:
        first_day = today.replace(day=1)
        return {
            'start_time': first_day.strftime('%Y-%m-%d'),
            'end_time': today.strftime('%Y-%m-%d'),
        }

    if '\u4e0a\u6708' in text:
        last_day_prev_month = today.replace(day=1) - timedelta(days=1)
        return build_month_range(last_day_prev_month.year, last_day_prev_month.month)

    return None
