"""分析 MongoDB 中的集合分布，并导出统计结果。"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def localize_argparse() -> None:
    translations = {
        'usage: ': '\u7528\u6cd5: ',
        'options': '\u53ef\u9009\u53c2\u6570',
        'positional arguments': '\u4f4d\u7f6e\u53c2\u6570',
        'show this help message and exit': '\u663e\u793a\u6b64\u5e2e\u52a9\u4fe1\u606f\u5e76\u9000\u51fa',
    }
    argparse._ = lambda text: translations.get(text, text)


localize_argparse()

from dotenv import load_dotenv
from pymongo import MongoClient


MONTHLY_COLLECTION_PATTERN = re.compile(r'^(?P<prefix>.+)_(?P<month>\d{6})$')


def create_mongo_client() -> MongoClient:
    """根据环境变量创建 MongoDB 客户端。"""
    load_dotenv()
    return MongoClient(
        host=os.getenv('MONGODB_HOST', '127.0.0.1'),
        port=int(os.getenv('MONGODB_PORT', 27017)),
        username=os.getenv('MONGODB_USER') or None,
        password=os.getenv('MONGODB_PASSWORD') or None,
        authSource=os.getenv('MONGODB_AUTH_SOURCE', 'admin'),
    )


def _summarize_months(months: list[str]) -> dict[str, Any]:
    months = sorted(months)
    if not months:
        return {'count': 0, 'date_range': '无'}
    return {'count': len(months), 'date_range': f'{months[0]} ~ {months[-1]}'}


def analyze_mongodb(output_path: str = 'data/mongodb_analysis.json') -> dict[str, Any]:
    """分析集合命名模式，并将结果保存为 JSON 文件。"""
    client = create_mongo_client()
    database_name = os.getenv('MONGODB_DATABASE', 'sensor_db')
    db = client[database_name]

    collections = sorted(db.list_collection_names())
    source_data_types: dict[str, dict[str, Any]] = {}
    other_types: dict[str, dict[str, Any]] = {}
    source_months: dict[str, list[str]] = defaultdict(list)
    other_months: dict[str, list[str]] = defaultdict(list)

    for collection_name in collections:
        match = MONTHLY_COLLECTION_PATTERN.match(collection_name)
        if match:
            prefix = match.group('prefix')
            month = match.group('month')
            if prefix.startswith('source_data_'):
                tag = prefix.removeprefix('source_data_')
                source_months[tag].append(month)
            else:
                other_months[prefix].append(month)
        else:
            other_months[collection_name].append('no_date')

    for tag, months in sorted(source_months.items()):
        source_data_types[tag] = {
            'prefix': f'source_data_{tag}',
            **_summarize_months(months),
        }

    for prefix, months in sorted(other_months.items()):
        if months == ['no_date']:
            other_types[prefix] = {'count': 1, 'date_range': '无'}
        else:
            other_types[prefix] = _summarize_months(months)

    result = {
        'database': database_name,
        'total_collections': len(collections),
        'unique_types': len(source_data_types),
        'source_data_types': source_data_types,
        'other_types': other_types,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'数据库: {database_name}')
    print(f'集合总数: {len(collections)}')
    print(f'source_data 标签数: {len(source_data_types)}')
    print()
    print(f"### source_data_* 标签集合 ({len(source_data_types)})")
    print('-' * 72)
    print(f"{'标签':<20} {'月份数':<10} {'时间范围':<25}")
    print('-' * 72)
    for tag, info in source_data_types.items():
        print(f"{tag:<20} {info['count']:<10} {info['date_range']:<25}")

    print()
    print(f"### 其他集合分组 ({len(other_types)})")
    print('-' * 72)
    print(f"{'前缀/名称':<32} {'数量':<10} {'时间范围':<25}")
    print('-' * 72)
    for prefix, info in other_types.items():
        print(f"{prefix:<32} {info['count']:<10} {info['date_range']:<25}")

    print()
    print(f'分析结果已保存到: {output_file.as_posix()}')
    client.close()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='分析 MongoDB 集合命名模式')
    parser.add_argument('--output', default='data/mongodb_analysis.json', help='输出 JSON 文件路径')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_mongodb(args.output)
