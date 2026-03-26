"""查看指定设备的 MongoDB 实际数据，并检测重复时间点。"""

from __future__ import annotations

import argparse
import os
import sys
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


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def create_mongo_client() -> MongoClient:
    """根据环境变量创建 MongoDB 客户端。"""
    load_dotenv()
    for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        os.environ.pop(key, None)
    os.environ['NO_PROXY'] = '*'
    return MongoClient(
        host=os.getenv('MONGODB_HOST', '127.0.0.1'),
        port=int(os.getenv('MONGODB_PORT', 27017)),
        username=os.getenv('MONGODB_USER') or None,
        password=os.getenv('MONGODB_PASSWORD') or None,
        authSource=os.getenv('MONGODB_AUTH_SOURCE', 'admin'),
    )


def resolve_collection_name(db, month: str, data_type: str, explicit_name: str | None) -> str:
    """自动解析集合名称。"""
    if explicit_name:
        return explicit_name
    candidates = [f'source_data_{data_type}_{month}', f'sensor_data_{month}']
    collection_names = set(db.list_collection_names())
    for candidate in candidates:
        if candidate in collection_names:
            return candidate
    available = sorted(name for name in collection_names if month in name)
    raise RuntimeError(
        f'未找到月份 {month} 对应的集合。候选: {candidates}; 可用示例: {available[:10]}'
    )


def detect_fields(document: dict[str, Any]) -> tuple[str | None, str | None]:
    """从样例文档中识别时间字段和数值字段。"""
    time_field = next((field for field in ('logTime', 'time', 'ts', 'timestamp') if field in document), None)
    value_field = next((field for field in ('val', 'value', 'v') if field in document), None)
    return time_field, value_field


def check_actual_data(
    device: str = 'a1_b9',
    month: str = '202401',
    data_type: str = 'ep',
    limit: int = 10,
    collection_name: str | None = None,
) -> None:
    """打印样例记录和重复时间点检查结果。"""
    client = create_mongo_client()
    database_name = os.getenv('MONGODB_DATABASE', 'sensor_db')
    db = client[database_name]

    resolved_collection = resolve_collection_name(db, month, data_type, collection_name)
    collection = db[resolved_collection]

    query: dict[str, Any] = {'device': device}
    if resolved_collection.startswith('sensor_data_'):
        query['tag'] = data_type

    sample_doc = collection.find_one(query)
    if not sample_doc:
        print(f'在集合 {resolved_collection} 中未找到设备 {device} 的数据。')
        client.close()
        return

    time_field, value_field = detect_fields(sample_doc)
    sort_field = time_field or '_id'
    results = list(collection.find(query).sort(sort_field, 1).limit(limit))

    print('=' * 72)
    print(f'集合: {resolved_collection}')
    print(f'设备: {device}，标签: {data_type}，样例条数: {len(results)}')
    print('=' * 72)
    print(f"时间字段: {time_field or '无'}")
    print(f"数值字段: {value_field or '无'}")
    print(f'文档字段: {sorted(sample_doc.keys())}')
    print()

    for index, doc in enumerate(results, start=1):
        print(f'记录 {index}:')
        print(f"  _id: {doc.get('_id')}")
        print(f"  device: {doc.get('device')}")
        print(f"  tag: {doc.get('tag')}")
        if time_field:
            print(f"  {time_field}: {doc.get(time_field)}")
        if value_field:
            print(f"  {value_field}: {doc.get(value_field)}")
        print()

    total_count = collection.count_documents(query)
    print(f'总行数: {total_count}')

    if time_field:
        print('\n正在检查重复时间点...')
        pipeline = [
            {'$match': query},
            {
                '$group': {
                    '_id': f'${time_field}',
                    'count': {'$sum': 1},
                    'sample_values': {'$push': f'${value_field}'} if value_field else {'$push': '$_id'},
                }
            },
            {'$match': {'count': {'$gt': 1}}},
            {'$sort': {'count': -1, '_id': 1}},
            {'$limit': 10},
        ]
        duplicates = list(collection.aggregate(pipeline))
        if duplicates:
            print(f'发现 {len(duplicates)} 个重复时间点（最多展示 10 个）:')
            for item in duplicates:
                print(
                    f"  时间: {item['_id']}, 次数: {item['count']}, "
                    f"样例值: {item['sample_values'][:5]}"
                )
        else:
            print('未发现重复时间点。')

    client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='检查指定设备的 MongoDB 样例数据')
    parser.add_argument('--device', default='a1_b9', help='设备代号，默认 a1_b9')
    parser.add_argument('--month', default='202401', help='月份，格式 YYYYMM，默认 202401')
    parser.add_argument('--data-type', default='ep', help='标签名称，默认 ep')
    parser.add_argument('--limit', type=int, default=10, help='样例行数上限，默认 10')
    parser.add_argument('--collection', default=None, help='显式指定集合名')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    check_actual_data(
        device=args.device,
        month=args.month,
        data_type=args.data_type,
        limit=args.limit,
        collection_name=args.collection,
    )
