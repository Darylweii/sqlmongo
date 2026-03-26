"""
旧版完整查询链路验证脚本（已归档）

用途：直接调用 `fetch_sensor_data_with_components`，快速验证单设备查询、
分页和返回结构是否正常。
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

for key in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"

load_dotenv()

from src.fetcher.data_fetcher import DataFetcher
from src.tools.sensor_tool import fetch_sensor_data_with_components


def _build_fetcher() -> tuple[MongoClient, DataFetcher]:
    mongo_client = MongoClient(
        host=os.getenv("MONGODB_HOST", "127.0.0.1"),
        port=int(os.getenv("MONGODB_PORT", 27017)),
        username=os.getenv("MONGODB_USER") or None,
        password=os.getenv("MONGODB_PASSWORD") or None,
        authSource=os.getenv("MONGODB_AUTH_SOURCE", "admin"),
    )
    data_fetcher = DataFetcher(
        mongo_client=mongo_client,
        database_name=os.getenv("MONGODB_DATABASE", "sensor_db"),
        max_records=2000,
    )
    return mongo_client, data_fetcher


def main() -> None:
    mongo_client, data_fetcher = _build_fetcher()
    try:
        request = {
            "device_codes": ["a1_b9"],
            "start_time": "2024-01-01",
            "end_time": "2024-01-31",
            "data_type": "ep",
            "page": 1,
            "page_size": 50,
        }

        print("=" * 70)
        print("旧版完整查询链路验证 - a1_b9 2024-01 用电情况")
        print("=" * 70)
        print(f"\n查询参数: {request}\n")

        result = fetch_sensor_data_with_components(
            device_codes=request["device_codes"],
            start_time=request["start_time"],
            end_time=request["end_time"],
            data_fetcher=data_fetcher,
            compressor=None,
            data_type=request["data_type"],
            page=request["page"],
            page_size=request["page_size"],
            output_format="json",
            use_aggregation=False,
        )

        if not result.get("success"):
            print(f"❌ 查询失败: {result.get('error')}")
            return

        print("✅ 查询成功")
        print(f"  总数据量: {result.get('total_count')} 条")
        print(f"  当前页: {result.get('page')}/{result.get('total_pages')}")
        print(f"  每页: {result.get('page_size')} 条")
        print(f"  还有更多: {result.get('has_more')}")

        data = json.loads(result.get("data", "[]"))
        print(f"  返回记录数: {len(data)} 条")

        if data:
            print("\n前 3 条数据:")
            for idx, record in enumerate(data[:3], 1):
                print(f"  {idx}. {record.get('logTime')} | device={record.get('device')} | val={record.get('val')}")

            print("\n后 3 条数据:")
            start_idx = max(len(data) - 2, 1)
            for idx, record in enumerate(data[-3:], start_idx):
                print(f"  {idx}. {record.get('logTime')} | device={record.get('device')} | val={record.get('val')}")

        print("\n" + "=" * 70)
        print("分页叉页重复检查")
        print("=" * 70)

        page_samples = {}
        for page_num in [4, 6]:
            page_result = fetch_sensor_data_with_components(
                device_codes=request["device_codes"],
                start_time=request["start_time"],
                end_time=request["end_time"],
                data_fetcher=data_fetcher,
                compressor=None,
                data_type=request["data_type"],
                page=page_num,
                page_size=request["page_size"],
                output_format="json",
                use_aggregation=False,
            )
            if not page_result.get("success"):
                print(f"⚠️ 第 {page_num} 页查询失败: {page_result.get('error')}")
                continue

            page_data = json.loads(page_result.get("data", "[]"))
            page_samples[page_num] = [record.get("_id") for record in page_data]
            print(f"\n第 {page_num} 页: {len(page_data)} 条")
            if page_data:
                print(f"  首条: {page_data[0].get('logTime')} | _id={page_data[0].get('_id')}")
                print(f"  末条: {page_data[-1].get('logTime')} | _id={page_data[-1].get('_id')}")

        if 4 in page_samples and 6 in page_samples:
            duplicates = set(page_samples[4]) & set(page_samples[6])
            if duplicates:
                print(f"\n⚠️ 发现跨页重复记录: {len(duplicates)} 条")
            else:
                print("\n✅ 未发现跨页重复记录")

    finally:
        mongo_client.close()
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
