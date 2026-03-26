"""
测试后端分页是否有重复数据
直接调用后端 API 测试
"""
import requests
import json
from collections import defaultdict

API_BASE = "http://localhost:8080"

def test_pagination():
    """测试分页是否有重复"""
    
    # 查询参数
    params = {
        "device_codes": ["a1_b9"],
        "start_time": "2024-01-01",
        "end_time": "2024-01-31",
        "data_type": "ep",
        "page_size": 50
    }
    
    print("=" * 80)
    print("测试后端分页 - 检查是否有重复数据")
    print("=" * 80)
    
    # 存储所有数据的唯一标识
    seen_records = set()
    all_data = []
    duplicates = []
    
    # 获取前15页数据
    for page in range(1, 16):
        print(f"\n请求第 {page} 页...")
        
        response = requests.post(
            f"{API_BASE}/api/query",
            json={**params, "page": page}
        )
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            continue
        
        data = response.json()
        
        if not data.get("success"):
            print(f"❌ 查询失败: {data.get('error')}")
            continue
        
        records = data.get("data", [])
        print(f"✓ 返回 {len(records)} 条记录")
        
        if records:
            first = records[0]
            last = records[-1]
            print(f"  第一条: {first.get('time')} - {first.get('value')}")
            print(f"  最后一条: {last.get('time')} - {last.get('value')}")
        
        # 检查重复
        for record in records:
            # 使用 (time, device, tag, value) 作为唯一标识
            key = (
                record.get("time"),
                record.get("device"),
                record.get("tag"),
                record.get("value")
            )
            
            if key in seen_records:
                duplicates.append({
                    "page": page,
                    "record": record
                })
                print(f"  ⚠️  发现重复: {record.get('time')} - {record.get('value')}")
            else:
                seen_records.add(key)
                all_data.append(record)
    
    # 输出结果
    print("\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"总记录数: {len(all_data)}")
    print(f"重复记录数: {len(duplicates)}")
    
    if duplicates:
        print("\n❌ 发现重复数据:")
        for dup in duplicates[:10]:  # 只显示前10个
            print(f"  页码 {dup['page']}: {dup['record'].get('time')} - {dup['record'].get('value')}")
    else:
        print("\n✅ 没有发现重复数据，分页正常！")
    
    # 检查数据连续性
    print("\n" + "=" * 80)
    print("检查数据连续性")
    print("=" * 80)
    
    if len(all_data) >= 2:
        # 检查时间是否递增
        times = [r.get("time") for r in all_data if r.get("time")]
        is_sorted = all(times[i] <= times[i+1] for i in range(len(times)-1))
        
        if is_sorted:
            print("✅ 数据按时间正确排序")
        else:
            print("❌ 数据排序有问题")
            
            # 找出乱序的位置
            for i in range(len(times)-1):
                if times[i] > times[i+1]:
                    print(f"  位置 {i}: {times[i]} > {times[i+1]}")

if __name__ == "__main__":
    try:
        test_pagination()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
