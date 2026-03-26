"""
20道题完整验证测试
1. 先查询数据库获取正确答案
2. 让AI回答同样的问题
3. 对比两个答案（改进的数字匹配逻辑）
"""
import os
import sys
import json
import time
import re
from datetime import datetime

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = '*'

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from src.fetcher.data_fetcher import DataFetcher
from src.metadata.metadata_engine import MetadataEngine
from src.compressor.context_compressor import ContextCompressor
from src.agent.orchestrator import create_agent_with_streaming
from src.config import load_config
from src.tools.device_tool import find_device_metadata_with_engine
from langchain_openai import ChatOpenAI
import httpx

# 初始化组件
config = load_config()
metadata_engine = MetadataEngine(config.mysql.connection_string)
mongo_client = MongoClient(config.mongodb.uri)
db = mongo_client[config.mongodb.database_name]

# 初始化 LLM
vllm_base = os.getenv("VLLM_API_BASE") or os.getenv("LLM_BASE_URL")
http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(300.0))
llm = ChatOpenAI(
    model=os.getenv("VLLM_MODEL") or os.getenv("LLM_MODEL", "/models/Qwen3-32B-AWQ"),
    openai_api_base=vllm_base,
    openai_api_key=os.getenv("VLLM_API_KEY") or os.getenv("LLM_API_KEY") or "not-needed",
    temperature=0.7,
    max_tokens=16384,
    http_client=http_client,
    request_timeout=300.0,
)

# 创建 Agent
agent = create_agent_with_streaming(
    llm=llm,
    llm_non_streaming=llm,
    metadata_engine=metadata_engine,
    data_fetcher=DataFetcher(mongo_client, config.mongodb.database_name, 2000),
    cache_manager=None,
    compressor=ContextCompressor(max_tokens=4000)
)

# 20个测试用例
TEST_CASES = [
    # 基础查询（1-5）
    {"id": 1, "query": "a1_b9 设备 2024年1月的电量", "type": "sensor_data", "device": "a1_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31")},
    {"id": 2, "query": "查询 b1_b14 设备 2024年1月的电流数据", "type": "sensor_data", "device": "b1_b14", "data_type": "i", "time_range": ("2024-01-01", "2024-01-31")},
    {"id": 3, "query": "a1_b9 设备 2024年1月5日到1月10日的电量", "type": "sensor_data", "device": "a1_b9", "data_type": "ep", "time_range": ("2024-01-05", "2024-01-10")},
    {"id": 4, "query": "a2_b9 设备 2024年1月的电量数据", "type": "sensor_data", "device": "a2_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31")},
    {"id": 5, "query": "b1_b9 设备 2024年1月的电量", "type": "sensor_data", "device": "b1_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31")},
    
    # 设备搜索（6-8）
    {"id": 6, "query": "搜索包含 b9 的设备", "type": "device_search", "keyword": "b9"},
    {"id": 7, "query": "搜索包含 b14 的设备", "type": "device_search", "keyword": "b14"},
    {"id": 8, "query": "有哪些电梯设备？", "type": "device_search", "keyword": "电梯"},
    
    # 项目查询（9-10）
    {"id": 9, "query": "有哪些项目可用？", "type": "project_list"},
    {"id": 10, "query": "哪个项目的设备最多？", "type": "project_stats"},
    
    # 统计分析（11-15）
    {"id": 11, "query": "a1_b9 设备2024年1月的平均用电量是多少？", "type": "sensor_data", "device": "a1_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "avg"},
    {"id": 12, "query": "a1_b9 设备2024年1月的最大用电量", "type": "sensor_data", "device": "a1_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "max"},
    {"id": 13, "query": "a1_b9 设备2024年1月的最小用电量", "type": "sensor_data", "device": "a1_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "min"},
    {"id": 14, "query": "b1_b14 设备2024年1月的平均电流", "type": "sensor_data", "device": "b1_b14", "data_type": "i", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "avg"},
    {"id": 15, "query": "a2_b9 设备2024年1月的电量统计", "type": "sensor_data", "device": "a2_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "all"},
    
    # 对比查询（16-18）
    {"id": 16, "query": "对比 a1_b9 和 b1_b14 的用电量", "type": "comparison", "devices": ["a1_b9", "b1_b14"], "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31")},
    {"id": 17, "query": "a1_b9 与 a2_b9 哪个用电更多？", "type": "comparison", "devices": ["a1_b9", "a2_b9"], "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31")},
    {"id": 18, "query": "比较 a1_b9、a2_b9、b1_b9 三个设备的用电情况", "type": "comparison", "devices": ["a1_b9", "a2_b9", "b1_b9"], "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31")},
    
    # 复杂查询（19-20）
    {"id": 19, "query": "a1_b9 设备2024年1月的总用电量", "type": "sensor_data", "device": "a1_b9", "data_type": "ep", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "total"},
    {"id": 20, "query": "b1_b14 设备2024年1月电流数据的记录数", "type": "sensor_data", "device": "b1_b14", "data_type": "i", "time_range": ("2024-01-01", "2024-01-31"), "stat_type": "count"}
]

def normalize_number(text):
    """标准化数字：移除千位分隔符"""
    text = str(text).replace(',', '').replace('，', '')
    numbers = re.findall(r'\d+\.?\d*', text)
    return [float(n) for n in numbers]

def number_match(expected, text, tolerance=0.02):
    """检查数字是否匹配（允许2%误差）"""
    if expected is None:
        return False
    
    numbers_in_text = normalize_number(text)
    expected_float = float(expected)
    
    for num in numbers_in_text:
        if abs(num - expected_float) / max(abs(expected_float), 1) <= tolerance:
            return True
        if abs(int(num) - int(expected_float)) <= 1:
            return True
    
    return False

def get_ground_truth(test_case):
    """步骤1: 查询数据库获取正确答案"""
    test_type = test_case["type"]
    
    print(f"\n[步骤1] 查询数据库...")
    
    if test_type == "sensor_data":
        device = test_case.get("device")
        data_type = test_case.get("data_type", "ep")
        time_range = test_case.get("time_range")
        
        if not device or not time_range:
            return None
        
        start_date, end_date = time_range
        collection_name = f"source_data_{data_type}_{start_date[:7].replace('-', '')}"
        
        if collection_name not in db.list_collection_names():
            print(f"  集合 {collection_name} 不存在")
            return None
        
        collection = db[collection_name]
        mongo_query = {
            "device": device,
            "logTime": {"$gte": f"{start_date} 00:00:00", "$lte": f"{end_date} 23:59:59"}
        }
        
        cursor = list(collection.find(mongo_query).sort([("logTime", 1), ("_id", 1)]))
        
        if cursor:
            values = [doc["val"] for doc in cursor if "val" in doc]
            gt = {
                "count": len(cursor),
                "min": round(min(values), 2) if values else None,
                "max": round(max(values), 2) if values else None,
                "avg": round(sum(values) / len(values), 2) if values else None,
                "total": round(max(values) - min(values), 2) if values else None
            }
            
            print(f"  数据库: 记录{gt['count']}条, 最小{gt['min']}, 最大{gt['max']}, 平均{gt['avg']}")
            return gt
    
    elif test_type == "comparison":
        devices = test_case.get("devices", [])
        data_type = test_case.get("data_type", "ep")
        time_range = test_case.get("time_range")
        
        if not devices or not time_range:
            return None
        
        start_date, end_date = time_range
        collection_name = f"source_data_{data_type}_{start_date[:7].replace('-', '')}"
        
        if collection_name not in db.list_collection_names():
            return None
        
        collection = db[collection_name]
        device_stats = {}
        
        for device in devices:
            mongo_query = {
                "device": device,
                "logTime": {"$gte": f"{start_date} 00:00:00", "$lte": f"{end_date} 23:59:59"}
            }
            cursor = list(collection.find(mongo_query))
            
            if cursor:
                values = [doc["val"] for doc in cursor if "val" in doc]
                device_stats[device] = {
                    "count": len(cursor),
                    "total": round(max(values) - min(values), 2) if values else 0
                }
        
        gt = {
            "devices": device_stats,
            "winner": max(device_stats, key=lambda d: device_stats[d].get("total", 0)) if device_stats else None
        }
        
        print(f"  数据库: {len(devices)}个设备对比, 最多用电: {gt['winner']}")
        return gt
    
    elif test_type == "device_search":
        keyword = test_case.get("keyword", "")
        devices = find_device_metadata_with_engine(keyword, metadata_engine)
        device_list = [d for d in devices if isinstance(d, dict) and 'device' in d and 'error' not in d]
        
        gt = {"count": len(device_list)}
        print(f"  数据库: 找到{gt['count']}个设备")
        return gt
    
    elif test_type == "project_list":
        projects = metadata_engine.list_projects()
        gt = {"count": len(projects)}
        print(f"  数据库: {gt['count']}个项目")
        return gt
    
    elif test_type == "project_stats":
        projects = metadata_engine.list_projects()
        max_count = 0
        top_project = None
        
        for proj in projects[:10]:  # 只检查前10个
            try:
                devices = metadata_engine.get_devices_by_project(proj["id"])
                if len(devices) > max_count:
                    max_count = len(devices)
                    top_project = proj["project_name"]
            except:
                pass
        
        gt = {"top_project": top_project, "device_count": max_count}
        print(f"  数据库: 最多设备的项目是 {top_project} ({max_count}个)")
        return gt
    
    return None

def get_ai_answer(query):
    """步骤2: AI回答"""
    print(f"\n[步骤2] AI回答...")
    
    start_time = time.time()
    
    try:
        events = list(agent.run_with_progress(query))
        response_time = round(time.time() - start_time, 2)
        
        ai_response = None
        for event in reversed(events):
            if event.get("type") in ["final_answer", "direct_response"]:
                ai_response = event.get("response", "")
                break
        
        if ai_response:
            print(f"  AI: {ai_response[:150]}...")
            print(f"  耗时: {response_time}s")
            return ai_response, response_time
        else:
            return None, response_time
    
    except Exception as e:
        print(f"  错误: {e}")
        return None, 0

def compare_answers(ground_truth, ai_response, test_case):
    """步骤3: 对比答案（改进的匹配逻辑）"""
    print(f"\n[步骤3] 对比...")
    
    if not ground_truth or not ai_response:
        return {"match": False, "score": 0, "details": []}
    
    test_type = test_case["type"]
    details = []
    score = 0
    
    if test_type == "sensor_data":
        # 检查数据量
        if number_match(ground_truth.get("count"), ai_response):
            details.append(f"✓ 数据量: {ground_truth['count']}")
            score += 25
        else:
            details.append(f"✗ 数据量不匹配")
        
        # 检查统计值
        for key in ["min", "max", "avg", "total"]:
            if key in ground_truth and ground_truth[key]:
                if number_match(ground_truth[key], ai_response):
                    details.append(f"✓ {key}: {ground_truth[key]}")
                    score += 25
                else:
                    details.append(f"✗ {key}不匹配")
    
    elif test_type == "comparison":
        winner = ground_truth.get("winner")
        if winner and winner in ai_response:
            details.append(f"✓ 识别出{winner}用电更多")
            score += 50
        
        devices = ground_truth.get("devices", {})
        for device in devices:
            if device in ai_response:
                score += 25
                details.append(f"✓ 提到{device}")
    
    elif test_type == "device_search":
        if number_match(ground_truth.get("count"), ai_response, tolerance=0.1):
            details.append(f"✓ 设备数量: {ground_truth['count']}")
            score += 100
        else:
            details.append(f"✗ 设备数量不匹配")
    
    elif test_type == "project_list":
        if number_match(ground_truth.get("count"), ai_response):
            details.append(f"✓ 项目数量: {ground_truth['count']}")
            score += 100
        else:
            details.append(f"✗ 项目数量不匹配")
    
    elif test_type == "project_stats":
        top_project = ground_truth.get("top_project")
        if top_project and top_project in ai_response:
            details.append(f"✓ 识别出{top_project}")
            score += 100
        else:
            details.append(f"✗ 未识别出最多设备的项目")
    
    match = score >= 60
    print(f"  得分: {score}分 ({'✓通过' if match else '✗不通过'})")
    for d in details[:3]:
        print(f"    {d}")
    
    return {"match": match, "score": score, "details": details}

def run_test(test_case):
    """运行单个测试"""
    print("\n" + "="*70)
    print(f"测试 #{test_case['id']}: {test_case['query']}")
    print("="*70)
    
    ground_truth = get_ground_truth(test_case)
    ai_response, response_time = get_ai_answer(test_case["query"])
    comparison = compare_answers(ground_truth, ai_response, test_case)
    
    return {
        "id": test_case["id"],
        "query": test_case["query"],
        "ground_truth": ground_truth,
        "ai_response": ai_response[:200] if ai_response else None,
        "response_time": response_time,
        "comparison": comparison
    }

def main():
    """主函数"""
    print("="*70)
    print("20道题完整验证测试")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    for test_case in TEST_CASES:
        result = run_test(test_case)
        results.append(result)
        time.sleep(0.5)
    
    # 统计
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    total = len(results)
    passed = sum(1 for r in results if r["comparison"]["match"])
    avg_score = sum(r["comparison"]["score"] for r in results) / total
    avg_time = sum(r["response_time"] for r in results) / total
    
    print(f"总测试数: {total}")
    print(f"通过数: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"平均得分: {avg_score:.1f}分")
    print(f"平均响应时间: {avg_time:.2f}s")
    
    print("\n详细结果:")
    for r in results:
        status = "✓" if r["comparison"]["match"] else "✗"
        print(f"  {status} 测试{r['id']}: {r['comparison']['score']}分 - {r['query'][:40]}...")
    
    # 保存
    result_file = f"test_20_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {result_file}")
    
    mongo_client.close()

if __name__ == "__main__":
    main()
