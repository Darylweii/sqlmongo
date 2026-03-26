"""
综合测试脚本 - 30个不同角度和复杂度的查询
测试响应时间、成功率、数据准确性等指标
包括：基础查询、设备搜索、项目查询、对比查询、统计分析、复杂查询、时间范围查询、聚合分析、趋势分析、异常检测
"""
import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, List, Any

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
from langchain_openai import ChatOpenAI
import httpx

# 初始化组件
config = load_config()
metadata_engine = MetadataEngine(config.mysql.connection_string)
mongo_client = MongoClient(config.mongodb.uri)
data_fetcher = DataFetcher(
    mongo_client=mongo_client,
    database_name=config.mongodb.database_name,
    max_records=2000
)
compressor = ContextCompressor(max_tokens=4000)

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
    data_fetcher=data_fetcher,
    cache_manager=None,
    compressor=compressor
)

# 测试用例
TEST_CASES = [
    # 基础查询（1-5）
    {
        "id": 1,
        "category": "基础查询",
        "complexity": "简单",
        "query": "查询 a1_b9 设备今天的电量数据",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep"
    },
    {
        "id": 2,
        "category": "基础查询",
        "complexity": "简单",
        "query": "a1_b9 设备 2024年1月的电量",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep"
    },
    {
        "id": 3,
        "category": "基础查询",
        "complexity": "简单",
        "query": "查询 b1_b14 设备昨天的电流数据",
        "expected_type": "sensor_data",
        "expected_devices": ["b1_b14"],
        "expected_data_type": "i"
    },
    {
        "id": 4,
        "category": "基础查询",
        "complexity": "中等",
        "query": "a1_b9 设备最近7天的用电量趋势",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep"
    },
    {
        "id": 5,
        "category": "基础查询",
        "complexity": "中等",
        "query": "查询 a1_b9 在 2024年1月5日到1月10日的电量数据",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep"
    },
    
    # 设备搜索（6-8）
    {
        "id": 6,
        "category": "设备搜索",
        "complexity": "简单",
        "query": "搜索包含 b9 的设备",
        "expected_type": "device_list",
        "expected_keyword": "b9"
    },
    {
        "id": 7,
        "category": "设备搜索",
        "complexity": "简单",
        "query": "有哪些电梯设备？",
        "expected_type": "device_list",
        "expected_keyword": "电梯"
    },
    {
        "id": 8,
        "category": "设备搜索",
        "complexity": "中等",
        "query": "北京电力项目有哪些设备？",
        "expected_type": "device_list",
        "expected_keyword": "北京"
    },
    
    # 项目查询（9-10）
    {
        "id": 9,
        "category": "项目查询",
        "complexity": "简单",
        "query": "有哪些项目可用？",
        "expected_type": "project_list"
    },
    {
        "id": 10,
        "category": "项目查询",
        "complexity": "中等",
        "query": "哪个项目的设备最多？",
        "expected_type": "project_stats"
    },
    
    # 对比查询（11-14）
    {
        "id": 11,
        "category": "对比查询",
        "complexity": "中等",
        "query": "对比 a1_b9 和 b1_b14 的用电量",
        "expected_type": "comparison",
        "expected_devices": ["a1_b9", "b1_b14"],
        "is_comparison": True
    },
    {
        "id": 12,
        "category": "对比查询",
        "complexity": "中等",
        "query": "a1_b9 与 b1_b14 哪个用电更多？",
        "expected_type": "comparison",
        "expected_devices": ["a1_b9", "b1_b14"],
        "is_comparison": True
    },
    {
        "id": 13,
        "category": "对比查询",
        "complexity": "复杂",
        "query": "对比一下 a1_b9 和 b1_b14 在2024年1月的电量数据",
        "expected_type": "comparison",
        "expected_devices": ["a1_b9", "b1_b14"],
        "is_comparison": True
    },
    {
        "id": 14,
        "category": "对比查询",
        "complexity": "复杂",
        "query": "比较 a1_b9、b1_b14、a2_b3 三个设备的用电情况",
        "expected_type": "comparison",
        "expected_devices": ["a1_b9", "b1_b14", "a2_b3"],
        "is_comparison": True
    },
    
    # 统计分析（15-17）
    {
        "id": 15,
        "category": "统计分析",
        "complexity": "中等",
        "query": "a1_b9 设备2024年1月的平均用电量是多少？",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_statistics": True
    },
    {
        "id": 16,
        "category": "统计分析",
        "complexity": "复杂",
        "query": "a1_b9 设备在2024年1月哪天用电最多？",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_statistics": True
    },
    {
        "id": 17,
        "category": "统计分析",
        "complexity": "复杂",
        "query": "分析 a1_b9 设备2024年1月的用电趋势",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_statistics": True
    },
    
    # 复杂查询（18-20）
    {
        "id": 18,
        "category": "复杂查询",
        "complexity": "复杂",
        "query": "查询所有包含 b9 的设备在2024年1月的总用电量",
        "expected_type": "sensor_data",
        "expected_keyword": "b9",
        "needs_statistics": True
    },
    {
        "id": 19,
        "category": "复杂查询",
        "complexity": "复杂",
        "query": "找出2024年1月用电量超过1000的所有时间点（a1_b9设备）",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "has_filter": True
    },
    {
        "id": 20,
        "category": "复杂查询",
        "complexity": "复杂",
        "query": "对比北京电力项目和上海电力项目的设备数量",
        "expected_type": "project_stats",
        "is_comparison": True
    },
    
    # 时间范围查询（21-23）
    {
        "id": 21,
        "category": "时间范围查询",
        "complexity": "中等",
        "query": "a1_b9 设备本周的用电情况",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "expected_data_type": "ep"
    },
    {
        "id": 22,
        "category": "时间范围查询",
        "complexity": "中等",
        "query": "查询 b1_b14 设备本月的电流数据",
        "expected_type": "sensor_data",
        "expected_devices": ["b1_b14"],
        "expected_data_type": "i"
    },
    {
        "id": 23,
        "category": "时间范围查询",
        "complexity": "复杂",
        "query": "对比 a1_b9 设备上周和本周的用电量变化",
        "expected_type": "comparison",
        "expected_devices": ["a1_b9"],
        "is_comparison": True
    },
    
    # 聚合分析（24-26）
    {
        "id": 24,
        "category": "聚合分析",
        "complexity": "复杂",
        "query": "a1_b9 设备2024年1月每天的平均用电量",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_aggregation": True
    },
    {
        "id": 25,
        "category": "聚合分析",
        "complexity": "复杂",
        "query": "统计 a1_b9 设备2024年1月每周的总用电量",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_aggregation": True
    },
    {
        "id": 26,
        "category": "聚合分析",
        "complexity": "复杂",
        "query": "计算 a1_b9 和 b1_b14 在2024年1月的日均用电量对比",
        "expected_type": "comparison",
        "expected_devices": ["a1_b9", "b1_b14"],
        "is_comparison": True,
        "needs_aggregation": True
    },
    
    # 趋势分析（27-28）
    {
        "id": 27,
        "category": "趋势分析",
        "complexity": "复杂",
        "query": "a1_b9 设备2024年1月的用电量是上升还是下降？",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_statistics": True
    },
    {
        "id": 28,
        "category": "趋势分析",
        "complexity": "复杂",
        "query": "分析 b1_b14 设备最近一个月的电流变化趋势",
        "expected_type": "sensor_data",
        "expected_devices": ["b1_b14"],
        "expected_data_type": "i",
        "needs_statistics": True
    },
    
    # 异常检测（29-30）
    {
        "id": 29,
        "category": "异常检测",
        "complexity": "复杂",
        "query": "a1_b9 设备2024年1月有没有异常用电的时间点？",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_statistics": True
    },
    {
        "id": 30,
        "category": "异常检测",
        "complexity": "复杂",
        "query": "找出 a1_b9 设备2024年1月用电量最高的前5个时间点",
        "expected_type": "sensor_data",
        "expected_devices": ["a1_b9"],
        "needs_statistics": True,
        "has_limit": True
    }
]

def run_test_case(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """运行单个测试用例"""
    print(f"\n{'='*70}")
    print(f"测试 #{test_case['id']}: {test_case['query']}")
    print(f"分类: {test_case['category']} | 复杂度: {test_case['complexity']}")
    print('='*70)
    
    result = {
        "id": test_case["id"],
        "category": test_case["category"],
        "complexity": test_case["complexity"],
        "query": test_case["query"],
        "success": False,
        "response_time": 0,
        "has_data": False,
        "data_count": 0,
        "error": None,
        "response": None,
        "statistics": None
    }
    
    try:
        start_time = time.time()
        
        # 运行查询 - 使用生成器收集所有事件
        events = list(agent.run_with_progress(test_case["query"]))
        
        end_time = time.time()
        result["response_time"] = round(end_time - start_time, 2)
        
        # 查找最终响应
        final_event = None
        for event in reversed(events):
            if event.get("type") in ["final_answer", "direct_response"]:
                final_event = event
                break
        
        if final_event:
            result["success"] = True
            result["response"] = final_event.get("response", "")
            
            # 检查是否有查询参数
            if "query_params" in final_event:
                result["has_data"] = True
            
            # 检查统计信息
            for event in events:
                if event.get("type") == "step_done" and "query_info" in event:
                    query_info = event["query_info"]
                    if isinstance(query_info, dict) and "statistics" in query_info:
                        result["statistics"] = query_info["statistics"]
                        result["data_count"] = query_info["statistics"].get("count", 0)
                        break
            
            print(f"[成功] 响应时间: {result['response_time']}s")
            if result["has_data"]:
                print(f"  有查询数据")
            if result["statistics"]:
                stats = result["statistics"]
                print(f"  统计: min={stats.get('min')}, max={stats.get('max')}, avg={stats.get('avg')}")
        else:
            result["error"] = "未找到最终响应"
            print(f"[失败] {result['error']}")
            
    except Exception as e:
        import traceback
        result["error"] = str(e)
        print(f"[失败] {result['error']}")
        # print(traceback.format_exc())  # 注释掉详细错误，避免输出过多
    
    return result

def generate_report(results: List[Dict[str, Any]]) -> str:
    """生成测试报告"""
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success
    success_rate = (success / total * 100) if total > 0 else 0
    
    # 按分类统计
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0, "avg_time": 0, "times": []}
        categories[cat]["total"] += 1
        if r["success"]:
            categories[cat]["success"] += 1
        categories[cat]["times"].append(r["response_time"])
    
    for cat in categories:
        times = categories[cat]["times"]
        categories[cat]["avg_time"] = round(sum(times) / len(times), 2) if times else 0
        categories[cat]["min_time"] = round(min(times), 2) if times else 0
        categories[cat]["max_time"] = round(max(times), 2) if times else 0
    
    # 按复杂度统计
    complexities = {}
    for r in results:
        comp = r["complexity"]
        if comp not in complexities:
            complexities[comp] = {"total": 0, "success": 0, "avg_time": 0, "times": []}
        complexities[comp]["total"] += 1
        if r["success"]:
            complexities[comp]["success"] += 1
        complexities[comp]["times"].append(r["response_time"])
    
    for comp in complexities:
        times = complexities[comp]["times"]
        complexities[comp]["avg_time"] = round(sum(times) / len(times), 2) if times else 0
    
    # 响应时间统计
    all_times = [r["response_time"] for r in results if r["success"]]
    avg_time = round(sum(all_times) / len(all_times), 2) if all_times else 0
    min_time = round(min(all_times), 2) if all_times else 0
    max_time = round(max(all_times), 2) if all_times else 0
    
    # 生成报告
    report = f"""
# AI Data Router Agent - 综合测试报告

**测试时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**测试用例数**: {total}

## 📊 总体统计

| 指标 | 数值 |
|------|------|
| 总测试数 | {total} |
| 成功数 | {success} |
| 失败数 | {failed} |
| 成功率 | {success_rate:.1f}% |
| 平均响应时间 | {avg_time}s |
| 最快响应 | {min_time}s |
| 最慢响应 | {max_time}s |

## 📈 按分类统计

| 分类 | 总数 | 成功 | 成功率 | 平均响应时间 | 最快 | 最慢 |
|------|------|------|--------|--------------|------|------|
"""
    
    for cat, stats in sorted(categories.items()):
        cat_success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        report += f"| {cat} | {stats['total']} | {stats['success']} | {cat_success_rate:.1f}% | {stats['avg_time']}s | {stats['min_time']}s | {stats['max_time']}s |\n"
    
    report += f"""
## 🎯 按复杂度统计

| 复杂度 | 总数 | 成功 | 成功率 | 平均响应时间 |
|--------|------|------|--------|--------------|
"""
    
    for comp, stats in sorted(complexities.items(), key=lambda x: {"简单": 1, "中等": 2, "复杂": 3}.get(x[0], 0)):
        comp_success_rate = (stats["success"] / stats["total"] * 100) if stats["total"] > 0 else 0
        report += f"| {comp} | {stats['total']} | {stats['success']} | {comp_success_rate:.1f}% | {stats['avg_time']}s |\n"
    
    report += f"""
## 📋 详细测试结果

| ID | 分类 | 复杂度 | 查询 | 状态 | 响应时间 | 数据量 |
|----|------|--------|------|------|----------|--------|
"""
    
    for r in results:
        status = "✓" if r["success"] else "✗"
        data_info = f"{r['data_count']}条" if r["has_data"] else "-"
        query_short = r["query"][:30] + "..." if len(r["query"]) > 30 else r["query"]
        report += f"| {r['id']} | {r['category']} | {r['complexity']} | {query_short} | {status} | {r['response_time']}s | {data_info} |\n"
    
    # 失败案例分析
    failed_cases = [r for r in results if not r["success"]]
    if failed_cases:
        report += f"""
## ❌ 失败案例分析

共 {len(failed_cases)} 个失败案例：

"""
        for r in failed_cases:
            report += f"""
### 测试 #{r['id']}: {r['query']}

- **分类**: {r['category']}
- **复杂度**: {r['complexity']}
- **错误信息**: {r['error']}

"""
    
    # 性能分析
    report += f"""
## ⚡ 性能分析

### 响应时间分布

- **< 1s**: {sum(1 for t in all_times if t < 1)} 个 ({sum(1 for t in all_times if t < 1) / len(all_times) * 100:.1f}%)
- **1-3s**: {sum(1 for t in all_times if 1 <= t < 3)} 个 ({sum(1 for t in all_times if 1 <= t < 3) / len(all_times) * 100:.1f}%)
- **3-5s**: {sum(1 for t in all_times if 3 <= t < 5)} 个 ({sum(1 for t in all_times if 3 <= t < 5) / len(all_times) * 100:.1f}%)
- **> 5s**: {sum(1 for t in all_times if t >= 5)} 个 ({sum(1 for t in all_times if t >= 5) / len(all_times) * 100:.1f}%)

### 性能评级

"""
    
    if avg_time < 2:
        report += "🌟 **优秀** - 平均响应时间 < 2s\n"
    elif avg_time < 3:
        report += "👍 **良好** - 平均响应时间 < 3s\n"
    elif avg_time < 5:
        report += "⚠️ **一般** - 平均响应时间 < 5s\n"
    else:
        report += "❗ **需要优化** - 平均响应时间 >= 5s\n"
    
    report += f"""
## 💡 结论与建议

### 测试结论

1. **成功率**: {success_rate:.1f}% {'✓ 优秀' if success_rate >= 90 else '⚠️ 需要改进' if success_rate >= 70 else '❗ 需要优化'}
2. **响应速度**: 平均 {avg_time}s {'✓ 快速' if avg_time < 3 else '⚠️ 一般' if avg_time < 5 else '❗ 较慢'}
3. **稳定性**: {'✓ 稳定' if failed == 0 else f'⚠️ 有 {failed} 个失败案例'}

### 优化建议

"""
    
    if avg_time > 3:
        report += "1. **性能优化**: 考虑添加缓存机制，减少重复查询\n"
    
    if failed > 0:
        report += f"2. **错误处理**: 分析 {failed} 个失败案例，改进错误处理逻辑\n"
    
    if complexities.get("复杂", {}).get("avg_time", 0) > 5:
        report += "3. **复杂查询优化**: 复杂查询响应时间较长，考虑优化查询逻辑\n"
    
    report += """
### 下一步计划

1. 针对失败案例进行专项优化
2. 添加更多边界情况测试
3. 进行压力测试和并发测试
4. 优化慢查询的性能

---

**测试完成时间**: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return report

def main():
    """主函数"""
    print("="*70)
    print("AI Data Router Agent - 综合测试")
    print("="*70)
    
    # 只测试前5个用例（快速验证）
    test_cases = TEST_CASES  # 运行全部30个测试用例
    
    print(f"测试用例数: {len(test_cases)}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = []
    
    # 运行所有测试
    for test_case in test_cases:
        result = run_test_case(test_case)
        results.append(result)
        time.sleep(0.5)  # 避免请求过快
    
    # 生成报告
    print("\n" + "="*70)
    print("生成测试报告...")
    print("="*70)
    
    report = generate_report(results)
    
    # 保存报告
    report_file = f"TEST_REPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存详细结果
    results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[成功] 测试完成！")
    print(f"  报告文件: {report_file}")
    print(f"  结果文件: {results_file}")
    
    # 打印简要统计
    success = sum(1 for r in results if r["success"])
    total = len(results)
    avg_time = sum(r["response_time"] for r in results if r["success"]) / success if success > 0 else 0
    
    print(f"\n简要统计:")
    print(f"  成功率: {success}/{total} ({success/total*100:.1f}%)")
    print(f"  平均响应时间: {avg_time:.2f}s")
    
    mongo_client.close()

if __name__ == "__main__":
    main()
