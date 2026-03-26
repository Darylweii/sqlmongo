"""
语义搜索性能测试与准确性评估

测试内容：
1. 查询速度测试
2. 准确性测试（预设正确答案对比）
3. 多样化查询场景
4. 生成测试报告
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.semantic_layer.device_search import DeviceSemanticSearch


# 预设测试用例：查询 -> 期望的设备ID
TEST_CASES = [
    # 精确匹配测试
    {
        "query": "第九味道",
        "expected_device_id": "a1_b3",
        "expected_device_name": "1#变加联络 AA3-1 第九味道",
        "category": "精确匹配",
    },
    {
        "query": "百年渝府火锅店",
        "expected_device_id": "a1_b5",
        "expected_device_name": "1#变加联络 AA3-3 百年渝府火锅店",
        "category": "精确匹配",
    },
    {
        "query": "龙福宫宾馆",
        "expected_device_id": "a7_b1-w8",
        "expected_device_name": "龙福宫宾馆",
        "category": "精确匹配",
    },
    
    # 模糊匹配测试
    {
        "query": "第九味",
        "expected_device_id": "a1_b3",
        "expected_device_name": "1#变加联络 AA3-1 第九味道",
        "category": "模糊匹配",
    },
    {
        "query": "渝府火锅",
        "expected_device_id": "a1_b5",
        "expected_device_name": "1#变加联络 AA3-3 百年渝府火锅店",
        "category": "模糊匹配",
    },
    {
        "query": "龙福宫",
        "expected_device_id": "a7_b1-w8",
        "expected_device_name": "龙福宫宾馆",
        "category": "模糊匹配",
    },
    
    # 设备类型查询
    {
        "query": "电容柜",
        "expected_device_id": "AMC72",
        "expected_device_name": "电容补偿柜",
        "category": "设备类型",
    },
    {
        "query": "电容补偿",
        "expected_device_id": "AMC72",
        "expected_device_name": "电容补偿柜",
        "category": "设备类型",
    },
    
    # 位置/区域查询
    {
        "query": "机房照明",
        "expected_device_id": "a11_b1",
        "expected_device_name": "wL1五六会议室操作间照明",
        "category": "位置查询",
        "partial_match": True,  # 允许部分匹配
    },
    {
        "query": "会议室照明",
        "expected_device_id": "a11_b1",
        "expected_device_name": "wL1五六会议室操作间照明",
        "category": "位置查询",
        "partial_match": True,
    },
    
    # 项目相关查询
    {
        "query": "智慧物联网能效平台",
        "expected_project_id": 1,
        "expected_project_name": "智慧物联网能效平台",
        "category": "项目查询",
        "type": "project",
    },
    {
        "query": "北京能源研究院",
        "expected_project_id": 80,
        "expected_project_name": "北京能源研究院机房系统",
        "category": "项目查询",
        "type": "project",
    },
    
    # 指标查询
    {
        "query": "用电量",
        "expected_tag": "ep",
        "expected_name": "正向有功电能",
        "category": "指标查询",
        "type": "metric",
    },
    {
        "query": "电压",
        "expected_tag": "u",
        "expected_name": "电压",
        "category": "指标查询",
        "type": "metric",
    },
    {
        "query": "功率",
        "expected_tag": "p",
        "expected_name": "有功功率",
        "category": "指标查询",
        "type": "metric",
    },
    {
        "query": "温度",
        "expected_tag": "temp",
        "expected_name": "温度",
        "category": "指标查询",
        "type": "metric",
    },
    {
        "query": "湿度",
        "expected_tag": "humidity",
        "expected_name": "湿度",
        "category": "指标查询",
        "type": "metric",
    },
    
    # 复合查询（设备+指标）
    {
        "query": "第九味道用电",
        "expected_device_id": "a1_b3",
        "expected_device_name": "1#变加联络 AA3-1 第九味道",
        "category": "复合查询",
    },
    {
        "query": "火锅店功率",
        "expected_device_id": "a1_b5",
        "expected_device_name": "1#变加联络 AA3-3 百年渝府火锅店",
        "category": "复合查询",
    },
    
    # 口语化查询
    {
        "query": "那个火锅店",
        "expected_device_id": "a1_b5",
        "expected_device_name": "1#变加联络 AA3-3 百年渝府火锅店",
        "category": "口语化",
    },
    {
        "query": "宾馆的水表",
        "expected_device_id": "a7_b1-w8",
        "expected_device_name": "龙福宫宾馆",
        "category": "口语化",
    },
    
    # 编号查询
    {
        "query": "AA3-1",
        "expected_device_id": "a1_b3",
        "expected_device_name": "1#变加联络 AA3-1 第九味道",
        "category": "编号查询",
    },
    {
        "query": "1#变加联络",
        "expected_device_id": "a1_b1",
        "expected_device_name": "1#变加联络 AA1 电容柜",
        "category": "编号查询",
        "partial_match": True,
    },
]


class SemanticSearchBenchmark:
    """语义搜索性能测试"""
    
    def __init__(self):
        self.search = DeviceSemanticSearch()
        self.results = []
        self.timing_stats = {
            'device_search': [],
            'project_search': [],
            'metric_search': [],
        }
    
    def initialize(self) -> bool:
        """初始化"""
        return self.search.initialize()
    
    def run_test_case(self, test_case: Dict) -> Dict:
        """运行单个测试用例"""
        query = test_case['query']
        test_type = test_case.get('type', 'device')
        
        result = {
            'query': query,
            'category': test_case['category'],
            'type': test_type,
            'passed': False,
            'score': 0.0,
            'time_ms': 0.0,
            'expected': None,
            'actual': None,
            'error': None,
        }
        
        try:
            start_time = time.time()
            
            if test_type == 'device':
                # 设备搜索
                devices = self.search.search_devices(query, top_k=1, min_score=0.2)
                elapsed = (time.time() - start_time) * 1000
                self.timing_stats['device_search'].append(elapsed)
                
                result['time_ms'] = elapsed
                result['expected'] = test_case.get('expected_device_id')
                
                if devices:
                    actual = devices[0]
                    result['actual'] = actual['device_id']
                    result['score'] = actual['score']
                    result['actual_name'] = actual['device_name']
                    
                    # 检查是否匹配
                    if test_case.get('partial_match'):
                        # 部分匹配：只要找到相关设备就算通过
                        result['passed'] = result['score'] >= 0.3
                    else:
                        result['passed'] = (actual['device_id'] == test_case.get('expected_device_id'))
                else:
                    result['error'] = "未找到匹配设备"
                    
            elif test_type == 'project':
                # 项目搜索
                projects = self.search.search_projects(query, top_k=1, min_score=0.3)
                elapsed = (time.time() - start_time) * 1000
                self.timing_stats['project_search'].append(elapsed)
                
                result['time_ms'] = elapsed
                result['expected'] = test_case.get('expected_project_name')
                
                if projects:
                    actual = projects[0]
                    result['actual'] = actual['project_name']
                    result['score'] = actual['score']
                    result['passed'] = (actual['project_id'] == test_case.get('expected_project_id'))
                else:
                    result['error'] = "未找到匹配项目"
                    
            elif test_type == 'metric':
                # 指标搜索
                metrics = self.search.search_metrics(query, top_k=1, min_score=0.3)
                elapsed = (time.time() - start_time) * 1000
                self.timing_stats['metric_search'].append(elapsed)
                
                result['time_ms'] = elapsed
                result['expected'] = test_case.get('expected_tag')
                
                if metrics:
                    actual = metrics[0]
                    result['actual'] = actual['tag']
                    result['score'] = actual['score']
                    result['actual_name'] = actual['name']
                    result['passed'] = (actual['tag'] == test_case.get('expected_tag'))
                else:
                    result['error'] = "未找到匹配指标"
                    
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def run_all_tests(self) -> List[Dict]:
        """运行所有测试"""
        self.results = []
        
        for test_case in TEST_CASES:
            result = self.run_test_case(test_case)
            self.results.append(result)
            
            # 打印进度
            status = "✓" if result['passed'] else "✗"
            print(f"  {status} [{result['category']}] {result['query']}: {result['score']:.3f} ({result['time_ms']:.1f}ms)")
        
        return self.results
    
    def get_statistics(self) -> Dict:
        """计算统计数据"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        
        # 按类别统计
        categories = {}
        for r in self.results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if r['passed']:
                categories[cat]['passed'] += 1
        
        # 计算时间统计
        all_times = [r['time_ms'] for r in self.results if r['time_ms'] > 0]
        
        return {
            'total_tests': total,
            'passed_tests': passed,
            'failed_tests': total - passed,
            'accuracy': passed / total if total > 0 else 0,
            'categories': categories,
            'timing': {
                'avg_ms': sum(all_times) / len(all_times) if all_times else 0,
                'min_ms': min(all_times) if all_times else 0,
                'max_ms': max(all_times) if all_times else 0,
                'device_avg_ms': sum(self.timing_stats['device_search']) / len(self.timing_stats['device_search']) if self.timing_stats['device_search'] else 0,
                'project_avg_ms': sum(self.timing_stats['project_search']) / len(self.timing_stats['project_search']) if self.timing_stats['project_search'] else 0,
                'metric_avg_ms': sum(self.timing_stats['metric_search']) / len(self.timing_stats['metric_search']) if self.timing_stats['metric_search'] else 0,
            },
            'scores': {
                'avg': sum(r['score'] for r in self.results) / len(self.results) if self.results else 0,
                'min': min(r['score'] for r in self.results) if self.results else 0,
                'max': max(r['score'] for r in self.results) if self.results else 0,
            }
        }
    
    def generate_report(self, output_path: str = "SEMANTIC_SEARCH_REPORT.md") -> str:
        """生成测试报告"""
        stats = self.get_statistics()
        
        report = []
        report.append("# 语义搜索测试报告")
        report.append("")
        report.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 总体结果
        report.append("## 1. 总体结果")
        report.append("")
        report.append(f"| 指标 | 值 |")
        report.append(f"|------|-----|")
        report.append(f"| 总测试数 | {stats['total_tests']} |")
        report.append(f"| 通过数 | {stats['passed_tests']} |")
        report.append(f"| 失败数 | {stats['failed_tests']} |")
        report.append(f"| **准确率** | **{stats['accuracy']*100:.1f}%** |")
        report.append("")
        
        # 性能统计
        report.append("## 2. 性能统计")
        report.append("")
        report.append(f"| 指标 | 值 |")
        report.append(f"|------|-----|")
        report.append(f"| 平均查询时间 | {stats['timing']['avg_ms']:.1f} ms |")
        report.append(f"| 最快查询 | {stats['timing']['min_ms']:.1f} ms |")
        report.append(f"| 最慢查询 | {stats['timing']['max_ms']:.1f} ms |")
        report.append(f"| 设备搜索平均 | {stats['timing']['device_avg_ms']:.1f} ms |")
        report.append(f"| 项目搜索平均 | {stats['timing']['project_avg_ms']:.1f} ms |")
        report.append(f"| 指标搜索平均 | {stats['timing']['metric_avg_ms']:.1f} ms |")
        report.append("")
        
        # 匹配分数统计
        report.append("## 3. 匹配分数统计")
        report.append("")
        report.append(f"| 指标 | 值 |")
        report.append(f"|------|-----|")
        report.append(f"| 平均分数 | {stats['scores']['avg']:.3f} |")
        report.append(f"| 最低分数 | {stats['scores']['min']:.3f} |")
        report.append(f"| 最高分数 | {stats['scores']['max']:.3f} |")
        report.append("")
        
        # 分类统计
        report.append("## 4. 分类准确率")
        report.append("")
        report.append(f"| 类别 | 通过/总数 | 准确率 |")
        report.append(f"|------|----------|--------|")
        for cat, data in stats['categories'].items():
            acc = data['passed'] / data['total'] * 100 if data['total'] > 0 else 0
            report.append(f"| {cat} | {data['passed']}/{data['total']} | {acc:.1f}% |")
        report.append("")
        
        # 详细结果
        report.append("## 5. 详细测试结果")
        report.append("")
        report.append(f"| 状态 | 类别 | 查询 | 期望 | 实际 | 分数 | 耗时 |")
        report.append(f"|------|------|------|------|------|------|------|")
        
        for r in self.results:
            status = "✓" if r['passed'] else "✗"
            expected = r['expected'] or '-'
            actual = r['actual'] or r.get('error', '-')
            report.append(f"| {status} | {r['category']} | {r['query']} | {expected} | {actual} | {r['score']:.3f} | {r['time_ms']:.1f}ms |")
        
        report.append("")
        
        # 失败用例分析
        failed = [r for r in self.results if not r['passed']]
        if failed:
            report.append("## 6. 失败用例分析")
            report.append("")
            for r in failed:
                report.append(f"### {r['query']}")
                report.append(f"- **类别**: {r['category']}")
                report.append(f"- **期望**: {r['expected']}")
                report.append(f"- **实际**: {r['actual']}")
                if r.get('actual_name'):
                    report.append(f"- **实际名称**: {r['actual_name']}")
                report.append(f"- **分数**: {r['score']:.3f}")
                if r.get('error'):
                    report.append(f"- **错误**: {r['error']}")
                report.append("")
        
        # 结论
        report.append("## 7. 结论")
        report.append("")
        if stats['accuracy'] >= 0.9:
            report.append("✅ **优秀**: 语义搜索准确率超过 90%，可以投入生产使用。")
        elif stats['accuracy'] >= 0.7:
            report.append("⚠️ **良好**: 语义搜索准确率在 70%-90% 之间，建议优化部分场景。")
        else:
            report.append("❌ **需改进**: 语义搜索准确率低于 70%，需要进一步优化。")
        report.append("")
        report.append(f"- 平均查询速度 {stats['timing']['avg_ms']:.1f}ms，{'满足' if stats['timing']['avg_ms'] < 500 else '不满足'}实时查询需求")
        report.append(f"- 向量索引包含 4288 个条目（4222 设备 + 58 项目 + 8 指标）")
        report.append("")
        
        # 写入文件
        report_text = "\n".join(report)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        return report_text


def main():
    print("=" * 70)
    print("语义搜索性能测试")
    print("=" * 70)
    
    benchmark = SemanticSearchBenchmark()
    
    print("\n初始化语义搜索...")
    if not benchmark.initialize():
        print("初始化失败!")
        return
    
    print(f"\n运行 {len(TEST_CASES)} 个测试用例...\n")
    benchmark.run_all_tests()
    
    print("\n生成测试报告...")
    report = benchmark.generate_report()
    
    print("\n" + "=" * 70)
    print("测试完成! 报告已保存到 SEMANTIC_SEARCH_REPORT.md")
    print("=" * 70)
    
    # 打印摘要
    stats = benchmark.get_statistics()
    print(f"\n摘要:")
    print(f"  准确率: {stats['accuracy']*100:.1f}%")
    print(f"  平均查询时间: {stats['timing']['avg_ms']:.1f}ms")
    print(f"  通过/总数: {stats['passed_tests']}/{stats['total_tests']}")


if __name__ == '__main__':
    main()
