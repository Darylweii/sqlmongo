"""
智能设备过滤器 - 用于对比查询的设备消歧义

实现基于置信度的多策略设备过滤：
1. 共同项目过滤（置信度 0.95）
2. 精确名称匹配（置信度 0.90）
3. 排除辅助设备（置信度 0.75）
4. 限制数量（置信度 0.50）
"""

from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SmartDeviceFilter:
    """智能设备过滤器 - 自动消歧义，无需用户交互"""

    # 辅助设备关键词
    AUXILIARY_KEYWORDS = ['照明', '插座', '空调', '风机', '水泵', '备用', '应急']

    @staticmethod
    def filter_comparison_devices(
        devices_by_target: Dict[str, List[Dict[str, Any]]],
        context: Dict[str, Any] = None
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        智能过滤对比查询中的设备

        Args:
            devices_by_target: {
                "第九味道": [device1, device2, ...],
                "火锅店": [device3, device4, ...]
            }
            context: 额外上下文信息

        Returns:
            (filtered_devices, filter_info)
            - filtered_devices: 过滤后的设备字典
            - filter_info: {
                'strategy': str,      # 使用的策略
                'confidence': float,  # 置信度 0-1
                'reason': str,        # 过滤原因
                'original_count': int,
                'filtered_count': int
            }
        """
        if not devices_by_target:
            return {}, {
                'strategy': 'empty',
                'confidence': 0.0,
                'reason': '没有设备需要过滤',
                'original_count': 0,
                'filtered_count': 0
            }

        original_count = sum(len(devs) for devs in devices_by_target.values())

        # 策略1: 共同项目过滤（置信度 0.95）
        result = SmartDeviceFilter._filter_by_common_project(devices_by_target)
        if result['confidence'] >= 0.85:
            logger.info(f"使用共同项目过滤策略: {result['reason']}")
            result['original_count'] = original_count
            result['filtered_count'] = sum(len(devs) for devs in result['devices'].values())
            return result['devices'], result

        # 策略2: 精确名称匹配（置信度 0.90）
        result = SmartDeviceFilter._filter_by_exact_name(devices_by_target)
        if result['confidence'] >= 0.85:
            logger.info(f"使用精确名称匹配策略: {result['reason']}")
            result['original_count'] = original_count
            result['filtered_count'] = sum(len(devs) for devs in result['devices'].values())
            return result['devices'], result

        # 策略3: 排除辅助设备（置信度 0.75）
        result = SmartDeviceFilter._filter_main_devices(devices_by_target)
        if result['confidence'] >= 0.60:
            logger.info(f"使用主设备过滤策略: {result['reason']}")
            result['original_count'] = original_count
            result['filtered_count'] = sum(len(devs) for devs in result['devices'].values())
            return result['devices'], result

        # 策略4: 限制数量（置信度 0.50）
        result = SmartDeviceFilter._limit_device_count(devices_by_target)
        logger.info(f"使用数量限制策略: {result['reason']}")
        result['original_count'] = original_count
        result['filtered_count'] = sum(len(devs) for devs in result['devices'].values())
        return result['devices'], result

    @staticmethod
    def _filter_by_common_project(
        devices_by_target: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        策略1: 根据共同项目过滤
        
        如果所有对比目标都在同一个项目中有设备，自动过滤到该项目
        """
        # 按项目分组所有设备
        projects_by_target = {}
        for target, devices in devices_by_target.items():
            projects = {}
            for dev in devices:
                project_id = dev.get('project_id')
                project_name = dev.get('project_name', '未知项目')
                key = f"{project_id}:{project_name}"
                
                if key not in projects:
                    projects[key] = []
                projects[key].append(dev)
            
            projects_by_target[target] = projects
        
        # 查找共同项目
        common_projects = SmartDeviceFilter._find_common_projects(projects_by_target)
        
        if len(common_projects) == 0:
            # 没有共同项目
            return {
                'devices': devices_by_target,
                'strategy': 'no_common_project',
                'confidence': 0.0,
                'reason': '对比目标分散在不同项目中'
            }
        
        elif len(common_projects) == 1:
            # 只有一个共同项目（最理想）
            project_key = common_projects[0]
            filtered = SmartDeviceFilter._filter_by_project(
                devices_by_target, projects_by_target, project_key
            )
            
            project_name = project_key.split(':')[1]
            return {
                'devices': filtered,
                'strategy': 'single_common_project',
                'confidence': 0.95,
                'reason': f'自动过滤到共同项目: {project_name}'
            }
        
        else:
            # 多个共同项目，选择设备最多的
            best_project = max(
                common_projects,
                key=lambda p: SmartDeviceFilter._count_devices_in_project(
                    projects_by_target, p
                )
            )
            
            filtered = SmartDeviceFilter._filter_by_project(
                devices_by_target, projects_by_target, best_project
            )
            
            project_name = best_project.split(':')[1]
            return {
                'devices': filtered,
                'strategy': 'best_common_project',
                'confidence': 0.85,
                'reason': f'自动过滤到设备最多的项目: {project_name}'
            }
    
    @staticmethod
    def _find_common_projects(
        projects_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> List[str]:
        """查找所有目标的共同项目"""
        if not projects_by_target:
            return []
        
        # 获取第一个目标的项目集合
        first_target = list(projects_by_target.keys())[0]
        common = set(projects_by_target[first_target].keys())
        
        # 与其他目标的项目求交集
        for target, projects in projects_by_target.items():
            if target != first_target:
                common &= set(projects.keys())
        
        return list(common)
    
    @staticmethod
    def _filter_by_project(
        devices_by_target: Dict[str, List[Dict[str, Any]]],
        projects_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]],
        project_key: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """按项目过滤设备"""
        filtered = {}
        for target in devices_by_target.keys():
            filtered[target] = projects_by_target[target].get(project_key, [])
        return filtered
    
    @staticmethod
    def _count_devices_in_project(
        projects_by_target: Dict[str, Dict[str, List[Dict[str, Any]]]],
        project_key: str
    ) -> int:
        """统计某个项目中的设备总数"""
        count = 0
        for target, projects in projects_by_target.items():
            count += len(projects.get(project_key, []))
        return count

    @staticmethod
    def _filter_by_exact_name(
        devices_by_target: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        策略2: 精确名称匹配
        
        如果目标名称在设备名称中精确出现，只保留这些设备
        """
        filtered = {}
        has_exact_match = False
        
        for target, devices in devices_by_target.items():
            exact_matches = []
            for dev in devices:
                device_name = dev.get('name', '').lower()
                if target.lower() in device_name:
                    exact_matches.append(dev)
            
            if exact_matches and len(exact_matches) <= 3:
                filtered[target] = exact_matches
                has_exact_match = True
            else:
                filtered[target] = devices
        
        if has_exact_match:
            return {
                'devices': filtered,
                'strategy': 'exact_name_match',
                'confidence': 0.90,
                'reason': '设备名称精确匹配目标关键词'
            }
        else:
            return {
                'devices': devices_by_target,
                'strategy': 'no_exact_match',
                'confidence': 0.0,
                'reason': '未找到精确名称匹配'
            }
    
    @staticmethod
    def _filter_main_devices(
        devices_by_target: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        策略3: 过滤主设备，排除辅助设备
        
        排除包含辅助关键词的设备（照明、插座、空调等）
        """
        filtered = {}
        has_filtered = False
        
        for target, devices in devices_by_target.items():
            main_devices = []
            for dev in devices:
                device_name = dev.get('name', '').lower()
                is_auxiliary = any(
                    kw in device_name 
                    for kw in SmartDeviceFilter.AUXILIARY_KEYWORDS
                )
                if not is_auxiliary:
                    main_devices.append(dev)
            
            if main_devices and len(main_devices) < len(devices):
                filtered[target] = main_devices
                has_filtered = True
            else:
                filtered[target] = devices
        
        if has_filtered:
            original_count = sum(len(devs) for devs in devices_by_target.values())
            filtered_count = sum(len(devs) for devs in filtered.values())
            return {
                'devices': filtered,
                'strategy': 'main_devices_only',
                'confidence': 0.75,
                'reason': f'自动排除辅助设备，保留 {filtered_count}/{original_count} 个主设备'
            }
        else:
            return {
                'devices': devices_by_target,
                'strategy': 'no_auxiliary_devices',
                'confidence': 0.0,
                'reason': '未检测到辅助设备'
            }
    
    @staticmethod
    def _limit_device_count(
        devices_by_target: Dict[str, List[Dict[str, Any]]],
        max_per_target: int = 5
    ) -> Dict[str, Any]:
        """
        策略4: 限制设备数量
        
        如果某个目标的设备数量过多，只保留前N个
        """
        filtered = {}
        has_limited = False
        
        for target, devices in devices_by_target.items():
            if len(devices) > max_per_target:
                # 按设备代号排序，保持一致性
                sorted_devices = sorted(devices, key=lambda d: d.get('device', ''))
                filtered[target] = sorted_devices[:max_per_target]
                has_limited = True
            else:
                filtered[target] = devices
        
        if has_limited:
            original_count = sum(len(devs) for devs in devices_by_target.values())
            filtered_count = sum(len(devs) for devs in filtered.values())
            return {
                'devices': filtered,
                'strategy': 'limit_count',
                'confidence': 0.50,
                'reason': f'设备数量过多，自动限制为每个目标最多 {max_per_target} 个设备（{filtered_count}/{original_count}）'
            }
        else:
            return {
                'devices': devices_by_target,
                'strategy': 'no_limit_needed',
                'confidence': 0.50,
                'reason': f'设备数量适中，无需限制'
            }
