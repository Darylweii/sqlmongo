"""从 MySQL 元数据和语义配置构建 semantic_entries.json。"""

from __future__ import annotations

import argparse
import json
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

import pymysql
import yaml
from dotenv import load_dotenv


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CN_AIR_CONDITIONER = '空调'
CN_ELEC_METER = '电表'
CN_ENV_SENSOR = '环境传感器'
CN_GAS_METER = '燃气表'
CN_WATER_METER = '水表'
CN_PROJECT = '项目'
CONFIG_DESCRIPTION = '语义层配置 - 设备与项目元数据'

DEFAULT_DEVICE_TYPES = {
    'air_conditioner': CN_AIR_CONDITIONER,
    'elec_meter': CN_ELEC_METER,
    'environmental_sensor': CN_ENV_SENSOR,
    'gas_meter': CN_GAS_METER,
    'water_meter': CN_WATER_METER,
}


def load_semantic_config(config_path: str = 'config/semantic_layer.yaml') -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {
            'description': CONFIG_DESCRIPTION,
            'version': '1.0',
            'device_types': DEFAULT_DEVICE_TYPES,
            'metric_types': {},
            'mongodb': {'database': os.getenv('MONGODB_DATABASE', 'sensor_db'), 'collection_pattern': 'source_data_{tag}_{YYYYMM}'},
            'mysql': {
                'device_database': 'device',
                'device_table': 'device_info',
                'project_database': 'project',
                'project_table': 'project_info',
            },
            'projects': {},
        }
    return yaml.safe_load(path.read_text(encoding='utf-8')) or {}


def get_mysql_connection(database: str | None = None):
    load_dotenv()
    return pymysql.connect(
        host=os.getenv('MYSQL_HOST', '127.0.0.1'),
        port=int(os.getenv('MYSQL_PORT', 3306)),
        user=os.getenv('MYSQL_USER', 'root'),
        password=os.getenv('MYSQL_PASSWORD', ''),
        database=database or os.getenv('MYSQL_DATABASE') or None,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )


def export_projects(config: dict[str, Any], include_disabled: bool = False) -> list[dict[str, Any]]:
    mysql_cfg = config.get('mysql', {})
    project_db = mysql_cfg.get('project_database', 'project')
    project_table = mysql_cfg.get('project_table', 'project_info')
    where_clause = '' if include_disabled else 'WHERE enable = 1'
    sql = f"""
        SELECT id, project_name, project_code_name, enable
        FROM {project_db}.{project_table}
        {where_clause}
        ORDER BY id ASC
    """
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
    return [row for row in rows if row.get('project_name')]


def export_devices(config: dict[str, Any]) -> list[dict[str, Any]]:
    mysql_cfg = config.get('mysql', {})
    device_db = mysql_cfg.get('device_database', 'device')
    device_table = mysql_cfg.get('device_table', 'device_info')
    sql = f"""
        SELECT id, device, device_name, device_type, project_id, asset_number, tg
        FROM {device_db}.{device_table}
        ORDER BY id ASC
    """
    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
    return [row for row in rows if row.get('device') and row.get('device_name')]


def infer_device_type(raw_device_type: str | None, device_name: str | None) -> str:
    text = ' '.join(filter(None, [str(raw_device_type or ''), str(device_name or '')])).lower()
    if any(keyword in text for keyword in ('水表', 'water')):
        return 'water_meter'
    if any(keyword in text for keyword in ('燃气', '天然气', 'gas')):
        return 'gas_meter'
    if any(keyword in text for keyword in ('温度', '湿度', '环境', '传感器', 'sensor')):
        return 'environmental_sensor'
    if any(keyword in text for keyword in ('空调', 'hvac', '冷机', '风机盘管', '新风')):
        return 'air_conditioner'
    return 'elec_meter'


def build_semantic_entries(projects: list[dict[str, Any]], devices: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    project_index = {int(project['id']): project for project in projects if project.get('id') is not None}
    device_type_map = config.get('device_types') or DEFAULT_DEVICE_TYPES
    metric_types = config.get('metric_types') or {}

    for project in projects:
        project_name = str(project.get('project_name') or '').strip()
        project_code = str(project.get('project_code_name') or '').strip()
        if not project_name:
            continue
        semantic_parts = [project_name]
        if project_code:
            semantic_parts.append(project_code)
        semantic_parts.append(CN_PROJECT)
        entries.append(
            {
                'type': 'project',
                'semantic_text': ' '.join(part for part in semantic_parts if part),
                'metadata': {
                    'project_id': project.get('id'),
                    'project_name': project_name,
                    'project_code': project_code,
                },
            }
        )

    for tag, metric in metric_types.items():
        name = str(metric.get('name') or tag).strip()
        description = str(metric.get('description') or '').strip()
        unit = str(metric.get('unit') or '').strip()
        synonyms = [name, *list(metric.get('synonyms') or [])]
        seen: set[str] = set()
        for synonym in synonyms:
            value = str(synonym or '').strip()
            if not value or value in seen:
                continue
            seen.add(value)
            metadata = {
                'tag': tag,
                'name': name,
                'unit': unit,
                'description': description,
            }
            if value != name:
                metadata['synonym'] = value
            semantic_parts = [value]
            if description:
                semantic_parts.append(description)
            if value == name:
                semantic_parts.append(tag)
            entries.append(
                {
                    'type': 'metric',
                    'semantic_text': ' '.join(part for part in semantic_parts if part),
                    'metadata': metadata,
                }
            )

    for device in devices:
        project_id = int(device.get('project_id') or 0)
        project = project_index.get(project_id)
        project_name = str(project.get('project_name') or '') if project else ''
        device_type = infer_device_type(device.get('device_type'), device.get('device_name'))
        device_type_cn = str(device_type_map.get(device_type) or device_type)
        semantic_parts = [str(device.get('device_name') or '').strip(), device_type_cn, project_name]
        entries.append(
            {
                'type': 'device',
                'semantic_text': ' '.join(part for part in semantic_parts if part),
                'metadata': {
                    'device_id': str(device.get('device') or '').strip(),
                    'device_name': str(device.get('device_name') or '').strip(),
                    'device_type': device_type,
                    'device_type_cn': device_type_cn,
                    'project_id': device.get('project_id'),
                    'project_name': project_name,
                    'tg': str(device.get('tg') or '').strip(),
                },
            }
        )

    return entries


def save_semantic_data(entries: list[dict[str, Any]], output_dir: str = 'data') -> Path:
    output_path = Path(output_dir) / 'semantic_entries.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding='utf-8')
    return output_path


def generate_semantic_layer_config(
    projects: list[dict[str, Any]],
    base_config: dict[str, Any],
    output_path: str = 'config/semantic_layer.yaml',
) -> Path:
    config = dict(base_config)
    config['description'] = config.get('description') or CONFIG_DESCRIPTION
    config['version'] = str(config.get('version') or '1.0')
    config['device_types'] = config.get('device_types') or DEFAULT_DEVICE_TYPES
    config['metric_types'] = config.get('metric_types') or {}
    config['mongodb'] = config.get('mongodb') or {
        'database': os.getenv('MONGODB_DATABASE', 'sensor_db'),
        'collection_pattern': 'source_data_{tag}_{YYYYMM}',
    }
    config['mysql'] = config.get('mysql') or {
        'device_database': 'device',
        'device_table': 'device_info',
        'project_database': 'project',
        'project_table': 'project_info',
    }

    database_name = os.getenv('MONGODB_DATABASE', 'sensor_db')
    projects_map: dict[str, dict[str, Any]] = {}
    for project in projects:
        code = str(project.get('project_code_name') or project.get('id') or '').strip()
        if not code:
            continue
        projects_map[code] = {
            'id': int(project['id']) if project.get('id') is not None else None,
            'name': str(project.get('project_name') or '').strip(),
            'database': database_name,
        }
    config['projects'] = dict(sorted(projects_map.items(), key=lambda item: item[0]))

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='根据元数据构建语义条目文件 semantic_entries.json')
    parser.add_argument('--config', default='config/semantic_layer.yaml', help='语义层配置路径')
    parser.add_argument('--output-dir', default='data', help='输出目录')
    parser.add_argument('--write-config', action='store_true', help='刷新配置文件中的项目映射段')
    parser.add_argument('--config-output', default='config/semantic_layer.yaml', help='配置输出路径')
    parser.add_argument('--include-disabled-projects', action='store_true', help='包含未启用项目')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    config = load_semantic_config(args.config)

    print('正在导出项目元数据...')
    projects = export_projects(config, include_disabled=args.include_disabled_projects)
    print(f'项目数: {len(projects)}')

    print('正在导出设备元数据...')
    devices = export_devices(config)
    print(f'设备数: {len(devices)}')

    print('正在构建语义条目...')
    entries = build_semantic_entries(projects, devices, config)
    output_path = save_semantic_data(entries, args.output_dir)
    print(f'条目数: {len(entries)}')
    print(f'已保存到: {output_path.as_posix()}')

    if args.write_config:
        config_path = generate_semantic_layer_config(projects, config, args.config_output)
        print(f'配置已刷新: {config_path.as_posix()}')

    print('下一步：运行 build_vector_index.py 生成向量和 FAISS 索引')


if __name__ == '__main__':
    main()
