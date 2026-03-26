import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.device_tool import find_device_metadata_with_engine
from src.metadata.metadata_engine import MetadataEngine
from src.config import load_config

config = load_config()
engine = MetadataEngine(config.mysql.connection_string)
devices = find_device_metadata_with_engine('b9', engine)

# 过滤掉非设备信息
device_list = [d for d in devices if isinstance(d, dict) and 'device' in d and 'error' not in d]

print(f"找到设备数: {len(device_list)}")
print(f"前5个设备:")
for i, d in enumerate(device_list[:5], 1):
    print(f"  {i}. {d.get('device')} - {d.get('name')}")
