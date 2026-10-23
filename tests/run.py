"""使用 Agent 或仿真环境运行测试，无需启动业务服务或提供 Key。"""

import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=sorted(path.name.removesuffix('_tests') for path in (ROOT / 'tests').glob('*_tests') if path.is_dir()))
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests' / (args.suite + '_tests')))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
