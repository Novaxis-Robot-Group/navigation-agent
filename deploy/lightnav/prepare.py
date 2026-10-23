"""校验上游版本，仅生成待挂载的时间处理文件，不修改上游或启动服务。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="现有 LightNav 源码目录")
    parser.add_argument("--output", type=Path, required=True, help="新的补丁输出目录")
    parser.add_argument("--reuse", action="store_true", help="核对并复用相同补丁，不覆盖已有文件")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "sources.json").read_text())
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and args.reuse:
        # 在临时目录重新生成，逐文件核对；不信任仅存在标记文件的旧目录。
        with tempfile.TemporaryDirectory(prefix="lightnav-patch-") as directory:
            expected = Path(directory) / "patch"
            subprocess.run([sys.executable, __file__, str(source), "--output", str(expected)],
                           check=True, capture_output=True, text=True)
            for name in manifest["sha256"]:
                if not (output / name).is_file() or (output / name).read_bytes() != (expected / name).read_bytes():
                    parser.error(f"已有补丁与当前版本不同：{output / name}；不自动覆盖")
        print(f"已核对采图时间补丁：{output}")
        return
    if output.exists():
        parser.error("输出目录已存在；请选择新的空目录，避免覆盖现有部署")
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            parser.error(f"上游文件版本不符：{name}；请先审核补丁，不要强制应用")
    output.mkdir(parents=True)
    try:
        for name in manifest["sha256"]:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        subprocess.run(
            ["patch", "--batch", "-p1", "-d", str(output), "-i", str(here / "capture-time.patch")],
            check=True,
        )
    except Exception:
        shutil.rmtree(output)
        raise
    print(f"已生成 {len(manifest['sha256'])} 个文件：{output}")
    print("在现有 docker run 命令中追加以下挂载参数，再启动服务：")
    for name in manifest["sha256"]:
        value = f"type=bind,source={output / name},target=/workspace/lightnav-0/{name},readonly"
        print("--mount " + shlex.quote(value))


if __name__ == "__main__":
    main()
