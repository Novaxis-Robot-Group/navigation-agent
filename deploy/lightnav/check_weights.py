"""离线核对固定 LightNav 权重所需文件；不读取或修改模型参数。"""
import json
from pathlib import Path
import sys


def check(directory):
    root = Path(directory)
    names = [
        "config.json", "eval_config.json", "tokenizer.json", "tokenizer_config.json",
        "processor_config.json", "chat_template.jinja", "model.safetensors.index.json",
        "action_tokenizer/manifest.json", "action_tokenizer/alpha_per_source.json",
        "action_tokenizer/jacobian_weights.npy",
    ]
    for level in range(3):
        names.extend(["action_tokenizer/codebook_l%d.npy" % level,
                      "action_tokenizer/distance_l%d.npy" % level])
    index = root / "model.safetensors.index.json"
    if index.is_file():
        names.extend(json.loads(index.read_text())["weight_map"].values())
    missing = sorted({name for name in names if not (root / name).is_file() or (root / name).stat().st_size == 0})
    if missing:
        raise ValueError("LightNav 权重不完整：" + ", ".join(missing))


if __name__ == "__main__":
    try:
        check(sys.argv[1])
    except (ValueError, OSError, KeyError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
