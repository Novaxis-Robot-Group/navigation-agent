[English](../en/deployment.md) · **简体中文** · [首页](../../README.zh-CN.md)

# 模型部署

MuJoCo 与 Go2 共用 LightNav 和 YOLO 服务。`make mujoco-setup` 已包含模型准备；本页用于单独部署或维护模型。

## 准备与启动

需要 Git、Python 3、`patch`、Docker（含 buildx）、NVIDIA Container Toolkit 和兼容 GPU。首次构建与权重下载可达数 GB，Thor 首次编译耗时较长。

1. 运行 `make configure`。
2. 如需复用资源，在 `deploy/local.env` 设置下列选项；默认部署可跳过。

   | 选项 | 用途 |
   | --- | --- |
   | `LIGHTNAV_IMAGE` | 已验证的兼容镜像 |
   | `LIGHTNAV_SOURCE` | 与固定版本匹配的源码目录 |
   | `LIGHTNAV_WEIGHTS` | 完整的 LightNav 权重目录 |
   | `DETECTOR_IMAGE` | 检测服务镜像 |
   | `DETECTOR_MODELS` | 含 YOLO 与 CLIP 权重的目录 |

3. 运行 `make models-setup`。
4. 出现“LightNav 和 YOLO 已准备；没有启动业务服务”后，运行 `make models`。

`make mujoco` 和 `make go2` 也会按配置启动本机模型。单独启动可用 `make lightnav`、`make yolo`。仅自动复用归属与服务标签均匹配的容器，保留原配置。

| 服务 | 默认地址 | 容器 |
| --- | --- | --- |
| LightNav | `ws://127.0.0.1:8053` | `navigation-lightnav` |
| YOLO | `http://127.0.0.1:18060` | `navigation-target-detector` |

运行 `make mujoco-check` 或 `make go2-check` 验证连接。YOLO 的 `/capabilities` 应返回 `ready: true`。连接通过不代表推理或导航已验证。

源码与权重版本见 [sources.json](../../deploy/lightnav/sources.json)。x86_64 使用上游 Dockerfile，Thor 使用 [CUDA 13 / SM110 配方](../../deploy/lightnav/thor/Dockerfile)；其他 ARM 设备需指定兼容镜像。

Thor 默认使用 2 路编译。需要调整时，在 `deploy/local.env` 设置 `export VLLM_BUILD_JOBS=2`（正整数）；仅内存充足时提高。此选项只影响新镜像构建，不改变模型推理参数或已有容器。

## 外部模型与检测器

使用其他主机的模型时，在后端 TOML 修改 `[vln].server_url` 和 `[target_watch].server_url`，并在 `deploy/local.env` 设置 `export START_MODELS=0`。LightNav 地址需从仿真主机或 Go2 本体可达，检测地址需从 Agent 主机可达。安装与启动均跳过本机模型管理。

只部署检测器时，使用包含兼容 PyTorch / TorchVision 的基础镜像：

```bash
bash deploy/detector/build.sh your-compatible-pytorch-image:tag
bash deploy/detector/run.sh /absolute/path/to/models
```

权重目录需包含 `yolov8l-worldv2.pt` 和 `clip/ViT-B-32.pt`。构建脚本不安装 PyTorch 或 CUDA。检测开关和采样配置见[运行配置](configuration.md#目标检测)。

## 采图时间补丁

标准安装自动应用补丁，保留官方模型、权重、历史帧选择和动作解码。以下仅用于手动维护已有 LightNav 服务。

1. 确认服务空闲，源码版本与 `sources.json` 一致。
2. 生成补丁文件；将源目录换成实际路径，输出目录必须不存在：

   ```bash
   python deploy/lightnav/prepare.py /path/to/LightNav-0 --output .runtime/lightnav-capture-time
   ```

3. 将输出的五项挂载参数加入原 `docker run` 命令。默认目标为 `/workspace/lightnav-0`；不同源码布局需调整挂载目标。
4. 保留原镜像、模型参数和数据卷，按新命令重建容器。
5. 在可导入补丁后源码的模型环境运行 `python /path/to/navigation-agent/deploy/lightnav/test_capture_time.py`，确认测试通过。

脚本不修改上游源码。下一次导航的 `history_elapsed_s` 应对应真实采图时间跨度，客户端不再提示“LightNav 服务未声明采图时间支持”。两台机器无需校时。

撤回时，在服务空闲后移除五项补丁挂载，按原配置重建容器。`docker restart` 不会更新挂载。

## 更新与排障

查看 `docker logs navigation-lightnav` 或 `docker logs navigation-target-detector`。源码或补丁版本不匹配时，使用独立目录准备匹配版本，不覆盖正在使用的文件。

修改镜像、端口、权重路径或挂载后，需要在服务空闲时重新创建对应容器。旧容器的标签不匹配时，先确认归属与服务用途，再用 `REUSE_EXISTING=1 make models` 显式复用。

只修改 Agent 或 TOML 时无需重建模型镜像。共享容器不会随 Agent 退出而停止。
