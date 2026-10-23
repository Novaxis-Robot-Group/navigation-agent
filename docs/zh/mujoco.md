[English](../en/mujoco.md) · **简体中文** · [首页](../../README.zh-CN.md)

# MuJoCo 使用指南

## 环境要求

- Linux、兼容 LightNav 的 NVIDIA GPU、NVIDIA 驱动和 EGL 支持。
- Git、GNU Make、系统 Python 3、Conda、Docker（含 buildx）和 NVIDIA Container Toolkit。
- 默认决策模型所需的百炼 API Key，以及下载源码、镜像和权重的网络与磁盘空间。

首次安装包含数 GB 下载和镜像编译。x86_64 使用上游构建配方；Thor 使用仓库内的兼容配方。其他 ARM 设备需提供兼容镜像，见[模型部署](deployment.md)。最低显存和磁盘需求尚未完整测定。

Ubuntu / Debian 缺少 GStreamer 或 `patch` 时，安装脚本通过 apt 补齐，普通用户需要 sudo。其他系统需预先安装 `patch` 和[录像依赖](replay.md#录像依赖)。

## 首次安装

在仓库根目录执行；不要在已有服务使用 Python 环境时重新安装。

1. 准备配置、Python 3.11 环境和模型：

   ```bash
   make mujoco-setup
   ```

2. 在生成的 `deploy/.env` 中填写 `DASHSCOPE_API_KEY`。

出现“MuJoCo 准备完成”表示安装完成，服务尚未启动。已有配置、镜像和完整权重会复用。
Conda 不在 PATH 或 `$HOME/miniconda3` 时，先运行 `make configure`，在 `deploy/local.env` 设置 `CONDA_EXE`。

## 启动与停止

```bash
make mujoco
```

启动器启动本机模型、仿真和 Agent，检查连接后输出“MuJoCo 已就绪”。在 [localhost:8088](http://127.0.0.1:8088) 输入“找到棕色沙发”。就绪检查不验证 API Key 的有效性，也不执行导航。

取消任务并等待空闲后，在启动终端按 `Ctrl+C`。本次启动的 Agent 和仿真退出，共享模型容器继续运行。端口已被占用时，启动器报错，不接管旧服务。

仅检查连接：`make mujoco-check`。分开启动：`make mujoco-sim`、`make mujoco-agent`。

## 排障

| 现象 | 检查 |
| --- | --- |
| 找不到 Conda | `CONDA_EXE` 是否指向 Conda 可执行文件 |
| 缺少模型配置或密钥 | `deploy/.env` 和 `configs/agent.toml` |
| 端口已有服务 | 确认服务归属；使用已有服务，或停止自己的空闲服务 |
| `robot_service` 未就绪 | 终端中的仿真错误、EGL 驱动和 `nav-sim` 环境 |
| `model_transport` 未就绪 | `docker logs navigation-lightnav`、模型地址和权重 |
| `target_detector` 未就绪 | `docker logs navigation-target-detector` 和检测服务地址 |
| `navigation_history_continuation` 未就绪 | Agent 与仿真是否来自同一版本 |

配置调整见[运行配置](configuration.md)，已有模型接入见[模型部署](deployment.md)，任务记录见[回放与录像](replay.md)。
