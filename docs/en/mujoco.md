**English** · [简体中文](../zh/mujoco.md) · [Home](../../README.md)

# MuJoCo Guide

## Requirements

- Linux, an NVIDIA GPU compatible with LightNav, NVIDIA drivers and EGL support.
- Git, GNU Make, system Python 3, Conda, Docker with buildx, and NVIDIA Container Toolkit.
- A DashScope API key for the default decision model, plus network access and disk space for source, images and weights.

Initial setup downloads several GB and builds container images. x86_64 uses the upstream build recipe; Thor uses the recipe in this repository. Other ARM devices require a compatible image; see [Model Deployment](deployment.md). Minimum VRAM and disk requirements have not been fully established.

On Ubuntu / Debian, setup installs missing GStreamer packages and `patch` through apt, requiring sudo for non-root users. On other distributions, install `patch` and the [recording dependencies](replay.md#recording-dependencies) first.

## Initial Setup

Run from the repository root. Do not reinstall Python environments while services are using them.

1. Prepare configuration, Python 3.11 environments and models:

   ```bash
   make mujoco-setup
   ```

2. Set `DASHSCOPE_API_KEY` in the generated `deploy/.env`.

“MuJoCo 准备完成” means setup is complete; services have not started. Existing configuration, images and complete weights are reused.
If Conda is outside PATH and `$HOME/miniconda3`, run `make configure` first and set `CONDA_EXE` in `deploy/local.env`.

## Start and Stop

```bash
make mujoco
```

The launcher starts local models, simulation and Agent. After connection checks, it prints “MuJoCo 已就绪” (ready). Open [localhost:8088](http://127.0.0.1:8088) and enter “Find the brown sofa.” Readiness checks do not validate your API key or run navigation.

Cancel the task and wait until idle, then press `Ctrl+C` in the launch terminal. The Agent and simulation started by this session exit; shared model containers keep running. Occupied ports cause an error rather than taking over existing services.

Check connections with `make mujoco-check`. To start services separately, use `make mujoco-sim` and `make mujoco-agent`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Conda not found | Whether `CONDA_EXE` points to the Conda executable |
| Missing model configuration or key | `deploy/.env` and `configs/agent.toml` |
| Port already in use | Identify the owner; use the existing service or stop your own idle service |
| `robot_service` not ready | Simulation errors in the terminal, EGL drivers and the `nav-sim` environment |
| `model_transport` not ready | `docker logs navigation-lightnav`, model address and weights |
| `target_detector` not ready | `docker logs navigation-target-detector` and detector address |
| `navigation_history_continuation` not ready | Whether Agent and simulation use the same repository version |

See [Configuration](configuration.md) for settings, [Model Deployment](deployment.md) for existing models, and [Replay and Recording](replay.md) for task records.
