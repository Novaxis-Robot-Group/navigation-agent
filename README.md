<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>

<h1 align="center">Navigation Agent</h1>
<p align="center"><em>Describe a goal. Let the robot find its way.</em></p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-52677D?style=flat-square" alt="Python 3.11">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-68766B?style=flat-square" alt="License: Apache 2.0"></a>
</p>

## Overview

Navigation Agent turns natural-language goals into autonomous visual exploration and navigation.
Start with a ready-to-run MuJoCo scene, then explore deployment on your robot.

<p align="center">
  <a href="https://novaxis-robot-group.github.io/navigation-agent-site/#architecture">
    <img src="docs/assets/architecture.svg" width="80%" alt="Navigation Agent architecture: one Agent with independent navigation model and robot adapters">
  </a>
</p>
<p align="center"><sub>One Agent, independent model and robot adapters. Dashed components are planned extensions.</sub><br>
  <a href="https://novaxis-robot-group.github.io/navigation-agent-site/">Website →</a>
</p>

## Quick Start

**Requires:** Linux, a LightNav-compatible NVIDIA GPU, Conda, Docker with NVIDIA Container Toolkit, Git and GNU Make. [Environment guide](docs/en/mujoco.md#requirements).

Run from the repository root. Initial setup downloads several GB of dependencies and models and builds container images.

**1. Set up once**

```bash
make mujoco-setup
```

This prepares the environments, scene and models. Then set `DASHSCOPE_API_KEY` in `deploy/.env`.

**2. Launch anytime**

```bash
make mujoco
```

When the terminal shows **MuJoCo 已就绪** (ready), open [localhost:8088](http://127.0.0.1:8088) and enter **“Find the brown sofa.”**
For later sessions, just run this command. Cancel any active task and wait until idle before pressing `Ctrl+C` to exit.

## Documentation

[MuJoCo Guide](docs/en/mujoco.md) · [Go2 Deployment](docs/en/go2.md) · [Custom Integrations](docs/en/integration.md)

[Model Deployment](docs/en/deployment.md) · [Configuration](docs/en/configuration.md) · [Replay & Recording](docs/en/replay.md)

---

[Apache-2.0](LICENSE) · © 2026 Novaxis（智疆启元）· [Third-party notices](THIRD_PARTY_NOTICES.md)

This GitHub repository hosts source releases. External pull requests are not currently accepted.
