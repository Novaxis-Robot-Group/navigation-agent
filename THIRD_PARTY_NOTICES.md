# 第三方声明

本项目自有代码采用 Apache-2.0；第三方作品保留原有许可与版权。

## 随源码分发的 LightNav 派生代码

来源：[LightNav-0](https://github.com/lightorigins/LightNav-0)，作者 Light Origins，Apache-2.0。完整许可见 [LICENSE](LICENSE)，归属见 [NOTICE](NOTICE)。

以下路径相对 `src/navigation_agent/`，保留上游 ROS 包结构：

- `robots/go2/go2_adapter`、`robots/go2/vln_mpc`：驱动、安全控制与 MPC。
- `vln/lightnav/vln_client`：模型客户端及 ROS 节点。
- vln/lightnav/transport.py：基于 d83fbfb 客户端连接循环，增加最新帧队列及采图时间。
- 仓库根目录 deploy/lightnav/capture-time.patch：基于公开上游 d83fbfb9748ca6b497a2f2a7f4a3e95e2110e40c 的时间处理代码，原文摘要见同目录 sources.json。
- `app/go2/vln_web`、`app/go2/vln_bringup`：网页服务与启动装配。

本项目增加或调整了 Agent 控制权租约、心跳、停车保护、原子观测和运行参数协议，并按职责移动源码、移除 TRON 装配依赖。原始提交、逐文件导入摘要见 [来源记录](docs/lightnav-source.json)。原文件中的版权与许可说明保留。

## 安装时取得的依赖

MuJoCo 客户端、仿真库和场景通过固定提交的 `vln-mujoco` 安装，不复制到本仓库。上游代码采用 Apache-2.0；其 ProcTHOR `val_2` 场景及 THOR 资产来自 Allen Institute for AI 的 [MolmoSpaces](https://github.com/allenai/molmospaces)，上游声明采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。使用这些资产时保留相应归属与许可；具体清单见安装包的 `assets/manifest.json` 及 [上游第三方声明](https://github.com/lightorigins/LightNav-0/blob/d83fbfb9748ca6b497a2f2a7f4a3e95e2110e40c/THIRD_PARTY_NOTICES.md)。

Go2 镜像安装 ROS 2 Humble、Unitree SDK2 Python、Orbbec ROS 驱动、CasADi 等依赖。它们不随本仓库源码分发：Unitree SDK 为 BSD-3-Clause；Orbbec ROS 包含单独许可的 SDK 二进制；CasADi 使用 LGPL。构建和分发镜像时仍需保留各依赖的许可要求。

模型服务器及权重由使用者另外部署，遵循所用项目与检查点的许可。上述说明不会将第三方软件、资产或权重重新许可为本项目自有代码。
