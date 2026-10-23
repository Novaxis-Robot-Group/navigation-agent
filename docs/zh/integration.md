[English](../en/integration.md) · **简体中文** · [首页](../../README.zh-CN.md)

# 自定义接入

先选择需要扩展的部分。更换兼容服务只需改地址，无需新增适配器。

| 需求 | 入口 |
| --- | --- |
| 更换决策模型 | [运行配置](configuration.md#决策模型) |
| 接入新导航协议 | `contracts/vln.py` → `vln/` → `app/deployments/vln.py` |
| 接入新本体 | `contracts/agent_backend.py` → `robots/` → 应用层装配 |
| 调整场景要求 | [运行配置](configuration.md#场景与提示词) |

表中的 Python 路径均相对 `src/navigation_agent/`。以下命令在仓库根目录、已准备的 `nav-agent` 环境执行。

## 导航模型

1. 按 [VlnClient](../../src/navigation_agent/contracts/vln.py) 实现客户端：接收带采图时间的 JPEG，返回路径点与明确的 STOP 信号。客户端负责排队、背压和过期会话隔离。
2. 参考 [custom_vln.py](../../examples/custom_vln.py)，在读取配置前调用 `register_vln`，登记参数模型和客户端工厂。
3. 在后端 TOML 选择适配器：

   ```toml
   [vln]
   backend = "example"
   server_url = "ws://model-host:9900"
   [vln.options]
   horizon = 6
   ```

4. 在 Agent 和实际执行客户端的服务进程中使用同一注册入口，再检查连接。

先运行离线装配示例：

```bash
PYTHONPATH=src conda run -n nav-agent python examples/custom_vln.py
```

应输出客户端状态和 `horizon=6`。该示例不执行真实推理。Go2 原生服务目前只支持 LightNav；只在 Agent 注册新名称不能更换狗端协议。

## 本体

1. 实现 [RobotBackend](../../src/navigation_agent/contracts/agent_backend.py) 的观察、转向、导航、停车和控制权接口。
2. 按 [custom_backend.py](../../examples/custom_backend.py) 通过 `NavigationTools` 注入 `run_task`，或向 `StandaloneDeployment` 传入 `backend_factory`。
3. 需要 TOML 入口时，在读取配置前调用 `register_robot`，注册 `RobotAdapter` 的配置类型与工厂，见 [factory.py](../../src/navigation_agent/app/deployments/factory.py)。

运行离线最小示例：

```bash
PYTHONPATH=src conda run -n nav-agent python examples/custom_backend.py
```

应输出 `completed` 和记录目录。示例使用固定图片与响应，不连接设备；实际接入需替换后端并配置决策模型。

动作结束、取消或异常时，只停止本连接控制的运动。观测中的图片、采图时间和位姿必须对应同一帧，不用分别查询的数据拼接。

## 可选能力

- **途中检测：** 实现 `StreamingObservationBackend`，提供只读采图和带动作身份的中断请求。过期动作不能中断新动作。
- **历史延续：** 后端实现 `configure_history`，模型客户端实现 `HistoryVlnClient.pause/resume`。不支持的自定义部署可在 Agent TOML 设置 `[harness].preserve_same_instruction_history = false`。
- **实际轨迹：** 动作结果的 `motion.sampled_positions` 可返回同一坐标系中的实测 XY 点。不要用起终点插值代替实测轨迹。

## 验证

```bash
bash deploy/agent/run.sh --backend-config configs/mujoco-lightnav.toml --check-config
bash deploy/agent/run.sh --backend-config configs/mujoco-lightnav.toml --check
```

替换为实际后端配置。第一条只校验配置；第二条检查连接与能力，不执行导航。内置 Go2 的检查还读取真实观测。配置、连接通过后，再验证实际任务；真机运动由现场人员执行。
