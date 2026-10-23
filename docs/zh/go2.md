[English](../en/go2.md) · **简体中文** · [首页](../../README.zh-CN.md)

# Go2 部署

Go2 本体运行相机和客户端；GPU 主机运行 Agent 与模型服务。两端使用同一版本仓库，真机运动由现场人员执行。

## 首次安装

两端需要 GNU Make。Go2 需要 Docker 和 Python 3.8+；GPU 主机需要 Conda、Docker 与 NVIDIA Container Toolkit；安装脚本创建 Python 3.11 的 `nav-agent` 环境。

1. 两端分别运行 `make configure`。
2. 在 Go2 的 `deploy/local.env` 填写 SDK 网卡与主机连接：

   ```bash
   export NETWORK_INTERFACE="eth0"
   export AGENT_HOST="user@agent-host"
   ```

   替换实际值，并确认 SSH 密钥登录可用。默认相机为 Gemini 335Le；USB 33x 相机另设 `ENUMERATE_NET_DEVICE=0`。使用直连网络时留空 `AGENT_HOST`。

3. 在 Go2 构建客户端：`make go2-build`。
4. 在 GPU 主机安装 Agent：`make go2-setup`。
5. 在 GPU 主机准备模型：`make models-setup`。已有模型见[模型部署](deployment.md)。
6. 在 GPU 主机的 `deploy/.env` 填写 `DASHSCOPE_API_KEY`。
7. 在 GPU 主机安装[录像依赖](replay.md#录像依赖)，或关闭录像。

核对 `configs/go2-lightnav.toml` 中的地址：

| 字段 | 默认值 | 访问方 |
| --- | --- | --- |
| `robot.base_url` | `http://127.0.0.1:18088` | Agent 主机 |
| `vln.server_url` | `ws://127.0.0.1:8053` | Go2 本体 |
| `target_watch.server_url` | `http://127.0.0.1:18060` | Agent 主机 |

配置了 `AGENT_HOST` 时，启动脚本维护两条隧道：Go2 的 `8053` 连接主机模型，主机的 `18088` 连接 Go2 的 `8088`。直连部署需填写各访问方可达的地址。

## 启动与检查

在 Go2 执行：

```bash
make go2-client
```

在 GPU 主机执行：

```bash
make go2
```

`make go2` 自动使用 `nav-agent`，无需手动激活环境。

模型加载后，在 GPU 主机另开终端执行 `make go2-check`。应返回 `robot_service: true`、`protocol: 1`、`streaming_observation: true`、`observation: true` 和 `missing_capabilities: []`；启用检测时还需 `target_detector: true`。

检查会读取观测，不发送运动命令。打开主机的 [Agent 页面](http://127.0.0.1:8090)，由现场人员确认画面、机器人状态及运动条件。

取消任务并确认停稳后，在两端按 `Ctrl+C`。仅本次启动的 Agent、客户端和隧道退出，共享模型继续运行。

## 排障与维护

| 现象 | 处理 |
| --- | --- |
| 本体连接失败 | 核对 `robot.base_url`、客户端及实际连接路径 |
| 模型连接失败 | 从 Go2 本体检查 `vln.server_url`，不能只检查主机 |
| 缺少协议或能力 | 两端使用匹配版本；客户端必须使用本仓库构建的镜像 |
| `release pending` 或 SDK 释放失败 | 保持任务停止，由现场人员检查本体与客户端日志 |
| 观测不可用 | 检查相机、里程计及采图时间 |

修改客户端代码后，先由现场人员确认停稳、释放控制，再构建镜像并重新创建自己的客户端容器。只更新主机或重启旧容器不会更新狗端代码。

网页保存的运行参数绑定本体地址，优先于 TOML。配置与场景规则见[运行配置](configuration.md)，录像见[回放与录像](replay.md)。来源与许可见[第三方声明](../../THIRD_PARTY_NOTICES.md)。
