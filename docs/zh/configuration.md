[English](../en/configuration.md) · **简体中文** · [首页](../../README.zh-CN.md)

# 运行配置

默认 MuJoCo 已配好服务连接、检测与录像。首次使用只需在 `deploy/.env` 填写密钥。

## 配置位置与生效

| 文件 | 用途 |
| --- | --- |
| `configs/agent.toml` | 决策模型、任务预算、探索、记忆、检测采样 |
| `configs/mujoco-lightnav.toml` | MuJoCo 连接、导航、相机与录像 |
| `configs/go2-lightnav.toml` | Go2 连接、运动保护与场景 |
| `deploy/.env` | API Key，不提交 Git |
| `deploy/local.env` | Conda、镜像与启动选项，不提交 Git |

TOML 相对路径以配置文件所在目录为基准。修改 TOML 后重启读取它的服务；修改 MuJoCo 连接或录像时重启 Agent 和仿真。修改 `deploy/local.env` 后重新启动 Make 会话，不会改写已有容器。

Go2 网页保存的运行参数优先于 TOML，绑定本体地址。任务表单可覆盖本次预算与单段导航时限；实际值见任务记录的 `configuration` 与 `effective_execution`。

## 决策模型

在 `configs/agent.toml` 的 `[llm]` 配置 `provider`、`model`、`base_url`、`api_key_env`。默认使用百炼，密钥从 `deploy/.env` 的 `DASHSCOPE_API_KEY` 读取。

选择项目支持的提供方和视觉模型；智谱使用 `provider = "zhipu"` 与 `api_key_env = "ZHIPU_API_KEY"`，并填写相应 API 地址和模型名。修改配置或密钥后重启 Agent。导航模型另由后端 `[vln]` 配置。

## 任务与导航

| 配置 | 当前默认值 | 作用 |
| --- | --- | --- |
| 后端 `[ui.task_defaults]` | 300 步、600 秒 | MuJoCo / Go2 页面预算 |
| Agent `[task_defaults]` | 300 步、300 秒 | 独立 Agent 默认预算 |
| 后端 `[navigation].time_limit_enabled` | `false` | 不按固定单段时长截断 |
| 后端 `[navigation].inactivity_timeout_s` | `15.0` | 无明显平移或转向时返回 `no_progress` |
| Agent `[exploration].viewpoint_max_displacement_m` | `1.5` | 局部观察位移上限；`0.0` 关闭 |
| Agent `[selective].max_interventions` | `16` | 规划干预上限 |

单段不限时仍受任务总预算和运动保护约束。网页可为单次任务开启单段超时；刷新页面后该开关恢复关闭。

MuJoCo 默认先将完整英文目标交给 LightNav；中文目标先翻译一次。事件触发后，Planner 选择后续动作。模型 STOP 后核验目标身份、到达和朝向，不能把 STOP 当作任务成功。

同一任务、同一指令的暂停可保留模型历史。转向、换指令、STOP、取消、断连或失去控制时清空历史；恢复使用新的动作编号。

## 目标检测

内置后端默认启用 YOLO。将后端 `[target_watch].server_url` 设为 `""` 关闭；填写服务地址开启，修改后重启 Agent。模型安装见[模型部署](deployment.md)。

Agent `[target_watch]` 管理采样：默认间隔 0.2 秒、置信度 0.25、连续两帧确认。Planner 指定类别后才开始检测；命中后申请停车复核。颜色与目标身份仍由决策模型判断，检测失败不等于目标不存在。

## 场景与提示词

通用 Prompt 为 `configs/prompts/common.md`，默认每轮热读。后端 `[prompt].path` 可追加场景规则；Go2 默认使用 `go2-office.md` 和 `go2_office` 校验器。MuJoCo 不加载该办公场景规则。

VLN 指令指南优先级为后端 `vln.instruction_guide`、Agent `prompt.instruction_guide`、适配器默认指南。Go2 网页保存的指南优先于配置。算法规则位于 `configs/prompts/scheduling.md`，网页指南不能覆盖它。

修改 Prompt 路径、指南或场景校验配置后重启 Agent。文字规则不替代本体避障与限速。

MuJoCo 的 `[simulation].camera_height_m` 默认为示例中的 `0.8` 米；修改后重启仿真。录像配置见[回放与录像](replay.md)。
