**English** · [简体中文](../zh/configuration.md) · [Home](../../README.md)

# Configuration

The default MuJoCo configuration includes service connections, detection and recording. For a first run, only the API key in `deploy/.env` is required.

## Files and Reloading

| File | Purpose |
| --- | --- |
| `configs/agent.toml` | Decision model, budgets, exploration, memory and detection sampling |
| `configs/mujoco-lightnav.toml` | MuJoCo connections, navigation, camera and recording |
| `configs/go2-lightnav.toml` | Go2 connections, motion protection and scene |
| `deploy/.env` | API keys; keep out of Git |
| `deploy/local.env` | Conda, image and launch options; keep out of Git |

Relative paths in TOML resolve from the configuration file's directory. Restart services that read a changed TOML file; restart both Agent and simulation after changing MuJoCo connections or recording. After changing `deploy/local.env`, restart the Make session. Existing containers are not rewritten.

Saved Go2 web settings take precedence over TOML and are bound to the robot address. The task form can override budgets and per-segment time limits for one task. Effective values are recorded in `configuration` and `effective_execution`.

## Decision Model

Set `provider`, `model`, `base_url` and `api_key_env` under `[llm]` in `configs/agent.toml`. The default provider is DashScope, using `DASHSCOPE_API_KEY` from `deploy/.env`.

Choose a supported provider and a vision-capable model. For Zhipu, use `provider = "zhipu"`, `api_key_env = "ZHIPU_API_KEY"`, and the corresponding API endpoint and model name. Restart the Agent after changing configuration or keys. The navigation model is configured separately under backend `[vln]`.

## Tasks and Navigation

| Setting | Current default | Purpose |
| --- | --- | --- |
| Backend `[ui.task_defaults]` | 300 steps, 600 seconds | MuJoCo / Go2 web budgets |
| Agent `[task_defaults]` | 300 steps, 300 seconds | Standalone Agent budget |
| Backend `[navigation].time_limit_enabled` | `false` | No fixed per-segment timeout |
| Backend `[navigation].inactivity_timeout_s` | `15.0` | Return `no_progress` when translation and rotation remain negligible |
| Agent `[exploration].viewpoint_max_displacement_m` | `1.5` | Local observation displacement limit; `0.0` disables it |
| Agent `[selective].max_interventions` | `16` | Maximum planning interventions |

Segments without a fixed timeout still obey the total task budget and motion protections. The web form can enable a segment timeout for one task; refreshing the page resets that switch to off.

MuJoCo initially sends the full English goal to LightNav, translating Chinese once when needed. Events trigger the Planner to choose subsequent actions. After model STOP, the Agent checks identity, arrival and orientation; STOP alone is not task success.

Pausing the same instruction within a task can preserve model history. Turning, changing instructions, STOP, cancellation, disconnection or losing control clears history. Resuming creates a new action identity.

## Target Detection

YOLO is enabled in the supplied backend configurations. Set backend `[target_watch].server_url` to `""` to disable it, or to a service address to enable it. Restart the Agent after changes. See [Model Deployment](deployment.md) for installation.

Agent `[target_watch]` controls sampling: defaults are 0.2-second intervals, 0.25 confidence and two consistent frames. Detection starts after the Planner specifies a category. A match requests a stop for verification. The decision model still judges color and identity; detection failure does not prove absence.

## Scenes and Prompts

The common prompt is `configs/prompts/common.md`, reloaded each step by default. Backend `[prompt].path` adds scene rules. Go2 uses `go2-office.md` and the `go2_office` validator by default; MuJoCo does not load these office rules.

VLN instruction-guide priority is backend `vln.instruction_guide`, Agent `prompt.instruction_guide`, then the adapter default. Saved Go2 web guides override configuration. Algorithm rules are in `configs/prompts/scheduling.md`; the web guide cannot override them.

Restart the Agent after changing prompt paths, guides or scene-validation configuration. Text rules do not replace robot collision avoidance or speed limits.

The supplied MuJoCo configuration sets `[simulation].camera_height_m` to `0.8` meters. Restart simulation after changing it. For recording settings, see [Replay and Recording](replay.md).
