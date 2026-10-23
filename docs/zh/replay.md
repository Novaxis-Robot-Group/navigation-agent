[English](../en/replay.md) · **简体中文** · [首页](../../README.zh-CN.md)

# 回放与录像

默认配置已启用录像：MuJoCo 保存第一、第三视角，Go2 保存第一视角。关闭录像仍保留任务日志与观察图片。

## 查看与下载

1. 等待任务结束、录像收尾。
2. 在“历史任务”中选择记录，打开“录像回放”；Go2 入口为“录像与历史回放”。
3. 选择视角，拖动时间轴或点击关键节点。
4. 点击“下载当前视角 MP4”。

显示“录像已保存”后即可下载；仍在生成时点击“刷新记录”。事件字幕通过 VTT 单独导出，MP4 不内嵌字幕。

MuJoCo 的“直接导航”也会保存记录。在历史任务中选择“VLN”查看；模型 STOP 只表示导航停止，不代表目标已完成。

## 开关与位置

| 后端 | 配置节 | 默认录像目录 | 修改后重启 |
| --- | --- | --- | --- |
| MuJoCo | `configs/mujoco-lightnav.toml` 的 `[simulation.recording]` | `runs/mujoco-recordings/` | Agent、仿真 |
| Go2 | `configs/go2-lightnav.toml` 的 `[recording]` | `runs/go2/recordings/` | Agent |

设置 `enabled = false` 关闭，`true` 开启。`directory` 相对 TOML 所在目录解析，`fps` 默认为 10。

每个任务目录包含 `first.mp4`；MuJoCo 还包含 `third.mp4`。`recording.json` 记录状态、帧数、丢帧与错误，`frames.jsonl` 记录采图时间及位姿。

录像不会自动删除。清理任务时，按任务 ID 同时处理日志与视频。

## 录像依赖

`make mujoco-setup` 会安装录像依赖。Go2 主机或已有环境缺少依赖时，Ubuntu / Debian 使用：

```bash
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

用 `gst-inspect-1.0 x264enc` 和 `gst-inspect-1.0 mp4mux` 确认编码插件可用，再重启对应服务。

## 排障

| 现象 | 处理 |
| --- | --- |
| 没有视频 | 检查录像开关、GStreamer 和 `recording.json` 的错误 |
| 旧任务只有图片 | 未录制的视频无法补录，使用已有图片回放 |
| 视频短于任务时长 | 检查丢帧数；MP4 按实际保存帧数编码 |
| 轨迹有断点 | 该段缺少有效位姿，回放不推算缺失轨迹 |

录像失败不会终止导航，也不意味着导航成功。
