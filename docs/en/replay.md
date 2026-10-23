**English** · [简体中文](../zh/replay.md) · [Home](../../README.md)

# Replay and Recording

Recording is enabled in the supplied configurations. MuJoCo saves first- and third-person views; Go2 saves the first-person view. Disabling video still preserves task logs and observation images.

## View and Download

1. Wait for the task and recording to finish.
2. Select a record in task history and open “录像回放” (video replay); Go2 uses “录像与历史回放”.
3. Choose a view and scrub the timeline or select a key event.
4. Select “下载当前视角 MP4” to download the current view.

Download when the page reports that the video is saved. If it is still processing, select “刷新记录” (refresh). Event captions export separately as VTT; they are not embedded in the MP4.

MuJoCo's direct-navigation panel also saves records. Select “VLN” in task history to view them. A model STOP ends navigation but does not establish task success.

## Settings and Storage

| Backend | Configuration section | Default recording directory | Restart after changes |
| --- | --- | --- | --- |
| MuJoCo | `[simulation.recording]` in `configs/mujoco-lightnav.toml` | `runs/mujoco-recordings/` | Agent and simulation |
| Go2 | `[recording]` in `configs/go2-lightnav.toml` | `runs/go2/recordings/` | Agent |

Set `enabled = false` to disable or `true` to enable. `directory` resolves relative to the TOML file; the default `fps` is 10.

Each task directory contains `first.mp4`; MuJoCo also includes `third.mp4`. `recording.json` stores status, frame counts, drops and errors. `frames.jsonl` stores capture times and poses.

Recordings are not deleted automatically. Match task IDs when removing logs and videos.

## Recording Dependencies

`make mujoco-setup` installs recording dependencies. For a Go2 host or an existing environment missing them, use this on Ubuntu / Debian:

```bash
sudo apt-get install gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

Verify encoders with `gst-inspect-1.0 x264enc` and `gst-inspect-1.0 mp4mux`, then restart the corresponding services.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| No video | Check recording settings, GStreamer and errors in `recording.json` |
| An old task has images only | Missing video cannot be recreated; use image replay |
| Video is shorter than the task | Check dropped frames; MP4 duration follows saved frame count |
| Gaps in the trajectory | Valid poses are missing; replay does not invent them |

Recording failure does not stop navigation and does not imply navigation success.
