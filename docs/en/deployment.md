**English** · [简体中文](../zh/deployment.md) · [Home](../../README.md)

# Model Deployment

MuJoCo and Go2 share LightNav and YOLO services. `make mujoco-setup` already prepares models; use this page for separate deployment or maintenance.

## Prepare and Start

Requires Git, Python 3, `patch`, Docker with buildx, NVIDIA Container Toolkit and a compatible GPU. Initial builds and weights require several GB of downloads. The first Thor compilation can take substantial time.

1. Run `make configure`.
2. To reuse resources, set the following options in `deploy/local.env`. Skip this for default deployment.

   | Option | Purpose |
   | --- | --- |
   | `LIGHTNAV_IMAGE` | A validated compatible image |
   | `LIGHTNAV_SOURCE` | Source matching the pinned revision |
   | `LIGHTNAV_WEIGHTS` | Complete LightNav weights directory |
   | `DETECTOR_IMAGE` | Detector service image |
   | `DETECTOR_MODELS` | Directory containing YOLO and CLIP weights |

3. Run `make models-setup`.
4. After “LightNav 和 YOLO 已准备；没有启动业务服务” (models prepared, services not started), run `make models`.

`make mujoco` and `make go2` also start local models according to configuration. Use `make lightnav` or `make yolo` to start one service. Automatic reuse requires matching owner and service labels; existing containers retain their configuration.

| Service | Default address | Container |
| --- | --- | --- |
| LightNav | `ws://127.0.0.1:8053` | `navigation-lightnav` |
| YOLO | `http://127.0.0.1:18060` | `navigation-target-detector` |

Check connections with `make mujoco-check` or `make go2-check`. YOLO's `/capabilities` must return `ready: true`. Connection checks do not validate inference or navigation.

Pinned source and weight revisions are in [sources.json](../../deploy/lightnav/sources.json). x86_64 uses the upstream Dockerfile; Thor uses the [CUDA 13 / SM110 recipe](../../deploy/lightnav/thor/Dockerfile). Other ARM devices require a compatible image.

Thor builds use 2 parallel jobs by default. To adjust this, set `export VLLM_BUILD_JOBS=2` (a positive integer) in `deploy/local.env`; increase it only with sufficient memory. This affects new image builds only, not inference settings or existing containers.

## External Models and Detector

For external models, change `[vln].server_url` and `[target_watch].server_url` in the backend TOML, then set `export START_MODELS=0` in `deploy/local.env`. LightNav must be reachable from the simulation host or Go2; the detector must be reachable from the Agent host. Both setup and launch skip local model management.

To deploy only the detector, use a base image with compatible PyTorch and TorchVision:

```bash
bash deploy/detector/build.sh your-compatible-pytorch-image:tag
bash deploy/detector/run.sh /absolute/path/to/models
```

The weights directory must contain `yolov8l-worldv2.pt` and `clip/ViT-B-32.pt`. The build script does not install PyTorch or CUDA. See [Configuration](configuration.md#target-detection) for detection and sampling settings.

## Capture-time Patch

Standard setup applies this patch automatically, preserving the official model, weights, history-frame selection and action decoding. The steps below are for manually maintaining an existing LightNav service.

1. Confirm the service is idle and its source matches `sources.json`.
2. Generate patched files. Replace the source path; the output directory must not exist:

   ```bash
   python deploy/lightnav/prepare.py /path/to/LightNav-0 --output .runtime/lightnav-capture-time
   ```

3. Add the five generated mount arguments to the original `docker run` command. The default target is `/workspace/lightnav-0`; adjust targets for a different source layout.
4. Recreate the container with those arguments, preserving its image, model parameters and data volumes.
5. In a model environment that can import the patched source, run `python /path/to/navigation-agent/deploy/lightnav/test_capture_time.py`. All tests should pass.

The script leaves upstream source unchanged. On the next navigation run, `history_elapsed_s` should match the actual image-capture interval. The client should no longer report missing capture-time support. The two machines do not need synchronized clocks.

To revert, wait until idle, remove the five patch mounts and recreate the container with its original configuration. `docker restart` does not change mounts.

## Updates and Troubleshooting

Inspect `docker logs navigation-lightnav` or `docker logs navigation-target-detector`. If source or patch revisions do not match, prepare matching files in a separate directory without overwriting files in use.

Changes to an image, port, weight path or mount require recreating the corresponding container while idle. If a legacy container has missing or mismatched labels, confirm its ownership and service purpose, then explicitly reuse it with `REUSE_EXISTING=1 make models`.

Agent or TOML changes alone do not require rebuilding model images. Shared containers remain running when the Agent exits.
