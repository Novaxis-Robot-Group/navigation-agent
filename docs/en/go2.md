**English** · [简体中文](../zh/go2.md) · [Home](../../README.md)

# Go2 Deployment

The Go2 runs the camera and robot client. A GPU host runs the Agent and model services. Use the same repository version on both machines. On-site operators perform physical motion tests.

## Initial Setup

Both machines need GNU Make. The Go2 needs Docker and Python 3.8+; the GPU host needs Conda, Docker and NVIDIA Container Toolkit. Setup creates the `nav-agent` environment with Python 3.11.

1. Run `make configure` on both machines.
2. Set the SDK network interface and host connection in the Go2's `deploy/local.env`:

   ```bash
   export NETWORK_INTERFACE="eth0"
   export AGENT_HOST="user@agent-host"
   ```

   Replace these values and verify SSH key authentication. The default camera is Gemini 335Le; set `ENUMERATE_NET_DEVICE=0` for a USB 33x camera. Leave `AGENT_HOST` empty for a direct network connection.

3. Build the client on Go2: `make go2-build`.
4. Install the Agent on the GPU host: `make go2-setup`.
5. Prepare models on the GPU host: `make models-setup`. For existing models, see [Model Deployment](deployment.md).
6. Set `DASHSCOPE_API_KEY` in the GPU host's `deploy/.env`.
7. Install [recording dependencies](replay.md#recording-dependencies) on the GPU host, or disable recording.

Check addresses in `configs/go2-lightnav.toml`:

| Field | Default | Accessed by |
| --- | --- | --- |
| `robot.base_url` | `http://127.0.0.1:18088` | Agent host |
| `vln.server_url` | `ws://127.0.0.1:8053` | Go2 |
| `target_watch.server_url` | `http://127.0.0.1:18060` | Agent host |

With `AGENT_HOST` set, the launcher maintains two tunnels: Go2 port `8053` reaches the host model, and host port `18088` reaches Go2 port `8088`. For direct connections, use addresses reachable by each caller.

## Start and Check

On Go2:

```bash
make go2-client
```

On the GPU host:

```bash
make go2
```

`make go2` selects `nav-agent` automatically; no manual activation is needed.

After models load, run `make go2-check` in another terminal on the GPU host. Expect `robot_service: true`, `protocol: 1`, `streaming_observation: true`, `observation: true` and `missing_capabilities: []`. With detection enabled, also expect `target_detector: true`.

The check reads observations without issuing motion commands. Open the host's [Agent page](http://127.0.0.1:8090). An on-site operator must confirm the camera view, robot state and conditions for movement.

Cancel the task and confirm the robot has stopped before pressing `Ctrl+C` on both machines. Only the Agent, client and tunnels started by this session exit; shared models keep running.

## Troubleshooting and Maintenance

| Symptom | Action |
| --- | --- |
| Robot connection fails | Check `robot.base_url`, the client and the connection route |
| Model connection fails | Check `vln.server_url` from Go2, not just from the host |
| Missing protocol or capability | Use matching versions and a client image built from this repository |
| `release pending` or SDK release failure | Keep tasks stopped; have the on-site operator inspect the robot and client logs |
| Observation unavailable | Check the camera, odometry and capture timestamps |

After changing client code, have an on-site operator confirm the robot is stationary and control is released. Then build the image and recreate your client container. Updating only the host or restarting an old container does not update robot-side code.

Web settings are bound to the robot address and take precedence over TOML. See [Configuration](configuration.md), [Replay and Recording](replay.md), and [Third-party Notices](../../THIRD_PARTY_NOTICES.md).
