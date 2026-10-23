**English** · [简体中文](../zh/integration.md) · [Home](../../README.md)

# Custom Integrations

Choose the component to extend. Switching to a compatible service only requires changing its address.

| Goal | Entry point |
| --- | --- |
| Change the decision model | [Configuration](configuration.md#decision-model) |
| Add a navigation protocol | `contracts/vln.py` → `vln/` → `app/deployments/vln.py` |
| Add a robot | `contracts/agent_backend.py` → `robots/` → application assembly |
| Change scene requirements | [Configuration](configuration.md#scenes-and-prompts) |

Python paths above are relative to `src/navigation_agent/`. Run the commands below from the repository root using the prepared `nav-agent` environment.

## Navigation Model

1. Implement [VlnClient](../../src/navigation_agent/contracts/vln.py): accept timestamped JPEG frames and return waypoints with an explicit STOP signal. The client owns queuing, backpressure and stale-session isolation.
2. Follow [custom_vln.py](../../examples/custom_vln.py). Call `register_vln` before loading configuration to register the options model and client factory.
3. Select the adapter in the backend TOML:

   ```toml
   [vln]
   backend = "example"
   server_url = "ws://model-host:9900"
   [vln.options]
   horizon = 6
   ```

4. Use the same registration entry point in the Agent and the service that runs the client, then check connections.

Run the offline assembly example first:

```bash
PYTHONPATH=src conda run -n nav-agent python examples/custom_vln.py
```

Expect a client status and `horizon=6`. This example does not run inference. The native Go2 service currently supports only LightNav; registering a name in the Agent does not replace the robot-side protocol.

## Robot Backend

1. Implement the observation, turning, navigation, stopping and control-ownership interfaces in [RobotBackend](../../src/navigation_agent/contracts/agent_backend.py).
2. Follow [custom_backend.py](../../examples/custom_backend.py) to inject `NavigationTools` into `run_task`, or pass `backend_factory` to `StandaloneDeployment`.
3. For a TOML entry point, call `register_robot` before reading configuration. Register the `RobotAdapter` settings type and factory; see [factory.py](../../src/navigation_agent/app/deployments/factory.py).

Run the minimal offline example:

```bash
PYTHONPATH=src conda run -n nav-agent python examples/custom_backend.py
```

Expect `completed` and an output directory. The example uses fixed images and responses without connecting to hardware. For real integration, replace the backend and configure the decision model.

On completion, cancellation or failure, stop only motion owned by the current connection. Image, timestamp and pose in an observation must refer to the same captured frame; do not assemble them from separate queries.

## Optional Capabilities

- **Detection during motion:** implement `StreamingObservationBackend` for read-only capture and interruption requests tied to an action identity. A stale action must not interrupt a new action.
- **History continuation:** implement `configure_history` in the backend and `HistoryVlnClient.pause/resume` in the model client. Custom deployments without this capability can set `[harness].preserve_same_instruction_history = false` in the Agent TOML.
- **Measured trajectory:** return measured XY points in a shared coordinate frame through `motion.sampled_positions`. Do not substitute endpoint interpolation for measured positions.

## Validation

```bash
bash deploy/agent/run.sh --backend-config configs/mujoco-lightnav.toml --check-config
bash deploy/agent/run.sh --backend-config configs/mujoco-lightnav.toml --check
```

Replace the backend configuration with your own. The first command checks configuration only; the second checks connections and capabilities without running navigation. The built-in Go2 check also reads real observations. After both checks pass, validate an actual task. On-site operators perform physical motion tests.
