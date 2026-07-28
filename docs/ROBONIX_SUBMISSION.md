# Robonix submission guide

## What this package exposes

The package declares one asynchronous skill and the required sibling tools:

- `robonix/skill/openvla_oft/execute`
- `robonix/skill/openvla_oft/execute/status`
- `robonix/skill/openvla_oft/execute/cancel`

`execute` returns a stable `run_id`; callers poll `status` and can call
`cancel`.

## Before using real hardware

1. Copy `configs/runtime/openvla_oft_skill.example.yaml` into the deployment
   repository.
2. Set `camera_provider_id` and `arm_provider_id` to the real Robonix
   provider instance names.
3. Fill `joint_min_rad` and `joint_max_rad` from the actual URDF or driver.
   Startup intentionally fails if safety is enabled and these values are
   missing.
4. Verify `gripper_min_m`, `gripper_max_m`, and `model_gripper_mode` against
   the inference server and arm driver.
5. Verify the VLA server URL. Enable `require_vla_healthcheck` only when the
   server implements `GET /health`.

## Validation commands

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/check_package.py .
pytest -q

rbnx validate .
rbnx build -p .
```

Then add the deployment fragment and validate the deployment repository:

```bash
rbnx validate /path/to/deployment-repository
rbnx build /path/to/deployment-repository
rbnx up /path/to/deployment-repository
rbnx caps
rbnx tools
```

## Submission checklist

- Package manifest metadata is accurate.
- The repository license file matches `MulanPSL-2.0`.
- Maintainer contact is current.
- No model weights, datasets, credentials, robot IPs, or machine-local paths
  are committed.
- All three long-task contracts are discoverable.
- Build and start scripts are executable.
- Unit tests and `scripts/check_package.py` pass.
- `rbnx validate .` and `rbnx build -p .` pass in the intended ROS2/Robonix
  environment.
- A real-robot dry run has verified cancel, timeout, stale camera/state
  handling, and joint/gripper limits.
