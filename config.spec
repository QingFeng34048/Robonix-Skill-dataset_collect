# Public runtime configuration reference.
# Robot-specific values should be supplied by the deployment manifest.
config:
  # string; absolute HTTP(S) URL; default http://127.0.0.1:8777/act
  vla_server_url: http://127.0.0.1:8777/act

  # string; Robonix provider ids; required and non-empty
  camera_provider_id: front_camera
  arm_provider_id: piper_arm

  # string/path; relative paths resolve from RBNX_PACKAGE_ROOT
  task_config_path: configs/runtime/tasks.yaml

  # number, Hz; > 0
  control_hz: 10.0
  # number, seconds; > 0
  request_timeout_s: 10.0
  task_timeout_s: 120.0
  dependency_wait_timeout_s: 5.0
  max_image_age_s: 1.0
  max_state_age_s: 0.5

  # integer [1, 100]
  jpeg_quality: 90
  # integer >= 1
  max_concurrent_runs: 1
  # boolean; only enable when the inference server implements GET /health
  require_vla_healthcheck: false

  # boolean; keep enabled for real hardware
  enable_safety_filter: true
  # arrays of six radians; REQUIRED when safety is enabled; use real URDF limits
  joint_min_rad: []
  joint_max_rad: []
  # number, radians per control command; > 0
  default_max_delta_rad: 0.04

  # numbers, metres; min < max; verify against the deployed gripper
  gripper_min_m: 0.0
  gripper_max_m: 0.08
  # enum: normalized_0_1 | absolute_m
  model_gripper_mode: normalized_0_1

  # six unique strings and one distinct gripper joint name
  joint_names: [joint1, joint2, joint3, joint4, joint5, joint6]
  gripper_name: gripper

  # Piper JointState command metadata
  piper_command_metadata: true
  # percent [1, 100]
  joint_velocity_pct: 30.0
  # number [0.5, 3.0]
  gripper_effort: 1.0
  hold_position_on_cancel: true
