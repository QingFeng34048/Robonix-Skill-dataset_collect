#!/usr/bin/env bash
set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

fail() {
  printf 'build error: %s\n' "$*" >&2
  exit 1
}

command -v rbnx >/dev/null 2>&1 || fail "rbnx is not on PATH"
command -v colcon >/dev/null 2>&1 || fail "colcon is not on PATH"
[[ -f "$ROS_SETUP" ]] || fail "ROS setup not found: $ROS_SETUP"

# shellcheck disable=SC1090
source "$ROS_SETUP"

python3 - <<'PY' || {
  echo "Install runtime dependencies first: python3 -m pip install -e '.[dev]'" >&2
  exit 1
}
import importlib
required = ["grpc_tools", "numpy", "requests", "yaml", "cv2"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing Python modules: " + ", ".join(missing))
PY

python3 "$PKG_ROOT/scripts/check_package.py" "$PKG_ROOT"
python3 -m compileall -q "$PKG_ROOT/robonix_openvla_skill"

if [[ "${RBNX_CLEAN_BUILD:-0}" == "1" ]]; then
  rm -rf "$PKG_ROOT/rbnx-build"
fi

rbnx codegen -p "$PKG_ROOT" --mcp --ros2

IDL_BUILD="$PKG_ROOT/rbnx-build/codegen/ros2_idl"
[[ -d "$IDL_BUILD" ]] || fail "generated ROS2 IDL directory not found"

(
  cd "$IDL_BUILD"
  colcon build
)

GENERATED_SETUP="$IDL_BUILD/install/setup.bash"
[[ -f "$GENERATED_SETUP" ]] || fail "generated ROS setup is missing"

if [[ "${RBNX_RUN_TESTS:-1}" == "1" ]]; then
  (
    cd "$PKG_ROOT"
    python3 -m pytest -q
  )
fi

echo "Robonix package build completed."
