#!/usr/bin/env bash
set -euo pipefail

PKG_ROOT="${
  RBNX_PACKAGE_ROOT:-$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
  )
}"

ROS_DISTRO="${ROS_DISTRO:-humble}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"

fail() {
  printf 'start error: %s\n' "$*" >&2
  exit 1
}

command -v rbnx >/dev/null 2>&1 \
  || fail "rbnx is not on PATH"

[[ -f "$ROS_SETUP" ]] \
  || fail "ROS setup not found: $ROS_SETUP"

# camera_adapter.py 和 arm_adapter.py 使用 ROS2。
# shellcheck disable=SC1090
source "$ROS_SETUP"

export RBNX_PACKAGE_ROOT="$PKG_ROOT"

export PYTHONPATH="$(
  rbnx path robonix-api
):$PKG_ROOT:${PYTHONPATH:-}"

cd "$PKG_ROOT"

exec python3 -m dataset_collect_skill.main
