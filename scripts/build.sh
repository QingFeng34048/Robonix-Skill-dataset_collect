#!/usr/bin/env bash
set -euo pipefail

PKG="${
  RBNX_PACKAGE_ROOT:-$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
  )
}"

fail() {
  printf 'build error: %s\n' "$*" >&2
  exit 1
}

command -v rbnx >/dev/null 2>&1 \
  || fail "rbnx is not on PATH"

python3 - <<'PY'
import importlib.util

required = [
    "grpc_tools",
    "h5py",
    "numpy",
    "cv2",
]

missing = [
    name
    for name in required
    if importlib.util.find_spec(name) is None
]

if missing:
    raise SystemExit(
        "missing Python modules: "
        + ", ".join(missing)
    )
PY

FLAGS=(--mcp)

if [[ "${RBNX_BUILD_CLEAN:-0}" == "1" ]]; then
  FLAGS+=(--clean)
fi

rbnx codegen -p "$PKG" "${FLAGS[@]}"

cd "$PKG"

python3 -m compileall \
  -q dataset_collect_skill

if python3 -c \
  "import pytest" >/dev/null 2>&1; then
  python3 -m pytest -q
else
  echo "[build] pytest not installed; tests skipped"
fi

echo "[build] dataset collection skill completed"
