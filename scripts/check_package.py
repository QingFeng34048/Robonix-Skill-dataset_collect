#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pathlib import Path
from typing import Any

import yaml


REQUIRED_PACKAGE_FIELDS = {
    "name",
    "version",
    "description",
    "license",
}


def fail(messages: list[str], message: str) -> None:
    messages.append(message)


def load_yaml(path: Path, messages: list[str]) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        fail(messages, f"{path}: invalid YAML: {exc}")
        return {}
    if not isinstance(value, dict):
        fail(messages, f"{path}: YAML root must be a mapping")
        return {}
    return value


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    errors: list[str] = []

    manifest_path = root / "package_manifest.yaml"
    if not manifest_path.is_file():
        fail(errors, "missing package_manifest.yaml")
        manifest: dict[str, Any] = {}
    else:
        manifest = load_yaml(manifest_path, errors)

    package = manifest.get("package", {})
    if not isinstance(package, dict):
        fail(errors, "manifest package must be a mapping")
        package = {}
    for field in sorted(REQUIRED_PACKAGE_FIELDS):
        if not str(package.get(field, "")).strip():
            fail(errors, f"manifest package.{field} is required")
    for field in ("build", "start"):
        if not str(manifest.get(field, "")).strip():
            fail(errors, f"manifest {field} is required")

    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        fail(errors, "manifest capabilities must be a non-empty list")
        capabilities = []

    manifest_ids: set[str] = set()
    for index, item in enumerate(capabilities):
        if not isinstance(item, dict):
            fail(errors, f"capabilities[{index}] must be a mapping")
            continue
        capability_id = str(item.get("name", "")).strip()
        capability_path = root / str(item.get("path", ""))
        if not capability_id:
            fail(errors, f"capabilities[{index}].name is required")
            continue
        if capability_id in manifest_ids:
            fail(errors, f"duplicate capability id: {capability_id}")
        manifest_ids.add(capability_id)
        if not capability_path.is_file():
            fail(errors, f"missing capability file: {capability_path}")
            continue
        try:
            contract = tomllib.loads(
                capability_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            fail(errors, f"{capability_path}: invalid TOML: {exc}")
            continue
        if contract.get("id") != capability_id:
            fail(
                errors,
                f"{capability_path}: id does not match manifest name",
            )
        if contract.get("transport") != "mcp":
            fail(errors, f"{capability_path}: transport must be mcp")
        if contract.get("mode") != "rpc":
            fail(errors, f"{capability_path}: mode must be rpc")
        idl_rel = contract.get("mcp", {}).get("idl")
        idl_path = capability_path.parent / str(idl_rel or "")
        if not idl_rel or not idl_path.is_file():
            fail(errors, f"{capability_path}: missing referenced MCP IDL")

    base_ids = {
        value
        for value in manifest_ids
        if not value.endswith("/status") and not value.endswith("/cancel")
    }
    for base in base_ids:
        if f"{base}/status" not in manifest_ids:
            fail(errors, f"long task missing status capability: {base}")
        if f"{base}/cancel" not in manifest_ids:
            fail(errors, f"long task missing cancel capability: {base}")

    main_path = root / "robonix_openvla_skill/main.py"
    main_text = (
        main_path.read_text(encoding="utf-8")
        if main_path.is_file()
        else ""
    )
    if not main_text:
        fail(errors, "missing or empty robonix_openvla_skill/main.py")
    decorated = set(
        re.findall(r'@skill\.mcp(?:_tool)?\(\s*"([^"]+)"', main_text)
    )
    missing_handlers = manifest_ids - decorated
    if missing_handlers:
        fail(
            errors,
            "main.py missing MCP handler(s): "
            + ", ".join(sorted(missing_handlers)),
        )

    required_nonempty = [
        "config.spec",
        "scripts/build.sh",
        "scripts/start.sh",
        "robonix_openvla_skill/runtime_config.py",
        "robonix_openvla_skill/task_registry.py",
        "robonix_openvla_skill/task_manager.py",
        "robonix_openvla_skill/safety.py",
        "robonix_openvla_skill/units.py",
        "robonix_openvla_skill/vla_client.py",
        "configs/runtime/tasks.yaml",
    ]
    for relative in required_nonempty:
        path = root / relative
        if not path.is_file() or not path.read_text(
            encoding="utf-8"
        ).strip():
            fail(errors, f"missing or empty required file: {relative}")

    if errors:
        print("Package check failed:")
        for item in errors:
            print(f"  - {item}")
        return 1
    print(
        f"Package check passed: {package.get('name')} "
        f"{package.get('version')} ({len(manifest_ids)} capabilities)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
