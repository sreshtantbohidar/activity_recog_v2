"""Fail loudly when a deploy drops a shipped feature.

On 2026-10-17 a patch built from an older baseline replaced ``app.py``, the
dashboard and ``swagger_docs.py``. The VRAM Monitor screen and the idle engine
recycle disappeared while their modules stayed on disk, so nothing crashed and
the loss was only noticed by looking at the UI.

These checks pin the wiring that made those features work. They run inside
``/unit-tests`` and in a plain ``python -m unittest`` run, so the same silent
overwrite fails a test instead of waiting for an operator to notice.

Source-only by design: no GPU, network, Flask or model import, so the suite
runs anywhere the source is checked out.
"""
from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parent
DASHBOARDS = ("templates/dashboard.html", "dashboard.html")


def read(name: str) -> str:
    return (BASE / name).read_text(encoding="utf-8", errors="replace")


def routes(source: str) -> dict:
    """Route path -> whether it carries a ``swag_from`` decorator."""
    found = {}
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        paths, documented = [], False
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            name = (decorator.func.id if isinstance(decorator.func, ast.Name)
                    else decorator.func.attr if isinstance(decorator.func, ast.Attribute)
                    else None)
            if name == "route" and decorator.args and isinstance(decorator.args[0], ast.Constant):
                paths.append(decorator.args[0].value)
            elif name == "swag_from":
                documented = True
        for path in paths:
            found[path] = documented
    return found


class MonitorFeatureTests(unittest.TestCase):
    """The Monitor tab and its read-only endpoint must stay wired together."""

    def setUp(self):
        self.app_source = read("app.py")
        self.api_routes = routes(self.app_source)

    def test_monitor_route_is_documented_and_uses_the_monitor_module(self):
        self.assertIn("/gpu/monitor", self.api_routes)
        self.assertTrue(self.api_routes["/gpu/monitor"], "/gpu/monitor lost its Swagger documentation")
        self.assertIn("from gpu_monitor import build_snapshot as gpu_snapshot", self.app_source)
        self.assertIn("gpu_snapshot(", self.app_source)
        self.assertIn("GPU_MONITOR_SWAGGER_DOC", read("swagger_docs.py"))

    def test_dashboard_keeps_the_monitor_tab_on_both_copies(self):
        markers = ('data-mode="monitor"', 'id="monitorPanel"', 'id="vramChart"',
                   'id="resourceLight"', "fetch('/gpu/monitor')", "startMonitor()")
        for name in DASHBOARDS:
            with self.subTest(dashboard=name):
                source = read(name)
                for marker in markers:
                    self.assertIn(marker, source, f"{marker} missing from {name}")

    def test_monitor_modules_and_their_tests_are_present(self):
        for name in ("gpu_monitor.py", "test_gpu_monitor.py",
                     "gpu_vram_gate.py", "test_vram_gate.py"):
            self.assertTrue((BASE / name).is_file(), f"{name} is missing")
        import gpu_monitor
        self.assertTrue(callable(gpu_monitor.build_snapshot))
        for name in ("test_gpu_monitor", "test_vram_gate", "test_vision_backend_vllm"):
            self.assertIn(f'"{name}"', self.app_source, f"{name} dropped from the unit-test suite")

    def test_monitor_polling_cannot_wake_the_gpu(self):
        # Polling the Monitor tab must stay a passive GET, or an open console
        # holds the engine awake and the idle release never happens.
        self.assertIn("_PASSIVE_GET_PREFIXES", self.app_source)
        self.assertIn('"/gpu/"', self.app_source)
        self.assertIn("for prefix in _PASSIVE_GET_PREFIXES", self.app_source)


class IdleRecycleTests(unittest.TestCase):
    """The idle engine recycle was lost in the same overwrite."""

    def test_recycle_is_wired_into_the_power_manager(self):
        source = read("app.py")
        for marker in ("recycle_after_idle_seconds", "recycle_vram_mib",
                       "engine_vram_fn=engine_memory_mib", "recycle_fn=_stop_idle_engine",
                       "ready_fn=_engine_ready", "release_idle_fn=_release_object_detector",
                       "probe_fn=engine_is_sleeping"):
            self.assertIn(marker, source, f"{marker} missing from app.py")

        from engine_power import EnginePowerManager
        parameters = inspect.signature(EnginePowerManager).parameters
        for name in ("recycle_after_idle_seconds", "recycle_vram_mib", "engine_vram_fn",
                     "recycle_fn", "ready_fn", "probe_fn", "release_idle_fn"):
            self.assertIn(name, parameters, f"EnginePowerManager lost its {name} hook")

        self.assertIn("gpu_vram_gate.py", read("entrypoint.sh"))


class VideoObjectDetectionTests(unittest.TestCase):
    """The newer feature must not be lost in the other direction either."""

    def test_endpoints_and_tab_survive(self):
        api_routes = routes(read("app.py"))
        for path in ("/detect-objects-video", "/object-videos/<path:filename>"):
            self.assertIn(path, api_routes)
            self.assertTrue(api_routes[path], f"{path} lost its Swagger documentation")
        for name in DASHBOARDS:
            self.assertIn('data-mode="objectsVideo"', read(name))


if __name__ == "__main__":
    unittest.main()
