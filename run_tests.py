"""Small test runner that works even when pytest is not installed."""
import importlib
import inspect
import traceback

modules = ["tests.test_rocket3d"]
passed = failed = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_"):
            continue
        try:
            fn()
            print(f"PASS {module_name}.{name}")
            passed += 1
        except Exception:
            print(f"FAIL {module_name}.{name}")
            traceback.print_exc()
            failed += 1
print(f"{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)
