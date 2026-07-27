import os
import importlib.util

if os.name == "nt":
    # Explicitly add specific paths known to contain required C++ runtimes
    known_paths = [
        r"C:\Windows\System32",
        r"C:\Program Files\Git\mingw64\bin",
        r"C:\Program Files\Git\usr\bin",
    ]
    for p in known_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except (OSError, AttributeError):
                pass

    # Also add the package's local pyvrp.libs directory if it exists
    try:
        spec = importlib.util.find_spec("pyvrp")
        if spec and spec.origin:
            pyvrp_dir = os.path.dirname(spec.origin)
            libs_dir = os.path.join(os.path.dirname(pyvrp_dir), "pyvrp.libs")
            if os.path.exists(libs_dir):
                os.add_dll_directory(libs_dir)
    except Exception:
        pass


from .read import read as read
from .write import write as write
