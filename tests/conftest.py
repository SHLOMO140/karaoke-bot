"""Pytest configuration: mock unavailable heavy dependencies."""
import sys
import types


def _make_mock_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# Mock heavy/optional dependencies that are not installed in the test environment
_MOCK_MODULES = [
    "yt_dlp",
    "librosa",
    "numpy",
    "torch",
    "whisper",
    "telegram",
    "telegram.ext",
]

for _mod_name in _MOCK_MODULES:
    if _mod_name not in sys.modules:
        _make_mock_module(_mod_name)
