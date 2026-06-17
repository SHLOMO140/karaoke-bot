"""Pytest configuration: mock unavailable heavy dependencies."""
import importlib.util
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
    if _mod_name in sys.modules:
        continue
    if importlib.util.find_spec(_mod_name) is not None:
        continue
    if _mod_name not in sys.modules:
        _make_mock_module(_mod_name)


if "telegram" in sys.modules:
    telegram_mod = sys.modules["telegram"]

    class _InlineKeyboardButton:
        def __init__(self, text, callback_data=None):
            self.text = text
            self.callback_data = callback_data

    class _InlineKeyboardMarkup:
        def __init__(self, inline_keyboard):
            self.inline_keyboard = inline_keyboard

    class _Update:
        pass

    telegram_mod.InlineKeyboardButton = _InlineKeyboardButton
    telegram_mod.InlineKeyboardMarkup = _InlineKeyboardMarkup
    telegram_mod.Update = _Update


if "telegram.ext" in sys.modules:
    telegram_ext_mod = sys.modules["telegram.ext"]

    class _ApplicationBuilder:
        def token(self, *_args, **_kwargs):
            return self

        def build(self):
            return self

    class _Handler:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _ContextTypes:
        DEFAULT_TYPE = object

    class _Filters:
        def __init__(self):
            self.TEXT = self
            self.COMMAND = self
            self.Document = self

        def __and__(self, other):
            return self

        def __invert__(self):
            return self

        def __getattr__(self, _name):
            return self

    telegram_ext_mod.ApplicationBuilder = _ApplicationBuilder
    telegram_ext_mod.CallbackQueryHandler = _Handler
    telegram_ext_mod.CommandHandler = _Handler
    telegram_ext_mod.ContextTypes = _ContextTypes
    telegram_ext_mod.MessageHandler = _Handler
    telegram_ext_mod.filters = _Filters()
