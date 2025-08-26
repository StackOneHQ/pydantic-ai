# Extensions for third-party integrations
try:
    from . import stackone  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError:
    pass  # stackone-ai not installed or Python < 3.11
