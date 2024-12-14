import os


_REC_BUCKET = "not decide"

_CACHE_DIR = os.path.expanduser(os.path.join('~', "reclab"))



try:
    from .version import __version__
except ImportError:
    pass

__all__ = [
    "data",
    "datasets",
    "utils",
    "transforms",
    "models",
    "metrics"
]