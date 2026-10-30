"""A four-block convolutional classifier trained from scratch."""

from importlib import import_module

__all__ = ["CustomCNN", "build_custom_cnn", "parameter_counts"]


def __getattr__(name):
    # Plotting commands need no torch runtime (or GPU memory).
    if name in __all__:
        return getattr(import_module(".model", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
