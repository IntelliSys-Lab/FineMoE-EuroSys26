"""FineMoE inference and expert-map policies."""

__version__ = "0.0.1"
__all__ = ["MoE", "OffloadEngine", "FineMoEConfig"]


def __getattr__(name):
    if name == "MoE":
        from .entrypoints.big_modeling import MoE
        return MoE
    if name == "OffloadEngine":
        from .runtime.model_offload import OffloadEngine
        return OffloadEngine
    if name == "FineMoEConfig":
        from .runtime.config import FineMoEConfig
        return FineMoEConfig
    raise AttributeError(name)
