def __getattr__(name):
    if name == "OffloadEngine":
        from .model_offload import OffloadEngine
        return OffloadEngine
    raise AttributeError(name)
