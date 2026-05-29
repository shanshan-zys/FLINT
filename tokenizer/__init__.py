from .absolute import AbsoluteTokenizer
from .relative import RelativeTokenizer
from .polar import PolarTokenizer
from .base import BaseCoordinateTokenizer


def build_tokenizer(config: dict) -> BaseCoordinateTokenizer:
    """根据配置构建tokenizer"""
    mode = config["tokenizer"]["mode"]
    resolution = tuple(config["data"]["resolution"])

    if mode == "absolute":
        cfg = config["tokenizer"]["absolute"]
        return AbsoluteTokenizer(resolution=resolution, bin_size=cfg["bin_size"])
    elif mode == "relative":
        cfg = config["tokenizer"]["relative"]
        return RelativeTokenizer(
            resolution=resolution,
            max_displacement=cfg["max_displacement"],
            bin_size=cfg["bin_size"],
        )
    elif mode == "polar":
        cfg = config["tokenizer"]["polar"]
        return PolarTokenizer(
            resolution=resolution,
            max_distance=cfg["max_distance"],
            dist_bins=cfg["dist_bins"],
            angle_bins=cfg["angle_bins"],
        )
    else:
        raise ValueError(f"Unknown tokenizer mode: {mode}")


__all__ = [
    "AbsoluteTokenizer",
    "RelativeTokenizer",
    "PolarTokenizer",
    "BaseCoordinateTokenizer",
    "build_tokenizer",
]
