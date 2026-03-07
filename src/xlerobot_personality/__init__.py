"""XLeRobot personality starter package."""

from .config import AppConfig, load_config

__all__ = ["AppConfig", "load_config", "XLERobotHead", "XLERobotHeadConfig"]


def __getattr__(name: str):
    if name in {"XLERobotHead", "XLERobotHeadConfig"}:
        from .xlerobot_head import XLERobotHead, XLERobotHeadConfig

        exports = {
            "XLERobotHead": XLERobotHead,
            "XLERobotHeadConfig": XLERobotHeadConfig,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
