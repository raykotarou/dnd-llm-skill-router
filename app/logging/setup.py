from pathlib import Path

from loguru import logger

from app.config.settings import LoggingSettings


def _resolve_log_path(log_file: str) -> Path:
    path = Path(log_file)
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def setup_logging(settings: LoggingSettings) -> None:
    logger.remove()
    logger.add(
        sink=lambda message: print(message, end=""),
        level=settings.level,
        backtrace=settings.debug_full_payload,
        diagnose=settings.debug_full_payload,
    )

    log_path = _resolve_log_path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        level=settings.level,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        backtrace=settings.debug_full_payload,
        diagnose=settings.debug_full_payload,
    )
