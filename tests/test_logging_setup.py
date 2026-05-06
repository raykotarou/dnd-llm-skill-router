from pathlib import Path

from loguru import logger

from app.config.settings import LoggingSettings
from app.logging.setup import setup_logging


def test_setup_logging_writes_to_console_and_file(tmp_path: Path, capsys) -> None:
    log_file = tmp_path / "router.log"
    settings = LoggingSettings(
        level="INFO",
        log_file=str(log_file),
        debug_full_payload=False,
    )

    setup_logging(settings)
    logger.info("logging smoke test")
    logger.complete()

    captured = capsys.readouterr()
    assert "logging smoke test" in captured.out
    assert "logging smoke test" in log_file.read_text(encoding="utf-8")
