from scripts.logger import setup_logger
from pathlib import Path

def test_logger_creates_log_file(tmp_path, monkeypatch):

    # Rediriger les logs vers un dossier temporaire
    monkeypatch.chdir(tmp_path)

    logger = setup_logger("test_logger")

    logger.info("Test log")

    log_dir = Path("logs")
    assert log_dir.exists()

    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0
