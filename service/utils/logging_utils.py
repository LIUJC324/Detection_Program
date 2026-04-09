from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

import yaml


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_CONFIGURED = False
_LOG_PATHS: Dict[str, str] = {}


def _resolve_log_config(config_path: str | Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fp:
        service_cfg = yaml.safe_load(fp)["service"]

    project_root = Path(config_path).resolve().parents[1]
    log_dir = Path(service_cfg.get("log_dir", "./outputs/service_logs"))
    if not log_dir.is_absolute():
        log_dir = (project_root / log_dir).resolve()

    return {
        "log_dir": log_dir,
        "log_level": str(service_cfg.get("log_level", "INFO")).upper(),
        "log_max_bytes": int(service_cfg.get("log_max_bytes", 10 * 1024 * 1024)),
        "log_backup_count": int(service_cfg.get("log_backup_count", 5)),
    }


def _build_handler(path: Path, level: int, max_bytes: int, backup_count: int, key: str) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    setattr(handler, "_rgbt_handler_key", key)
    return handler


def _ensure_handler(logger: logging.Logger, handler: logging.Handler) -> None:
    handler_key = getattr(handler, "_rgbt_handler_key", None)
    for existing in logger.handlers:
        if getattr(existing, "_rgbt_handler_key", None) == handler_key:
            return
    logger.addHandler(handler)


def configure_service_logging(config_path: str | Path) -> Dict[str, str]:
    global _CONFIGURED
    global _LOG_PATHS

    if _CONFIGURED:
        return _LOG_PATHS

    resolved = _resolve_log_config(config_path)
    log_dir: Path = resolved["log_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    level_name = resolved["log_level"]
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = resolved["log_max_bytes"]
    backup_count = resolved["log_backup_count"]

    service_log = log_dir / "service.log"
    session_log = log_dir / "session.log"
    error_log = log_dir / "error.log"

    service_handler = _build_handler(service_log, level, max_bytes, backup_count, "service")
    session_handler = _build_handler(session_log, level, max_bytes, backup_count, "session")
    error_handler = _build_handler(error_log, logging.ERROR, max_bytes, backup_count, "error")

    for logger_name in ("rgbt.api", "rgbt.session", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = False
        _ensure_handler(logger, service_handler)
        _ensure_handler(logger, error_handler)
        if logger_name == "rgbt.session":
            _ensure_handler(logger, session_handler)

    _LOG_PATHS = {
        "log_dir": str(log_dir),
        "service_log": str(service_log),
        "session_log": str(session_log),
        "error_log": str(error_log),
    }
    _CONFIGURED = True
    return _LOG_PATHS


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"rgbt.{name}")

