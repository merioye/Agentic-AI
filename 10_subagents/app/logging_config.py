"""
Structured (JSON) logging setup.

Why JSON logs: in production these get shipped to a log aggregator
(Datadog, CloudWatch, etc.) where you search/filter by field. Free-text
logs like "billing agent failed for order 123" are hard to query at
scale; {"event": "subagent_failed", "subagent": "billing", "order_id": "123"}
is trivially filterable.
"""
import logging
import json
import sys
from app.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
        }
        # Anything passed via logger.info("msg", extra={...}) shows up as
        # attributes on the record — merge it into the JSON payload.
        standard_keys = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())
        for key, value in record.__dict__.items():
            if key not in standard_keys and key != "message":
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    root.handlers = [handler]

    # Quiet down noisy third-party loggers so our own events aren't buried.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


logger = logging.getLogger("support_platform")