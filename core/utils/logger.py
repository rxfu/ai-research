import csv
import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    log_file: Path, name: str = "research", log_level: int = logging.INFO
) -> logging.Logger:
    log_file.parent.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    # logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


class ResearchLogger:
    def __init__(
        self,
        output_dir: str | Path,
        log_dirname: str = "logs",
        metrics_filename: str = "metrics",
        events_filename: str = "events",
    ) -> None:
        self.log_dir = Path(output_dir) / log_dirname
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.log_dir / f"{metrics_filename}.csv"
        self.events_path = self.log_dir / f"{events_filename}.jsonl"

    @staticmethod
    def _timestamp():
        return datetime.now().isoformat()

    def log_event(self, event, payload=None) -> None:
        record = {
            "timestamp": self._timestamp(),
            "event": event,
            "payload": payload or {},
        }

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_metrics(self, row) -> None:
        row = dict(row)
        row.setdefault("timestamp", self._timestamp())

        exists = self.metrics_path.exists()

        with self.metrics_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))

            if not exists:
                writer.writeheader()

            writer.writerow(row)

        self.log_event("metrics", row)
