import logging.config
import os 

LOGGING_CONFIG = {
    "version" : 1,
    "disable_existing_loggers" : False,
    "formatters" : {
        "standard" : {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(module)s %(message)s"
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file_json": {
            "level": "INFO",
            "formatter": "json",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "logs/data_pipeline.jsonl",
            "maxBytes": 10485760,
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file_json"],
            "level": "INFO",
            "propagate": True
        },

        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "supabase": {"level": "WARNING"},
        "mlflow": {"level": "WARNING"},
    }
}

def setup_logging():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    LOGGING_CONFIG["handlers"]["console"]["formatter"] = "json"
    logging.config.dictConfig(LOGGING_CONFIG)