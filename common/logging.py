import logging
import sys


def setup_logging(level=logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        root.addHandler(handler)

    for name in ["pycocotools", "PIL", "matplotlib", "fvcore"]:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
