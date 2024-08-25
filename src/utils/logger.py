import sys
from loguru import logger

def setup_logger():
    logger.remove()  # Remove default handler
    logger.add(sys.stderr, level="INFO")  # Add stderr handler
    logger.add("logs/app.log", rotation="500 MB", level="DEBUG")  # Add file handler

    return logger

app_logger = setup_logger()