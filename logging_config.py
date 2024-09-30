import logging
from logging.handlers import RotatingFileHandler
import tempfile
import streamlit as st

DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')

@st.cache_resource
def setup_logging(max_bytes=5 * 1024 * 1024, backup_count=3):
    """
    Sets up logging with a rotating file handler.
    
    :param max_bytes: Maximum size of the log file before rotating (default is 5 MB).
    :param backup_count: Number of backup files to keep (default is 3).
    """
    logging.basicConfig(level=ANALYSIS_LEVEL)

    # Create a temporary file for logging
    with tempfile.NamedTemporaryFile(mode='a', delete=False) as temp_file:
        temp_logging_file_name = temp_file.name

    # Set up a rotating file handler
    file_handler = RotatingFileHandler(
        temp_logging_file_name, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(ANALYSIS_LEVEL)

    # Define the logging format
    logging_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(logging_formatter)

    # Get the root logger and add the handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    return temp_logging_file_name
