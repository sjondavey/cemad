import sys
import os

import logging
from logging.handlers import RotatingFileHandler
import tempfile

# Track if logging is already set up and store the log file name
logging_initialized = False
global_logging_file_name = None

DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')

# ChatGPT suggests not setting this up as a cache_resource. It recommends just letting logging deal with it
def setup_logging(max_bytes=5 * 1024 * 1024, backup_count=3):
    """
    Sets up logging with a rotating file handler only once and returns the log file name.
    
    :param max_bytes: Maximum size of the log file before rotating (default is 5 MB).
    :param backup_count: Number of backup files to keep (default is 3).
    """
    global logging_initialized
    global global_logging_file_name

    if logging_initialized:
        # If logging is already set up, return the existing log file name
        return global_logging_file_name

    logging.basicConfig(level=ANALYSIS_LEVEL)

    # Create a file for logging
    log_directory = '/tmp/session_logs'
    os.makedirs(log_directory, exist_ok=True)        
    # Generate a unique filename for each session
    global_logging_file_name = os.path.join(log_directory, "app_log_data.txt")
    # Clear the contents of the file if it exists
    with open(global_logging_file_name, 'w') as file:
        pass  # Opening in 'w' mode clears the file; no need to write anything

    # Set up a rotating file handler
    file_handler = RotatingFileHandler(
        global_logging_file_name, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(ANALYSIS_LEVEL)

    # Define the logging format
    logging_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(logging_formatter)

    # Get the root logger and add the handler
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Mark logging as initialized
    logging_initialized = True

    return global_logging_file_name
