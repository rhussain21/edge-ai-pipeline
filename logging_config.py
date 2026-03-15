"""Logging Configuration for AI Industry Signals."""

import logging
import os
from datetime import datetime

def setup_logging(level=logging.DEBUG, log_file=None):
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/system_{timestamp}.log"
    
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
            logging.FileHandler(log_file.replace('.log', '_errors.log'), mode='w')
        ]
    )
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler) and 'errors' in handler.baseFilename:
            handler.setLevel(logging.ERROR)
    
    loggers_to_debug = [
        'agents',
        'agents.factory', 
        'agents.sql_agent',
        'agents.vector_agent',
        'agents.web_agent',
        'agents.router_agent',
        'db_vector',
        'db_relational',
        'tools',
        'tools.web_search',
        'etl',
        'discovery',
    ]
    
    for logger_name in loggers_to_debug:
        logging.getLogger(logger_name).setLevel(level)
    
    print(f"Logging configured:")
    print(f"   Level: {logging.getLevelName(level)}")
    print(f"   Console: YES")
    print(f"   File: {log_file}")
    print(f"   Errors: {log_file.replace('.log', '_errors.log')}")

def setup_quiet_logging():
    setup_logging(level=logging.INFO, log_file="logs/production.log")

def setup_debug_logging():
    setup_logging(level=logging.DEBUG, log_file="logs/debug.log")

def setup_silent_logging():
    setup_logging(level=logging.ERROR, log_file="logs/errors.log")

if __name__ == "__main__":
    print("\n1. Debug logging:")
    setup_debug_logging()
    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.error("This is an error message")
    
    print("\n2. Quiet logging:")
    setup_quiet_logging()
    logging.info("Quiet info message")
    logging.error("Quiet error message")
