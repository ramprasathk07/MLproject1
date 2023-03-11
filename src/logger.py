import logging
import logging.handlers
import os
import sys
from datetime import datetime

log_file=f"{datetime.now().strftime('%Y-%m-%d')}.log"
log_path=os.path.join(os.getcwd(),"logs",log_file)
os.makedirs(log_path,exist_ok=True)

log_file_path=os.path.join(log_path,log_file)

logging.basicConfig(
    filename=log_file_path,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s',
    level=logging.INFO
)

if __name__=="__main__":
    logger=logging.getLogger(__name__)
    logger.info("This is a test message")
    logger.error("This is a test error message")