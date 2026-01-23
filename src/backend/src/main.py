import os
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    # Change to project root (one level up from src/)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)
    os.chdir(PROJECT_ROOT)

    if len(sys.argv) == 1:
        logger.info("Starting control process.....")
        from control_process.main import start_control_process

        start_control_process()
    elif sys.argv[1] == "cv":
        logger.info("Starting cv process.....")
        from cv_process.main import start_cv_process

        start_cv_process()
    elif sys.argv[1] == "live_video":
        logger.info("Starting live video process.....")
        pass
    elif sys.argv[1] == "transcode":
        logger.info("Starting transcode process.....")
        pass
    else:
        logger.warning("Unknown command")
        sys.exit(1)
