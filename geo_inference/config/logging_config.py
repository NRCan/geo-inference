import logging.config
import yaml
from datetime import datetime
CONFIG_DIR = "geo_inference/config/log_config.yaml"
LOG_DIR = "geo_inference/logs"
timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
logfilename =  f"{LOG_DIR}/{timestamp}.log"

with open(CONFIG_DIR, "r") as f:
    config = yaml.safe_load(f.read())
    config["handlers"]["file"]["filename"] = logfilename    
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)