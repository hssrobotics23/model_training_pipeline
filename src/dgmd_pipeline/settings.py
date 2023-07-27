"""Configuration settings for Model Pipeline"""

import os

from dotenv import load_dotenv

load_dotenv()

# Define IO configurations:
PIPELINE_DATA_DIR = os.environ.get("DATA_DIR", "./data")
AWS_DATA_BUCKET = os.environ.get("AWS_DATA_BUCKET", "dgmd-s17-assets")
AWS_DATA_PATH = os.environ.get("AWS_DATA_PATH", "train/downloaded-images/")

MAX_DL_FILES = os.environ.get("MAX_DL_FILES", 300)
NUM_SHARDS = os.environ.get("NUM_SHARDS", 10)

# Define ML configurations:
VALIDATION_PCNT = os.environ.get("VALIDATION_PCNT", 0.2)
BATCH_SIZE = os.environ.get("BATCH_SIZE", 128)
IMAGE_WIDTH = os.environ.get("IMAGE_WIDTH", 256)
IMAGE_HEIGHT = os.environ.get("IMAGE_HEIGHT", 192)

# Define ML Hyperparameters:
EPOCHS = os.environ.get("EPOCHS", 2)
# EPOCHS = os.environ.get("EPOCHS", 15)
DECAY_RATE = os.environ.get("DECAY_RATE", 0.5)
LEARNING_RATE = os.environ.get("LEARNING_RATE", 0.01)

MODEL_NAME = os.environ.get("MODEL_NAME", "dgmd_model")
