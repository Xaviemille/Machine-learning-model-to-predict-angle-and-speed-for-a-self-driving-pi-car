import os
import pandas as pd
from pathlib import Path

# Paths 
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR   = os.path.join(REPO_DIR, "data")
TRAIN_CSV    = os.path.join(DATA_DIR, "train.csv")        
COMBINED_CSV = os.path.join(DATA_DIR, "combined.csv")    
CACHE_PATH   = Path(os.path.join(DATA_DIR, "combined_valid_ids.csv"))
NEW_IMG_DIR  = Path(os.path.join(DATA_DIR, "collected_images"))

# Config 
