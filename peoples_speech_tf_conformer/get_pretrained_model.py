
import os

def get_pretrained_model_path():
    return os.path.join(os.path.dirname(__file__), "pretrained_subword_conformer", "latest.h5")
