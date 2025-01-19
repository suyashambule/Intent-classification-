import os 
import yaml
from src.ic.logging import logging
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path


@ensure_annotations
def read_yaml(path_to_yaml: Path)->ConfigBox:
    try:
        with open(path_to_yaml) as yaml_file:
            contents=yaml.safe_load(yaml_file)
            return ConfigBox(contents)
    except BoxValueError:
        raise ValueError("yaml file is empty ")
@ensure_annotations
def create_directories(path_to_directories:list,verbose=True):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
    if verbose:
        logging.info(f'created directory at {path}')
