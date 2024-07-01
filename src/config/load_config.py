"""
Module for loading configuration from a YAML file.

This module provides a function `load_config` to load configuration settings
from a specified YAML file into a dictionary. If the file does not exist, 
it returns an empty dictionary.


"""
import os
import yaml


def load_config(config_path="config.yaml"):
    """
    Load the YAML configuration from the specified path.
    
    Parameters:
    config_path (str): The path to the configuration file. Default is "config.yaml".
    
    Returns:
    dict: The loaded configuration as a dictionary. Returns an
    empty dictionary if the file does not exist.
    """
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    return config
