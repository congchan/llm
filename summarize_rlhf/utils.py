"""
coding=utf-8
"""

import json
import logging

from yaml import Loader, dump, load


def load_yaml(config):
    with open(config) as stream:
        data = load(stream, Loader=Loader)
    return data

def dump_yaml(data, file_name):
    with open(f"{file_name}", "w") as stream:
        dump(data, stream)

def dump_json(data, file_name):
    with open(f"{file_name}", "w") as stream:
        json.dump(data, stream)

def get_logger(name, to_file, level=logging.INFO):
    logger = logging.getLogger(name=name)
    logger.setLevel(level=level)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')

    handler = logging.FileHandler(filename=to_file)
    handler.setLevel(level=level)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(level=level)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger