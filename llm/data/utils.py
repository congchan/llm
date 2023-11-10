"""data utils"""

import json


def read_text(path, debug=None):
    """read in text data"""
    data = []
    line_num = 1
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line == '\n':
                continue
            if debug and debug < line_num:
                break
            line_num += 1
            data.append(line.strip())
    return data


def read_json_line(file):
    tmp_ls = open(file, 'r', encoding='utf-8').readlines()
    tmp_ls = [json.loads(_.strip()) for _ in tmp_ls]
    return tmp_ls


def logging_rank0(*args):
    if is_initialized():
        if get_rank() == 0:
            logging.info(*args)
    else:
        print(*args)
