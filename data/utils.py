"""data utils"""

def read_text(path, debug=None):
    " read in text data "
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