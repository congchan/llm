import os


def run_cmd(cmd: str):
    """Run a bash command."""
    print(cmd)
    return os.system(cmd)
