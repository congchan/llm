"""Test command line interface."""

from llm.utils import run_cmd


def test_build():
    cmd = 'pip3 install --no-cache-dir -e ".[data]"'
    ret = run_cmd(cmd)
    if ret != 0:
        return


if __name__ == "__main__":
    test_build()
