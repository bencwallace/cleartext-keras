# IMPORTANT: DO NOT MOVE THIS FILE FROM ${PROJ_ROOT}/cleartext/utils/
from pathlib import Path


def get_proj_root():
    return Path(__file__).parent.parent.parent
