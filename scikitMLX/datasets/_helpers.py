import shutil

from os import environ, makedirs
from os.path import expanduser, join


def get_data_dir(data_dir):
    if data_dir is None:
        data_dir = environ.get("SKMX_DATA_DIR", join("~", "skmx_data_cache"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_dir(data_dir):
    data_dir = get_data_dir(data_dir)
    shutil.rmtree(data_dir)
