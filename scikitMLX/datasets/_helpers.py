import shutil
import hashlib

from os import environ, makedirs
from collections import namedtuple
from os.path import expanduser, join
from urllib.request import urlretrieve

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])


def get_data_dir(data_dir):
    if data_dir is None:
        data_dir = environ.get("SKMX_DATA_DIR", join("~", "skmx_data_cache"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home


def clear_data_dir(data_dir):
    data_dir = get_data_dir(data_dir)
    shutil.rmtree(data_dir)


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def _fetch_remote(remote, dirname=None):
    file_path = remote.filename if dirname is None else join(dirname, remote.filename)
    urlretrieve(remote.url, file_path)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise OSError(
            "{} has an SHA256 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(file_path, checksum, remote.checksum)
        )
    return file_path
