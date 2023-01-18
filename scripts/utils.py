import bz2
import os
import _pickle as cPickle
import shutil


def compress_pickle(path, data):
    """Compress and save a pickled object to a file"""
    with bz2.BZ2File(path, "w") as f:
        cPickle.dump(data, f)


def decompress_pickle(file):
    """Load and decompress a pickled object from a file"""
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


# credits: https://stackoverflow.com/questions/33282647/python-shutil-copy-if-i-have-a-duplicate-file-will-it-copy-to-new-location
def safe_copy(file_path, out_dir, dst=None):
    """
    Safely copy a file to the specified directory. If a file with the same name already
    exists, the copied file name is altered to preserve both.

    file_path: Path to the file to copy.
    out_dir: Directory to copy the file into.
    dst: New name for the copied file. If None, use the name of the original file.
    """
    name = dst or os.path.basename(file_path)
    if not os.path.exists(os.path.join(out_dir, name)):
        shutil.copy(file_path, os.path.join(out_dir, name))
    else:
        base, extension = os.path.splitext(name)
        i = 1
        while os.path.exists(
            os.path.join(out_dir, "{}_{}{}".format(base, i, extension))
        ):
            i += 1
        shutil.copy(
            file_path, os.path.join(out_dir, "{}_{}{}".format(base, i, extension))
        )
