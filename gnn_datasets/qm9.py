"""
Author: Shengli Jiang
Email: sjiang87@wisc.edu / shengli.jiang@anl.gov
"""

import os
from tensorflow.keras.utils import get_file

DATA_PATH = os.path.expanduser('~/.deephyper/datasets/qm9/')
DATASET_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'


def _download_data():
    """
    Load qm9 dataset
    Returns:

    """
    _ = get_file(
        'qm9.tar.gz', DATASET_URL,
        extract=True, cache_dir=DATA_PATH, cache_subdir=DATA_PATH
    )
    os.rename(DATA_PATH + 'gdb9.sdf', DATA_PATH + 'qm9.sdf')
    os.rename(DATA_PATH + 'gdb9.sdf.csv', DATA_PATH + 'qm9.sdf.csv')
    os.remove(DATA_PATH + 'qm9.tar.gz')


def test_download_data():
    _download_data()


if __name__ == "__main__":
    test_download_data()