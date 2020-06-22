"""
Author: Shengli Jiang
Email: sjiang87@wisc.edu / shengli.jiang@anl.gov
Adapted from Spektral tf1
https://github.com/danielegrattarola/spektral/tree/tf1
"""

import os
from tensorflow.keras.utils import get_file

DATA_PATH = os.path.expanduser('~/.deephyper/datasets/qm9/')
DATASET_URL = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz'
NODE_FEATURES = ['atomic_num', 'charge', 'coords', 'iso']
EDGE_FEATURES = ['type', 'stereo']


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


def load_data(nf_keys=None, ef_keys=None):
    """
    Loads the QM9 chemical data set of small molecules.

    Nodes represent heavy atoms (hydrogens are discarded), edges represent
    chemical bonds.

    The node features represent the chemical properties of each atom, and are
    loaded according to the `nf_keys` argument.
    See `deephyper.datasets.gnndataset.qm9.NODE_FEATURES` for possible node features, and
    see [this link](http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
    for the meaning of each property. Usually, it is sufficient to load the
    atomic number.

    The edge features represent the type and stereoscopy of each chemical bond
    between two atoms.
    See `deephyper.datasets.genn_datasets.qm9.EDGE_FEATURES` for possible edge features, and
    see [this link](http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx)
    for the meaning of each property. Usually, it is sufficient to load the
    type of bond.
    Args:
        nf_keys: list or str, node node features to return (see `qm9.NODE_FEATURES` for available features);
        ef_keys: list or str, edge features to return (see `qm9.EDGE_FEATURES` for available features);

    Returns:
        the adjacency matrix, node features, edge features, and a Pandas dataframe containing labels;
    """


def test_download_data():
    _download_data()


if __name__ == "__main__":
    test_download_data()