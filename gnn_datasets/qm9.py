"""
Author: Shengli Jiang
Email: sjiang87@wisc.edu / shengli.jiang@anl.gov
Adapted from Spektral tf1
https://github.com/danielegrattarola/spektral/tree/tf1
"""

import os
from tensorflow.keras.utils import get_file
from spektral.utils.io import load_csv, load_sdf
from spektral.chem import sdf_to_nx
from spektral.utils import nx_to_numpy

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


def load_data(nf_keys=None, ef_keys=None, amount=None):
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
        amount: the amount of molecules to return (in ascending order by number of atoms);

    Returns:
        the adjacency matrix, node features, edge features, and a Pandas dataframe containing labels;
    """
    if not os.path.exists(DATA_PATH):
        _download_data()    # Try to download dataset
    print('Loading QM9 dataset.')

    sdf_file = os.path.join(DATA_PATH, 'qm9.sdf')
    data = load_sdf(sdf_file, amount=amount)  # Internal SDF format

    # Load labels
    labels_file = os.path.join(DATA_PATH, 'qm9.sdf.csv')
    labels = load_csv(labels_file)
    if amount is not None:
        labels = labels[:amount]
    # Convert to Networkx
    data = [sdf_to_nx(_) for _ in data]
    if nf_keys is not None:
        if isinstance(nf_keys, str):
            nf_keys = [nf_keys]
    else:
        nf_keys = NODE_FEATURES
    if ef_keys is not None:
        if isinstance(ef_keys, str):
            ef_keys = [ef_keys]
    else:
        ef_keys = EDGE_FEATURES

    adj, nf, ef = nx_to_numpy(data,
                              auto_pad=True, self_loops=True,
                              nf_keys=nf_keys, ef_keys=ef_keys)
    return adj, nf, ef, labels


def test_download_data():
    _download_data()


if __name__ == "__main__":
    adj, nf, ef, labels = load_data(nf_keys=['atomic_num', 'charge', 'coords', 'iso'],
                                    ef_keys=['type', 'stereo'],
                                    amount=1000)