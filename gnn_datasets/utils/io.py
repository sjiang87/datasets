"""
Author: Shengli Jiang
Email: sjiang87@wisc.edu / shengli.jiang@anl.gov
Adapted from Spektral tf1
https://github.com/danielegrattarola/spektral/tree/tf1
"""
import numpy as np
# Reference for implementation:
# # http://www.nonlinear.com/progenesis/sdf-studio/v0.9/faq/sdf-file-format-guidance.aspx
#
# While parsing the SDF file, molecules are stored in a dictionary like this:
#
# {'atoms': [{'atomic_num': 7,
#             'charge': 0,
#             'coords': array([-0.0299,  1.2183,  0.2994]),
#             'index': 0,
#             'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             'iso': 0},
#            ...,
#            {'atomic_num': 1,
#             'charge': 0,
#             'coords': array([ 0.6896, -2.3002, -0.1042]),
#             'index': 14,
#             'info': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             'iso': 0}],
#  'bonds': [{'end_atom': 13,
#             'info': array([0, 0, 0]),
#             'start_atom': 4,
#             'stereo': 0,
#             'type': 1},
#            ...,
#            {'end_atom': 8,
#             'info': array([0, 0, 0]),
#             'start_atom': 7,
#             'stereo': 0,
#             'type': 3}],
#  'comment': '',
#  'data': [''],
#  'details': '-OEChem-03231823253D',
#  'n_atoms': 15,
#  'n_bonds': 15,
#  'name': 'gdb_54964',
#  'properties': []}
HEADER_SIZE = 3


def _parse_header(sdf):
    try:
        return sdf[0].strip(), sdf[1].strip(), sdf[2].strip()
    except IndexError:
        print(sdf)


def _parse_counts_line(sdf):
    # 12 fields
    # First 11 are 3 characters long
    # Last one is 6 characters long
    # First two give the number of atoms and bonds

    values = sdf[HEADER_SIZE]
    n_atoms = int(values[:3])
    n_bonds = int(values[3:6])

    return n_atoms, n_bonds


def _parse_atoms_block(sdf, n_atoms):
    # The first three fields, 10 characters long each, describe the atom's
    # position in the X, Y, and Z dimensions.
    # After that there is a space, and three characters for an atomic symbol.
    # After the symbol, there are two characters for the mass difference from
    # the monoisotope.
    # Next you have three characters for the charge.
    # There are ten more fields with three characters each, but these are all
    # rarely used.

    start = HEADER_SIZE + 1  # Add 1 for counts line
    stop = start + n_atoms
    values = sdf[start:stop]

    atoms = []
    for i, v in enumerate(values):
        coords = np.array([float(v[pos:pos+10]) for pos in range(0, 30, 10)])
        atomic_num = get_atomic_num(v[31:34].strip())
        iso = int(v[34:36])
        charge = int(v[36:39])
        info = np.array([int(v[pos:pos+3]) for pos in range(39, len(v), 3)])
        atoms.append({'index': i,
                      'coords': coords,
                      'atomic_num': atomic_num,
                      'iso': iso,
                      'charge': charge,
                      'info': info})
    return atoms


def _parse_bonds_block(sdf, n_atoms, n_bonds):
    # The first two fields are the indexes of the atoms included in this bond
    # (starting from 1). The third field defines the type of bond, and the
    # fourth the stereoscopy of the bond.
    # There are a further three fields, with 3 characters each, but these are
    # rarely used and can be left blank.

    start = HEADER_SIZE + n_atoms + 1  # Add 1 for counts line
    stop = start + n_bonds
    values = sdf[start:stop]

    bonds = []
    for v in values:
        start_atom = int(v[:3]) - 1
        end_atom = int(v[3:6]) - 1
        type_ = int(v[6:9])
        stereo = int(v[9:12])
        info = np.array([int(v[pos:pos + 3]) for pos in range(12, len(v), 3)])
        bonds.append({'start_atom': start_atom,
                      'end_atom': end_atom,
                      'type': type_,
                      'stereo': stereo,
                      'info': info})
    return bonds


def _parse_properties(sdf, n_atoms, n_bonds):
    # TODO This just returns a list of properties.
    # See https://docs.chemaxon.com/display/docs/MDL+MOLfiles%2C+RGfiles%2C+SDfiles%2C+Rxnfiles%2C+RDfiles+formats
    # for documentation.

    start = HEADER_SIZE + n_atoms + n_bonds + 1  # Add 1 for counts line
    stop = sdf.index('M  END')

    return sdf[start:stop]


def _parse_data_fields(sdf):
    # TODO This just returns a list of data fields.

    start = sdf.index('M  END') + 1

    return sdf[start:] if start < len(sdf) else []


def parse_sdf(sdf):
    """
    Parse an .sdf file.
    Args:
        sdf: single sdf element.

    Returns:
        array of output.
    """
    sdf_out = {}
    sdf = sdf.split('\n')
    sdf_out['name'], sdf_out['details'], sdf_out['comment'] = _parse_header(sdf)
    sdf_out['n_atoms'], sdf_out['n_bonds'] = _parse_counts_line(sdf)
    sdf_out['atoms'] = _parse_atoms_block(sdf, sdf_out['n_atoms'])
    sdf_out['bonds'] = _parse_bonds_block(sdf, sdf_out['n_atoms'], sdf_out['n_bonds'])
    sdf_out['properties'] = _parse_properties(sdf, sdf_out['n_atoms'], sdf_out['n_bonds'])
    sdf_out['data'] = _parse_data_fields(sdf)
    return sdf_out


def parse_sdf_file(sdf_file, amount=None):
    """
    Parse an .sdf file.
    Args:
        sdf_file: opened sdf file
        amount: only load the first `amount` molecules from the file

    Returns:
        a list of molecules in the internal SDF format.
    """
    data = sdf_file.read().split('$$$$\n')
    if data[-1] == '':
        data = data[:-1]
    if amount is not None:
        data = data[:amount]
    output = [parse_sdf(sdf) for sdf in data]  # Parallel execution doesn't help
    return output


def load_sdf(filename, amount=None):
    """
    Load an .sdf file and return a list of molecules in the internal SDF format.
    Args:
        filename: target SDF file;
        amount: only load the first `amount` molecules from the file

    Returns:
        a list of molecules in the internal SDF format.
    """
    print('Reading SDF')
    with open(filename) as f:
        return parse_sdf_file(f, amount=amount)
