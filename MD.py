import os, string, cmath, math
from matplotlib import pyplot as plt
import numpy as np
from numpy import loadtxt as lt
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors

''' Aldo Raeliarijaona: 

Program the parses VASP and QE outputs and perform analysis of M.D. runs '''


def transform_positions(x,ref,thr):
    """
    Transforms fractional atomic coordinates by shifting any component (w.r.t. ref) > thr back into the unit cell.

    Parameters:
        x (np.ndarray): An (n_atoms, 3) array of fractional atomic coordinates.
        ref           : An (n_atoms, 3) array of fractional atomic coordinates used as a reference
        thr           : A threshold for transformation of fractional atomic coordinates.

    Returns:
        np.ndarray: The transformed coordinates.
    """
    x_transformed = np.where((x-ref) > thr, x - 1, x)

    return x_transformed    

def Read_XDATCAR(fbase,fname,AvStart,AvSkip,AvEnd,outFlag):
    """
    Reads XDATCAR file and extracts atomic positions, lattice vectors, or volume.

    Parameters:
        fbase (str): File path base.
        fname (str): File name.
        av_start (int): Start index for averaging.
        av_skip (int): Skip interval for averaging.
        av_end (int): End index for averaging.
        out_flag (str): 'Avg', 'Vol', or 'Pos' to determine output.

    Returns:
        np.ndarray: Result based on out_flag.
    """
    num_coords = 3

    with open(fbase + fname, 'r') as file:
        lines = file.readlines()

    # Find all line indices for "Direct configuration="
    ln_md = [i for i, line in enumerate(lines) if line.strip().startswith("Direct configuration=")]
    md_steps = len(ln_md)

    # Number of atoms
    n_all = np.array(lines[ln_md[0] - 1].split(), dtype=int)
    n_atoms = np.sum(n_all)

    # Preallocate arrays
    a_bl_tmp = np.zeros((md_steps, num_coords, num_coords))
    md_pos = np.zeros((md_steps, n_atoms, num_coords))
    vol = np.zeros(md_steps)

    # Compute line indices for lattice vectors and atomic positions
    bl_ref_ln = [idx - 5 for idx in ln_md]
    at_ref_ln = [idx + 1 for idx in ln_md]

    for step in range(md_steps):
        for i in range(num_coords):
            a_bl_tmp[step, i] = np.fromstring(lines[bl_ref_ln[step] + i], sep=' ')
        vol[step] = np.dot(np.cross(a_bl_tmp[step, 0], a_bl_tmp[step, 1]), a_bl_tmp[step, 2])

        for atom in range(n_atoms):
            md_pos[step, atom] = np.fromstring(lines[at_ref_ln[step] + atom], sep=' ')

    # Output based on flag
    if outFlag == 'Avg':
        a_bl_avg = np.mean(a_bl_tmp[av_start:av_end:av_skip], axis=0)
        return a_bl_avg
    elif outFlag == 'Vol':
        return vol
    elif outFlag == 'Pos':
        return md_pos
    else:
        # Returning raw lattice vectors (could be further adjusted as needed)
        return a_bl_tmp

def Read_OUTCAR(fbase,fname):

    """
    Parses a VASP OUTCAR file to extract MD energy, pressure, temperature, stress, and lattice vectors.

    Parameters:
        fbase (str): Base file path.
        fname (str): File name.

    Returns:
        np.ndarray: Array containing temperature, energy, pressure, stress components, and lattice vectors.
    """
    nC = 3  # Dimensions (x, y, z)

    with open(fbase + fname, 'r') as file:
        lines = file.readlines()

    # Identify line indices of interest using list comprehensions
    ln_md    = [i + 8 for i, ln in enumerate(lines) if ln.strip().startswith("ENERGY OF THE ELECTRON-ION-THERMOSTAT SYSTEM (eV)")]
    ln_press = [i for i, ln in enumerate(lines) if ln.strip().startswith("total pressure")]
    ln_temp  = [i for i, ln in enumerate(lines) if ln.strip().startswith("kin. lattice")]
    ln_vol   = [i for i, ln in enumerate(lines) if ln.strip().startswith("volume of cell")]
    ln_bl    = [i for i, ln in enumerate(lines) if ln.strip().startswith("length of vectors")]
    ln_str   = [i for i, ln in enumerate(lines) if ln.strip().startswith("Total+kin.")]

    md_steps = len(ln_md)

    # Initialize arrays
    energy = np.empty(md_steps)
    pressure = np.empty(md_steps)
    temperature = np.empty(md_steps)
    volume = np.empty(md_steps)
    stress = np.empty((md_steps, 2 * nC))
    lattice_vectors = np.empty((md_steps, nC))

    # Fill arrays
    for i, idx in enumerate(ln_md):
        energy[i] = float(lines[idx].split()[4])

    for i, idx in enumerate(ln_press):
        pressure[i] = float(lines[idx].split()[3])

    for i, idx in enumerate(ln_temp):
        temperature[i] = float(lines[idx].split()[5])

    for i, idx in enumerate(ln_vol[1:-1]):
        volume[i] = float(lines[idx].split()[4])

    for i, idx in enumerate(ln_str):
        stress[i] = np.array(lines[idx].split()[1:7], dtype=float)

    for i, idx in enumerate(ln_bl[1:-1]):
        lattice_vectors[i] = np.array(lines[idx + 1].split()[0:3], dtype=float)

    # Combine all extracted arrays into one output
    output = np.vstack([
        temperature,
        energy,
        pressure,
        stress[:, 0], stress[:, 1], stress[:, 2],
        stress[:, 3], stress[:, 4], stress[:, 5],
        lattice_vectors[:, 0], lattice_vectors[:, 1], lattice_vectors[:, 2]
    ])

    return output


def Read_OSZICAR(fbase,fname):
    global aLatt;
    nC = 3  # Unused in this function, could be removed if not needed elsewhere

    # Read all lines from the file
    with open(fbase + fname) as frq:
        lines = frq.readlines()

    # Filter lines containing the keyword "T="
    kwrd = "T="
    md_lines = [i for i, line in enumerate(lines) if kwrd in line]
    MDsteps = len(md_lines)

    # Initialize arrays
    T  = np.zeros(MDsteps)
    E  = np.zeros(MDsteps)
    F  = np.zeros(MDsteps)
    E0 = np.zeros(MDsteps)
    Ek = np.zeros(MDsteps)

    # Extract data
    for i, idx in enumerate(md_lines):
        parts = lines[idx].split()
        T[i]  = float(parts[2])
        E[i]  = float(parts[4])
        F[i]  = float(parts[6])
        E0[i] = float(parts[8])
        Ek[i] = float(parts[10])

    return np.array([T, E, F, E0, Ek])


def Read_ZSTAR(fbase,fname,nIon):
    global aLatt;
    ndCoord = 3  # Number of coordinates (x, y, z)

    # Read all lines from the file
    with open(fbase + fname) as frq:
        lines = frq.readlines()

    # Locate the line index for "BORN EFFECTIVE CHARGES"
    ln_zstar = next(i for i, ln in enumerate(lines) if ln.strip().startswith("BORN EFFECTIVE CHARGES"))
    first_ion_line = ln_zstar + 2

    # Prepare storage
    zstar = np.zeros((nIon, ndCoord, ndCoord))
    ln_ion = np.zeros(nIon, dtype=int)

    # Extract ion line starts
    for i in range(nIon):
        line = lines[first_ion_line + 4 * i]
        ion_id = int(line.split()[1]) - 1
        ln_ion[ion_id] = first_ion_line + 4 * i + 1

    # Fill Z* values
    for i in range(nIon):
        for j in range(ndCoord):
            zstar[i, j, :] = list(map(float, lines[ln_ion[i] + j].split()[1:4]))

    return zstar

def Read_VASP(fbase, fname):
    with open(fbase + fname, 'r') as fvasp:
        lines = fvasp.readlines()

    # Lattice vectors and scaling factor
    scale = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scale

    # Atom types and counts
    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    total_atoms = sum(atom_counts)

    # Atomic coordinates (assume Direct format)
    coord_start_index = 8
    atomic_coords = np.array(
        [list(map(float, lines[coord_start_index + i].split()[:3])) for i in range(total_atoms)]
    )

    return {
        "lattice_vectors": lattice_vectors,
        "atom_types": atom_types,
        "atom_counts": atom_counts,
        "total_atoms": total_atoms,
        "coordinates": atomic_coords
    }
