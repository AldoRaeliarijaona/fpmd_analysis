import os, string, cmath, math
from matplotlib import pyplot as plt
import numpy as np
from numpy import loadtxt as lt
import pandas as pd
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
import MD


#plt.rcParams.update({'font.size': 18})

if __name__ == "__main__":
    num_atoms = 135
    conv_factor = 16.02

    # File paths
    fbase_benchmark = 'PATH_TO_ReferenceStructure'
    fbase_ref = fbase_benchmark
    fname_ref = 'Ref.vasp'
    fbase_perfect = 'PATH_TO_XDATCAR'
    fname_perfect = 'NAME_OF_XDATCAR'
    fbase_z = 'PATH_TO_ZSTAR'
    fname_z = 'NAME_OF_FILEwithZstar'

    # Averaging ranges
    av_ir, av_fr, av_delta_r = 0, 1, 1
    av_i, av_f, av_delta = 0, 30000, 1

    # Read data
    r_pos  = MD.Read_XDATCAR(fbase_perfect, fname_perfect, av_i, av_delta, av_f, 'Pos')
    a_bl   = MD.Read_XDATCAR(fbase_perfect, fname_perfect, av_i, av_delta, av_f, 'all')
    ref_p  = MD.Read_XDATCAR(fbase_ref, fname_ref, av_ir, av_delta_r, av_fr, 'Pos')
    z_eff  = MD.Read_ZSTAR(fbase_z, fname_z, num_atoms)

    n_md, n_at, n_c = r_pos.shape

    # Initialize arrays
    du = np.zeros((n_md, n_at, n_c))
    r_pos_cart = np.zeros_like(du)
    ploc = np.zeros((n_at, n_c))
    ptot = np.zeros((n_md, n_c))



    pos_thr = 0.50
    for i_md in range(n_md):
        # Compute displacement from reference
        du[i_md] = MD.transform_positions(r_pos[i_md],ref_p[0],pos_thr) - ref_p[0]
 
        # Convert to Cartesian coordinates
        a_bl_i = a_bl[i_md]
        r_pos_cart[i_md] = (du[i_md]) @ a_bl_i.T
 
        # Compute volume of unit cell
        vol_i_md = np.dot(np.cross(a_bl_i[0], a_bl_i[1]), a_bl_i[2])
 
        # Local polarization per atom
        ploc[:] = np.einsum('ijk,ik->ij', z_eff, r_pos_cart[i_md])
 
        # Total polarization
        ptot[i_md] = conv_factor * ploc.sum(axis=0) / vol_i_md

    avgW_min,avgW_max=10000,20000
    p_m = np.mean(ptot[avgW_min:avgW_max],axis=0)
    
    t = np.divide(range(0,avgW_max),1000);
   # fig, ax = plt.subplots(figsize=(5, 5), dpi=600)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(t[:],ptot[:avgW_max,0], 'g-', marker='s', ms=2.0, lw=0.5,label='r');
    ax.plot(t[:],ptot[:avgW_max,1], 'b-', marker='s', ms=2.0, lw=2.5,label='r');
    ax.plot(t[:],ptot[:avgW_max,2], 'r-', marker='s', ms=2.0, lw=0.5,label='r');
    ax.hlines(y=p_m[0],xmin=avgW_min/1000,xmax=avgW_max/1000,color = 'k', linestyle='--');
    ax.hlines(y=p_m[1],xmin=avgW_min/1000,xmax=avgW_max/1000,color = 'k', linestyle='--');
    ax.hlines(y=p_m[2],xmin=avgW_min/1000,xmax=avgW_max/1000,color = 'k', linestyle='--');
    plt.xlim(0,20)
    plt.ylim(-0.9,0.9)
    
    ax.set(xlabel='time (ps)', ylabel='P (C m$^{-2}$)')
    ax.grid()
    plt.show()
