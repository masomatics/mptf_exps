import os, numpy as np, matplotlib.pyplot as plt, imageio
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange
import random # for reproducibility
import pdb

# ──────────────────────────────────────────────────────────────────────────
# DYNAMICS  →  z  (n × num_steps × d)  and the matching time grid
# ──────────────────────────────────────────────────────────────────────────
def simulate_particles(
        n=64, T=15, dt=0.1, d=3, beta=1, half_sph=False, seed=42
):
    rng = np.random.default_rng(seed)        # Generator instance
    print(f"""Calling simulate_particles with:
    n={n}, T={T}, dt={dt}, d={d}, beta={beta}, half_sph={half_sph}, seed={seed}""") 

    num_steps        = int(T / dt) + 1
    t_grid           = np.linspace(0, T, num_steps)

    # initial points on unit sphere
    z0 = rng.normal(size=(n, d))   
    z0 /= np.linalg.norm(z0, axis=1, keepdims=True)
    if half_sph and d == 3:
        z0[z0[:, 2] < 0] *= -1

    V, A = np.eye(d), np.eye(d)
    z    = np.zeros((n, num_steps, d))
    z[:, 0] = z0

    for i in range(num_steps - 1):
        Az           = (A @ z[:, i].T).T
        attn_matrix  = np.exp(beta * (Az @ Az.T))
        attn_matrix  = attn_matrix / attn_matrix.sum(1, keepdims=True)
        dz           = attn_matrix @ (V @ z[:, i].T).T
        z[:, i+1]    = z[:, i] + dt * dz
        z[:, i+1]   /= np.linalg.norm(z[:, i+1], axis=1, keepdims=True)

    return z, t_grid


# ──────────────────────────────────────────────────────────────────────────
# Dynamics with an Anchor Point  →  z  (n × num_steps × d)  and the matching time grid
# ──────────────────────────────────────────────────────────────────────────

def simulate_particles_with_anchor(n=64, T=15, dt=0.1, d=3, beta=1, half_sph=False, seed=42,
                                   anchor=None, anchor_weight=None):\


    #asser anchor is of d dim 
    assert len(anchor) == d, f"Anchor must be of dimension {d}, got {len(anchor)}" 
    anchor = anchor / np.linalg.norm(anchor)  # Normalize anchor

    # set up the initial zs            
    rng = np.random.default_rng(seed)        # Generator instance
    print(f"""Calling simulate_particles with:
    n={n}, T={T}, dt={dt}, d={d}, beta={beta}, half_sph={half_sph}, seed={seed}""") 

    num_steps        = int(T / dt) + 1
    t_grid           = np.linspace(0, T, num_steps)

    # initial points on unit sphere
    z0 = rng.normal(size=(n, d))   
    z0 /= np.linalg.norm(z0, axis=1, keepdims=True)
    if half_sph and d == 3:
        z0[z0[:, 2] < 0] *= -1

    # Save place for the anchor point : n+1 
    V, A = np.eye(d), np.eye(d)
    z    = np.zeros((n+1, num_steps, d))
    #concatenate the anchor point (d dim) to z0
    z0 = np.concatenate((z0, anchor.reshape(1, -1)), axis=0)  # n+1 x d

    z[:, 0] = z0    
    if anchor_weight is None:
        anchor_weight = n  # Default weight for the anchor point in attention calculation
    else:
        assert isinstance(anchor_weight, (int, float)), "Anchor weight must be a number"
    print(f"""Anchor point: {anchor}, weight: {anchor_weight}""") 
    #anchor dimension immobile 
    for i in range(num_steps - 1):
        z4Attn = z[:, i].copy()  # Copy current positions for attention calculation
        z4Attn[-1] = z4Attn[-1] * anchor_weight # Scale anchor point for attention   
        Az           = (A @ z4Attn.T).T
        attn_matrix  = np.exp(beta * (Az @ Az.T))
        attn_matrix  = attn_matrix / attn_matrix.sum(1, keepdims=True)
        dz           = attn_matrix @ (V @ z[:, i].T).T
        z[:-1, i+1]    = z[:-1, i] + dt * dz[:-1, :]  # Exclude the anchor point from dynamics
        z[:-1, i+1]   /= np.linalg.norm(z[:-1, i+1], axis=1, keepdims=True)    
        z[-1, i+1] = anchor  # Keep the anchor point fixed

    return z, t_grid
