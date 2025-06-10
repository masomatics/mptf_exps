from typing import List, Optional, Tuple, Union
import numpy as np
import particle_simulators as ps

from importlib import reload
import visualize as vis
import pdb
from matplotlib import pyplot as plt
import os
import torch

def simple_great_circle_path(
    *, T: float = 15.0, dt: float = 0.1, d: int = 3, speed: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-particle great-circle trajectory on the unit d-sphere.

    Returns
    -------
    interject_ts : ndarray, shape (num_steps, d)
        The position at every time step.
    t_grid       : ndarray, shape (num_steps,)
        Matching time stamps.
    """
    assert d >= 2, "d must be at least 2 for a great circle"
    num_steps = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, num_steps)

    # pre-allocate for speed; (num_steps, d)
    z = np.zeros((num_steps, d))
    z[:, 0] = np.cos(speed * t_grid)   # x1
    z[:, 1] = np.sin(speed * t_grid)   # x2
    # remaining coordinates stay zero

    return z, t_grid


def kill_traj(arr, time_vec, death_time, sentinel=np.nan):
    """
    death_time 以降の時刻に対応する要素を sentinel に置換する。

    Parameters
    ----------
    arr : np.ndarray, shape (T, *, ...)
        時間軸が 0 次元目の配列
    time_vec : np.ndarray, shape (T,)
        各ステップの時刻
    death_time : float
        死亡時刻
    sentinel : float or object
        置換値。None を使うときは arr.astype(object) してから渡す

    Returns
    -------
    np.ndarray
        置換後の配列
    """
    mask = time_vec > death_time
    out = arr.astype(object) if sentinel is None else arr.copy()
    out[mask] = sentinel
    return out




import numpy as np
from typing import Tuple

def cart2sph(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cartesian → spherical:  returns (theta, phi) in radians.
    theta ∈ [0, π]   polar angle from +z
    phi   ∈ [0, 2π)  azimuth from +x toward +y
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))        # avoid NaNs
    phi   = np.arctan2(y, x) % (2.0 * np.pi)
    return theta, phi

def sph2cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Spherical → Cartesian (unit radius)."""
    sinT = np.sin(theta)
    return np.stack([sinT * np.cos(phi),
                     sinT * np.sin(phi),
                     np.cos(theta)], axis=-1)

def constant_speed_sphere_path(
    anchors: np.ndarray,          # (M, 3)   starting points (unit)
    *,
    theta_dot: float,             # dθ/dt  [rad / time-unit]
    eta_dot: float,               # dφ/dt  [rad / time-unit]
    T: float = 15.0,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a trajectory where every anchor moves with constant angular
    velocity (theta_dot, eta_dot) on the unit sphere.

    Returns
    -------
    traj : ndarray, shape (num_steps, M, 3)
        Cartesian positions for every anchor at every step.
    t_grid : ndarray, shape (num_steps,)
        Time stamps.
    """
    anchors = np.asarray(anchors, dtype=float)
    assert anchors.ndim == 2 and anchors.shape[1] == 3, "anchors must be (M,3)"
    # normalise defensively
    anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)

    M = anchors.shape[0]
    theta0, phi0 = cart2sph(anchors)               # (M,)

    num_steps = int(T / dt) + 1
    t_grid = np.linspace(0.0, T, num_steps)

    # evolve angles linearly
    theta = theta0[:, None] + theta_dot * t_grid   # (M, num_steps)
    phi   = phi0[:, None]   + eta_dot   * t_grid

    # keep angles in canonical ranges
    theta = np.mod(theta, 2.0 * np.pi)
    phi   = np.mod(phi,   2.0 * np.pi)

    # convert back to Cartesian for every time step
    traj = sph2cart(theta[..., None], phi[..., None])   # (M, num_steps, 3)
    traj = np.swapaxes(traj, 0, 1)                      # → (num_steps, M, 3)
    return traj, t_grid


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":


    keys = ['immobile', 'mobile', 'interjector', 'kuramotoAttn', 'kuramotoJ', 'kuramotoAttn2d', 'kuramotoJ2dClassic', 'kuramotoJ2d']
    playmode = keys[-1] 

    print(f"""Running simulation for playmode: {playmode}""")

    if playmode == 'immobile':
        '''
        IMMobile Anchor Simulation Example
        '''
        anchor1 = np.array([1.0, 0.0, 0.0])
        anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, anchor2))

        T = 150


        theta_dot = 0.1       # rad / unit-time
        eta_dot   = 0.25
        beta = 5.0

        myseed= 10
        weight = 0.1


        a_sim = ps.AnchorSimulator(anchor=anchors, anchor_weight=weight, T=T, beta=beta, dt=0.5)

        z_list, t_grid = a_sim.simulate()


        vis.render(z_list, 3,  t_grid, m=64, rootdrivepath='./figs', movie=True, fps=10,  title=f"immobile_anchors_weight{weight}_beta{beta}", interpolate=True) 

    elif playmode == 'mobile':
        '''
        Mobile Anchor Simulation Example
        '''
        anchor1 = np.array([1.0, 0.0, 0.0])
        anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, anchor2))

        T = 150


        theta_dot = 0.1       # rad / unit-time
        eta_dot   = 0.25
        beta = 5.0
        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=0.1,
        )

        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])


        anchor_traj = traj.squeeze(2)

        myseed= 10
        weight = 1.0


        m_sim = ps.MobileAnchorSimulator(anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=0.5)

        z_list, t_grid = m_sim.simulate()


        vis.render(z_list, 3,  t_grid, m=64, rootdrivepath='./figs', movie=True, fps=10,  title=f"mobile_anchors_weight{weight}_beta{beta}", interpolate=True) 



    elif playmode == 'interjector':
        anchor1 = np.array([1.0, 0.0, 0.0])
        anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, anchor2))

        T = 30

        dt = 0.5
        theta_dot = 0.1       # rad / unit-time
        eta_dot   = 0.25
        beta = 5.0
        fps = 8
        weight = 1.0


        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=dt,
        )

        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])


        anchor_traj = traj.squeeze(2)
        myseed= 10

        i_sim = ps.InterJectorSimulator(anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=dt)


        z_list, t_grid = i_sim.simulate()

        vis.render(z_list, 3, t_grid, m=64, rootdrivepath='./figs',
            movie=True, fps=fps, title=f"interjector_anchors_weight{weight}_beta{beta}", interpolate=True)


    elif playmode == 'kuramotoAttn':
        '''
        Kuramoto Simulation Example
        '''

        anchor1 = np.array([1.0, 0.0, 0.0])
        anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, anchor2))

        T = 150

        dt = 0.5
        theta_dot = 0       # rad / unit-time
        eta_dot   = 0.25
        beta = 5.0
        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=dt,
        )
        myseed= 10
        weight = 1.0
        fps = 8

        anchor_traj = traj.squeeze(2)


        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])



        k_sim = ps.KuramotoSelfAttnSimulator(anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=dt)


        z_list, t_grid = k_sim.simulate()

        vis.render(z_list, 3, t_grid, m=64, rootdrivepath='./figs',
            movie=True, fps=fps, title=f"{playmode}_anchors_weight{weight}_beta{beta}", interpolate=True)

    elif playmode == 'kuramotoJ':
        '''
        Kuramoto Simulation Example
        '''

        anchor1 = np.array([1.0, 0.0, 0.0])
        anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, anchor2))

        T = 50
        N0 = 5

        dt = 0.1
        theta_dot = 0       # rad / unit-time
        eta_dot   = 0
        beta = 5.0
        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=dt,
        )
        myseed= 10
        weight = 1.0
        fps = 8

        anchor_traj = traj.squeeze(2)

        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])




        k_sim = ps.KuramotoJSimulator(anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=dt, n=N0)

        print(f"""Simulation for {playmode}_anchors_weight{weight}_beta{beta} complete """ ) 


        z_list, t_grid = k_sim.simulate()

        vis.render(z_list, 3, t_grid, m=N0, rootdrivepath='./figs',
            movie=True, fps=fps, title=f"{playmode}_anchors_weight{weight}_beta{beta}", interpolate=True)        

    elif playmode == 'kuramotoAttn2d':
        '''
        Kuramoto Simulation Example
        '''

        d = 2
        anchor1 = np.array([1.0, 0.0, 0.0])
        anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, anchor2))

        T = 50
        N0 = 20

        dt = 0.1

        angular_speed = 0     # rad / unit-time
        theta_dot = angular_speed      # rad / unit-time
        eta_dot   = angular_speed
        beta = 5.0
        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=dt,
        )
        myseed= 6
        weight = 0.0
        fps = 8

        anchor_traj = traj.squeeze(2)[:, :, :d]

        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])

        k_sim = ps.KuramotoSelfAttnSimulator(anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=dt, n=N0, d=2)

        print(f"""Simulation for {playmode}_anchors_weight{weight}_beta{beta} complete """) 

        z_list, t_grid = k_sim.simulate()

        e_list = [k_sim.compute_energy(z).sum().item() for z in z_list]
        E = torch.tensor(e_list, dtype=torch.float32, device=k_sim.device)

        plt.plot(t_grid, E)              # do NOT specify colors → default style
        plt.xlabel("time $t$")
        plt.ylabel("total energy  $\\sum_i E_i(t)$")
        plt.title("Kuramoto self-attention energy vs. time")

        # --- save & show ------------------------------------
        plt.tight_layout()
        figpath = os.path.join('./figs', f"energy_vs_time_{playmode}_anchors_weight{weight}_N0{N0}_beta{beta}_ancspeed{angular_speed}_seed{myseed}.png")
        plt.savefig(figpath, dpi=150)
        plt.show()

        #vis.render(z_list, d, t_grid, m=N0, rootdrivepath='./figs',
        #    movie=True, fps=fps, title=f"{playmode}_anchors_weight{weight}_N0{N0}_beta{beta}_ancspeed{angular_speed}_seed{myseed}", interpolate=True,)        


    elif playmode == 'kuramotoJ2dClassic':
        '''
        Kuramoto Simulation Example
        '''

        d = 2
        anchor1 = np.array([1.0, 0.0, 0.0])
        # anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, ))


        T = 20
        N0 = 50

        dt = 0.1

        angular_speed = 0     # rad / unit-time
        theta_dot = angular_speed      # rad / unit-time
        eta_dot   = angular_speed
        beta = 5.0
        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=dt,
        )
        myseed= 42
        weight = 0.0  ##Set this to 0.0 to disable anchors. This and the following J shall realize the simple Kuramoto model.
        fps = 8

        anchor_traj = traj.squeeze(2)[:, :, :d]

        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])


        #### FIXING J to  1 - I so that we will realize the most basic Kuramoto model   ( K/N \sum_i cos(theta_i- theta_j) )
        K = 5.0
        fixed_J = (K / N0) * (torch.ones(N0, N0) - torch.eye(N0))


        k_sim = ps.KuramotoJSimulator(fixed_J=fixed_J, anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=dt, n=N0, d=2, 
                                      seed=myseed)

        print(f"""Simulation for {playmode}_anchors_weight{weight}_beta{beta} complete """) 


        z_list, t_grid = k_sim.simulate()
        e_list = [k_sim.compute_energy(z).sum().item() for z in z_list]
        E = torch.tensor(e_list, dtype=torch.float32, device=k_sim.device)

        plt.plot(t_grid, E)              # do NOT specify colors → default style
        plt.xlabel("time $t$")
        plt.ylabel("total energy  $\\sum_i E_i(t)$")
        plt.title("Kuramoto self-attention energy vs. time")

        # --- save & show ------------------------------------
        plt.tight_layout()
        figpath = os.path.join('./figs', f"energy_vs_time_{playmode}_anchors_weight{weight}_N0{N0}_beta{beta}_ancspeed{angular_speed}_seed{myseed}.png")
        plt.savefig(figpath, dpi=150)
        # plt.show()

        vis.render(z_list, d, t_grid, m=N0, rootdrivepath='./figs',
            movie=True, fps=fps, title=f"{playmode}_anchors_weight{weight}_N0{N0}_beta{beta}_ancspeed{angular_speed}_seed{myseed}", interpolate=True,)        


    elif playmode == 'kuramotoJ2d':
        '''
        Kuramoto Simulation Example
        '''

        d = 2
        anchor1 = np.array([1.0, 0.0, 0.0])
        # anchor2 = np.array([0.0, 1.0, 0.0])
        anchors = np.vstack((anchor1, ))

        T = 20
        N0 = 50

        Deathtime = 10.0
        dt = 0.1

        angular_speed = 0.5     # rad / unit-time
        theta_dot = angular_speed      # rad / unit-time
        eta_dot   = angular_speed
        beta = 5.0
        traj, t = constant_speed_sphere_path(
            anchors,
            theta_dot=theta_dot,
            eta_dot=eta_dot,
            T=T,
            dt=dt,
        )
        myseed= 42
        weight = 5.0 
        fps = 8


        anchor_traj = traj.squeeze(2)[:, :, :d]
        anchor_traj = kill_traj(anchor_traj, t, Deathtime, sentinel=np.nan)

        print("trajectory shape:", traj.shape)   # (num_steps, 2, 3)
        print("first frame:", traj[0])
        print("last  frame:", traj[-1])


        #### FIXING J to  1 - I so that we will realize the most basic Kuramoto model   ( K/N \sum_i cos(theta_i- theta_j) )
        K = 5.0
        fixed_J = (K / N0) * (torch.ones(N0, N0) - torch.eye(N0))


        k_sim = ps.KuramotoJSimulator(fixed_J=fixed_J, anchor_weight=weight, anchor_time_series=anchor_traj, T=T, beta=beta, dt=dt, n=N0, d=2, 
                                      seed=myseed)

        print(f"""Simulation for {playmode}_anchors_weight{weight}_beta{beta} complete """) 


        z_list, t_grid = k_sim.simulate()
        e_list = [k_sim.compute_energy(z).sum().item() for z in z_list]
        E = torch.tensor(e_list, dtype=torch.float32, device=k_sim.device)

        plt.plot(t_grid, E)              # do NOT specify colors → default style
        plt.xlabel("time $t$")
        plt.ylabel("total energy  $\\sum_i E_i(t)$")
        plt.title("Kuramoto self-attention energy vs. time")

        # --- save & show ------------------------------------
        plt.tight_layout()
        figpath = os.path.join('./figs', f"energy_vs_time_{playmode}_anchors_weight{weight}_N0{N0}_beta{beta}_ancspeed{angular_speed}_seed{myseed}.png")
        plt.savefig(figpath, dpi=150)
        # plt.show()

        vis.render(z_list, d, t_grid, m=N0, rootdrivepath='./figs',
            movie=True, fps=fps, title=f"{playmode}_anchors_weight{weight}_N0{N0}_beta{beta}_ancspeed{angular_speed}_seed{myseed}", interpolate=True,)              

    else:
        raise NotImplementedError(f"Unknown playmode: {playmode}.")