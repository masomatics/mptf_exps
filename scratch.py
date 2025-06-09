from typing import List, Optional, Tuple, Union
import numpy as np
import particle_simulators as ps
from importlib import reload
import visualize as vis
import pdb

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


    keys = ['anchor', 'mobile', 'interjector']
    playmode = keys[2] 



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


        vis.render(z_list, 3,  t_grid, m=64, rootdrivepath='./figs', movie=True, fps=10,  title=f"mobile_anchors_weight{weight}_beta{beta}", interpolate=True) 

    elif playmode == 'modile':
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

        T = 80

        dt = 0.5
        theta_dot = 0.1       # rad / unit-time
        eta_dot   = 0.25
        beta = 1.0
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
            movie=True, fps=fps, title='mobile_anchors', interpolate=True)


    else:
        raise NotImplementedError(f"Unknown playmode: {playmode}.")