"""
particle_simulators.py  — population‑aware & numerically stable
---------------------------------------------------------------

* **List trajectory** (`z_list[t]  → (N_t, d)`), same as previous revision.
* **Stable soft‑max** to avoid overflow when `anchor_weight` is large.
* **Self‑test** passes `anchor_weight = 1.0`, so results match the old tensor code.

Helper
~~~~~~
`stack_trajectory(z_list)` → `(N, T, d)` when every frame has the same N.

Classes
~~~~~~~
* `Simulator`          – free swarm.
* `AnchorSimulator`    – free swarm + immobile anchor (last row).

Legacy wrappers keep the original function names.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union, Callable
import numpy as np
import pdb

__all__ = [
    "Simulator",
    "AnchorSimulator",
    "simulate_particles",
    "simulate_particles_with_anchor",
    "stack_trajectory",
]


# ----------------------------------------------------------------------
#  Utility: numerically stable row‑softmax
# ----------------------------------------------------------------------

def _row_softmax(x: np.ndarray) -> np.ndarray:
    x_shift = x - x.max(axis=1, keepdims=True)  # subtract row‑wise max
    e = np.exp(x_shift)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
#  Helper: convert list‑trajectory back to dense tensor
# ----------------------------------------------------------------------

def stack_trajectory(z_list: List[np.ndarray]) -> np.ndarray:
    n0 = z_list[0].shape[0]
    if not all(f.shape[0] == n0 for f in z_list):
        raise ValueError("Cannot stack – particle count varies across frames")
    return np.stack(z_list, axis=1)  # (N, T, d)


# ======================================================================
#  BASE CLASS  – FREE SWARM
# ======================================================================
class Simulator:
    """Self‑attention swarm on the *d*-sphere (population may vary)."""

    def __init__(
        self,
        n: int = 64,
        T: float = 15.0,
        dt: float = 0.1,
        d: int = 3,
        beta: float = 1.0,
        *,
        half_sph: bool = False,
        seed: int = 42,
    ) -> None:
        self.n0 = n
        self.T = T
        self.dt = dt
        self.d = d
        self.beta = beta
        self.half_sph = half_sph
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def simulate(self) -> Tuple[List[np.ndarray], np.ndarray]:
        print(
            f"Calling simulate_particles with (population mode):\n"
            f"    n0={self.n0}, T={self.T}, dt={self.dt}, d={self.d}, beta={self.beta},"
            f" half_sph={self.half_sph}, seed={self.seed}"
        )
        num_steps = int(self.T / self.dt) + 1
        t_grid = np.linspace(0.0, self.T, num_steps)
        z_list: List[np.ndarray] = []

        z_curr = self._initial_positions(self.n0)

        for _ in range(num_steps):
            z_list.append(z_curr.copy())
            dz = self._step(z_curr)
            z_next = z_curr + self.dt * dz
            z_next /= np.linalg.norm(z_next, axis=1, keepdims=True)
            z_curr = z_next
        return z_list, t_grid

    # ------------------------------------------------------------------
    def _step(self, z_slice: np.ndarray) -> np.ndarray:
        V, A = np.eye(self.d), np.eye(self.d)
        Az = (A @ z_slice.T).T
        attn = _row_softmax(self.beta * (Az @ Az.T))
        return attn @ (V @ z_slice.T).T

    # ------------------------------------------------------------------
    def _initial_positions(self, n: int) -> np.ndarray:
        z0 = self._rng.normal(size=(n, self.d))
        z0 /= np.linalg.norm(z0, axis=1, keepdims=True)
        if self.half_sph and self.d == 3:
            z0[z0[:, 2] < 0] *= -1
        return z0


# ======================================================================
#  SUBCLASS – MULTIPLE IMMOBILE ANCHORS
# ======================================================================
class AnchorSimulator(Simulator):
    """
    Swarm with one or more immobile anchors.

    Parameters
    ----------
    anchor : (M, d) or (d,) array-like
        Each row is a unit vector giving an anchor position.
    anchor_weight : float or (M,) array-like, optional
        Attention multiplier(s).  Default = `n0` for every anchor.
    """

    def __init__(
        self,
        *,
        anchor: Union[np.ndarray, list, tuple],
        anchor_weight: Optional[Union[float, list, np.ndarray]] = None,  # CHANGED
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)

        # ---------- accept single anchor or stack of anchors ------------
        anchor = np.asarray(anchor, dtype=float)
        if anchor.ndim == 1:                       # (d,) → (1,d)
            anchor = anchor[None, :]
        assert anchor.ndim == 2 and anchor.shape[1] == self.d, (
            f"`anchor` must have shape (M, {self.d}) or ({self.d},)"
        )
        # normalise every anchor row
        if any(np.linalg.norm(anchor, axis=1, keepdims=True) ==0):
            print("Warning: not using valid anchors when Immobile")

        anchor /= np.linalg.norm(anchor, axis=1, keepdims=True)
        self.anchors = anchor                      # shape (M, d)
        self.M = anchor.shape[0]                   # number of anchors

        # ---------- anchor weights --------------------------------------
        if anchor_weight is None:
            anchor_weight = self.n0/self.M  # default: n0 per anchor
        anchor_weight = np.asarray(anchor_weight, dtype=float)
        if anchor_weight.ndim == 0:                # scalar → (M,)
            anchor_weight = np.full(self.M, anchor_weight)
        assert anchor_weight.shape == (self.M,), (
            "`anchor_weight` must be a scalar or shape (M,)"
        )
        self.anchor_weight = anchor_weight

    # ------------------------------------------------------------------
    def simulate(self, verbose=True) -> Tuple[List[np.ndarray], np.ndarray]:
        if verbose:
            print(
                f"""Calling simulate_particles with (multi-anchor, population mode):
                free_n0={self.n0}, M={self.M}, T={self.T}, dt={self.dt}, d={self.d}, beta={self.beta},
                half_sph={self.half_sph}, seed={self.seed}"""
            )
            print(f"""Anchors (rows):\n {self.anchors}\n weights: {self.anchor_weight}""")

        num_steps = int(self.T / self.dt) + 1
        t_grid = np.linspace(0.0, self.T, num_steps)
        z_list: List[np.ndarray] = []

        # concatenate initial free particles + anchors
        z_curr = np.vstack([self._initial_positions(self.n0), self.anchors])

        for _ in range(num_steps):
            z_list.append(z_curr.copy())

            dz = self._anchored_step(z_curr)
            z_next_free = z_curr[:-self.M] + self.dt * dz[:-self.M]
            z_next_free /= np.linalg.norm(z_next_free, axis=1, keepdims=True)

            # anchors remain fixed
            z_curr = np.vstack([z_next_free, self.anchors])

        return z_list, t_grid

    # ------------------------------------------------------------------
    def _anchored_step(self, z_slice: np.ndarray) -> np.ndarray:
        """Apply anchor weights only to the last M rows."""
        V, A = np.eye(self.d), np.eye(self.d)

        z4 = z_slice.copy()
        z4[-self.M :] *= self.anchor_weight[:, None]      # NEW

        Az = (A @ z4.T).T
        attn = _row_softmax(self.beta * (Az @ Az.T))
        return attn @ (V @ z_slice.T).T


# ======================================================================
#  SUBCLASS – MOBILE ANCHORS (POPULATION CONSTANT)
# ======================================================================
class MobileAnchorSimulator(AnchorSimulator):
    """Anchors follow a provided time‑series or ODE; population stays fixed."""

    def __init__(
        self,
        *,
        anchor_time_series: Optional[np.ndarray] = None,       # (T,M,d)
        anchor_ode: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        anchor_weight: Optional[Union[float, list, np.ndarray]] = None,
        **base_kwargs,
    ) -> None:

        if (anchor_time_series is None) == (anchor_ode is None):
            raise ValueError("Provide exactly one of anchor_time_series or anchor_ode.")

        if anchor_time_series is not None:
            ts = np.asarray(anchor_time_series, dtype=float)
            assert ts.ndim == 3, "anchor_time_series must be (T,M,d)"
            anchor_init = ts[0]
            self._anchor_ts = ts
            self._use_ts = True
        else:
            anchor_init = np.zeros((1, base_kwargs.get('d', 3)))  # placeholder
            self._anchor_ode = anchor_ode
            self._use_ts = False

        super().__init__(anchor=anchor_init, anchor_weight=anchor_weight, **base_kwargs)

    # ------------------------------------------------------------------
    def simulate(self) -> Tuple[List[np.ndarray], np.ndarray]:
        num_steps = int(self.T / self.dt) + 1
        t_grid = np.linspace(0.0, self.T, num_steps)
        z_list: List[np.ndarray] = []

        free  = self._initial_positions(self.n0)   # (n0,d)
        anc   = self.anchors.copy()                # (M,d)
        state = np.vstack([free, anc])

        for step in range(num_steps):
            z_list.append(state.copy())

            # ------ free particles step --------------------------------
            attn = state.copy(); attn[-self.M:] *= self.anchor_weight[:, None]
            dz = self._step(attn)
            free_next = state[:-self.M] + self.dt * dz[:-self.M]
            free_next /= np.linalg.norm(free_next, axis=1, keepdims=True)

            # ------ anchor update --------------------------------------
            if self._use_ts:
                anc_next = self._anchor_ts[step]
            else:
                anc_next = self._anchor_ode(t_grid[step], state[-self.M:])
            anc_next /= np.linalg.norm(anc_next, axis=1, keepdims=True)

            state = np.vstack([free_next, anc_next])

        return z_list, t_grid



# ======================================================================
#  SUBCLASS – INTERJECTOR (growing, weighted anchor history)
# ======================================================================
class InterJectorSimulator(Simulator):
    """Free block (size *n0*) interacts with a growing FIFO history of K‑sized
    anchor slices provided by `anchor_time_series`.

    Parameters
    ----------
    anchor_time_series : ndarray, shape (T, K, d)
        Slice *t* (0‑based) is appended at the **end of step t**.
    anchor_weight : float or (T,) ndarray, optional
        • scalar → same mass for every slice  
        • vector  → mass[t] applied to slice `anchor_time_series[t]`.
    max_history : int or None
        Maximum number of anchor slices to retain.  If reached, the oldest
        slice is dropped FIFO.  Default = None (grow unbounded).
    """

    def __init__(
        self,
        *,
        anchor_time_series: np.ndarray,      # (T,K,d)
        anchor_weight: Union[float, np.ndarray] = 1.0,
        max_history: Optional[int] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)

        ts = np.asarray(anchor_time_series, dtype=float)
        if ts.ndim != 3 or ts.shape[2] != self.d:
            raise ValueError("anchor_time_series must have shape (T, K, d)")
        self._ts = ts                      # (T,K,d)
        self.K = ts.shape[1]
        self.max_history = max_history

        w = np.asarray(anchor_weight, dtype=float)
        if w.ndim == 0:
            w = np.full(ts.shape[0], w)    # broadcast scalar → (T,)
        if w.shape != (ts.shape[0],):
            raise ValueError("anchor_weight must be scalar or shape (T,)")
        self._w = w

    # ------------------------------------------------------------------
    def simulate(self) -> Tuple[List[np.ndarray], np.ndarray]:
        T_steps = self._ts.shape[0]
        if T_steps != int(self.T / self.dt) + 1:
            raise ValueError("anchor_time_series length must equal num time steps")

        t_grid = np.linspace(0.0, self.T, T_steps)
        z_list: List[np.ndarray] = []

        free = self._initial_positions(self.n0)            # (n0,d)
        anchors: List[np.ndarray] = []                     # list of (K,d)
        masses:  List[float] = []                          # parallel list

        # ---------- record initial frame (free only) -------------------
        z_list.append(free.copy())

        # iterate over steps 0 .. T_steps-2  (since slice t+1 appended)
        for t in range(T_steps - 1):
            # ─── build state & weighted copy for attention ────────────
            state = np.vstack([free] + anchors)            # (M + p·K, d)
            weighted = state.copy()
            if masses:
                start = self.n0
                for mass, block in zip(masses, anchors):
                    weighted[start:start+self.K] *= mass
                    start += self.K

            # ─── Euler step -------------------------------------------
            dz = self._step(weighted)
            free += self.dt * dz[:self.n0]
            free /= np.linalg.norm(free, axis=1, keepdims=True)

            # update existing anchors in‑place
            off = self.n0
            for i in range(len(anchors)):
                anchors[i] += self.dt * dz[off:off+self.K]
                anchors[i] /= np.linalg.norm(anchors[i], axis=1, keepdims=True)
                off += self.K

            # ─── append new slice (unit norm) --------------------------
            new_anchor = self._ts[t+1].copy()
            new_anchor /= np.linalg.norm(new_anchor, axis=1, keepdims=True)
            anchors.append(new_anchor)
            masses.append(self._w[t+1])

            # FIFO cap
            if self.max_history is not None and len(anchors) > self.max_history:
                anchors.pop(0)
                masses.pop(0)

            # record frame after append
            z_list.append(np.vstack([free] + anchors))

        return z_list, t_grid


# ======================================================================
#  LEGACY WRAPPERS
# ======================================================================

def simulate_particles(**kwargs):
    return Simulator(**kwargs).simulate()


def simulate_particles_with_anchor(*, anchor, anchor_weight=None, verbose=True, **kwargs):
    return AnchorSimulator(anchor=anchor, anchor_weight=anchor_weight, **kwargs).simulate(verbose=verbose)


# ======================================================================
#  SELF‑TEST  — matches original tensor demo
# ======================================================================
if __name__ == "__main__":
    myseed = 12

    # free swarm
    z_list, _ = simulate_particles(seed=myseed)

    # build anchor from final frame
    pre_anchor = z_list[-1].mean(axis=0)
    anchor = pre_anchor / np.linalg.norm(pre_anchor)

    # anchored swarm with weight 1.0 (to match old code)
    z2_list, _ = simulate_particles_with_anchor(
        seed=myseed,
        anchor=anchor,
        anchor_weight=1.0,
    )

    # final comparison print
    print("z2_list[-1][-5:] ==>\n", z2_list[-1][-5:])
