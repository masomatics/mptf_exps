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

from typing import List, Optional, Tuple, Union
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
    def simulate(self) -> Tuple[List[np.ndarray], np.ndarray]:
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
#  SUBCLASS – continuous interjection of the input point 
# ======================================================================
class InterJectorSimulator(Simulator):
    """
    Swarm that injects an external time-series C[t] (shape K×d) at every step.
    `interject_ts` must have shape (num_steps, K, d).
    """

    def __init__(self, *, interject_ts, ts_weight=None, **base_kwargs):
        super().__init__(**base_kwargs)

        self.interject_ts = np.asarray(interject_ts, dtype=float)
        # --- accept (T,d) by auto-expanding to (T,1,d) --------------------
        if self.interject_ts.ndim == 2:              # (T, d)  →  (T, 1, d)
            self.interject_ts = self.interject_ts[:, None, :]

        self.ts_weight = ts_weight        
        assert self.interject_ts.ndim == 3 and self.interject_ts.shape[2] == self.d, (
            "expected interject_ts with shape (T, K, d)"
        )
        self.interject_ts = interject_ts


    def simulate(self):
        num_steps = int(self.T / self.dt) + 1
        t_grid = np.linspace(0.0, self.T, num_steps)

        z_list = []
        z_curr = self._initial_positions(self.n0)

        for t in range(num_steps):
            z_list.append(z_curr.copy())

            # ----- Euler step for the current population -----
            dz = self._step(z_curr)
            z_next = z_curr + self.dt * dz
            z_next /= np.linalg.norm(z_next, axis=1, keepdims=True)

            # ----- append the K fresh particles for step t -----
            inj = self.interject_ts[t]              # (K, d)
            if self.ts_weight is not None:          # bias them in attention
                inj_attn = inj * self.ts_weight
                # store two copies: one for state, one scaled for attention
                z_curr = np.concatenate((z_next, inj), axis=0)
                z_attn = np.concatenate((z_next, inj_attn), axis=0)
            else:
                z_curr = np.concatenate((z_next, inj), axis=0)
                z_attn = z_curr

            # replace for the next iteration
            z_curr_for_step = z_attn  # used by _step at the next loop
            z_curr = z_curr          # used by state accumulation

        return z_list, t_grid


# ======================================================================
#  LEGACY WRAPPERS
# ======================================================================

def simulate_particles(**kwargs):
    return Simulator(**kwargs).simulate()


def simulate_particles_with_anchor(*, anchor, anchor_weight=None, **kwargs):
    return AnchorSimulator(anchor=anchor, anchor_weight=anchor_weight, **kwargs).simulate()


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
