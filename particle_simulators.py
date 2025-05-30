"""
particle_simulators.py
---------------------

**Fixed** OOP refactor of the particle simulators.

• `Simulator`          – free swarm (unchanged).
• `AnchorSimulator`    – inherits from `Simulator`, adds one immobile anchor.
• Legacy wrappers      – `simulate_particles`, `simulate_particles_with_anchor`.

Key bug‑fix: `AnchorSimulator` now keeps `self.n` as the number of *free*
particles and allocates the anchor separately, eliminating the broadcast error.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

__all__ = [
    "Simulator",
    "AnchorSimulator",
    "simulate_particles",
    "simulate_particles_with_anchor",
]

# ====================================================================== #
#  BASE‑CLASS (FREE SWARM)
# ====================================================================== #
class Simulator:
    """N‑particle self‑attention dynamics on the *d*‑sphere."""

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
        self.n = n  # number of particles
        self.T = T
        self.dt = dt
        self.d = d
        self.beta = beta
        self.half_sph = half_sph
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the simulation (identical to old `simulate_particles`)."""
        print(
            f"Calling simulate_particles with:\n"
            f"    n={self.n}, T={self.T}, dt={self.dt}, d={self.d}, beta={self.beta},"
            f" half_sph={self.half_sph}, seed={self.seed}"
        )
        num_steps = int(self.T / self.dt) + 1
        t_grid = np.linspace(0.0, self.T, num_steps)

        z = np.zeros((self.n, num_steps, self.d))
        z[:, 0] = self._initial_positions()

        for i in range(num_steps - 1):
            dz = self._step(z[:, i])
            z[:, i + 1] = z[:, i] + self.dt * dz
            z[:, i + 1] /= np.linalg.norm(z[:, i + 1], axis=1, keepdims=True)

        return z, t_grid

    # ------------------------------------------------------------------ #
    def _step(self, z_slice: np.ndarray) -> np.ndarray:
        """One Euler step of the default ODE (override for custom dynamics)."""
        V, A = np.eye(self.d), np.eye(self.d)
        Az = (A @ z_slice.T).T
        attn = np.exp(self.beta * (Az @ Az.T))
        attn /= attn.sum(axis=1, keepdims=True)
        return attn @ (V @ z_slice.T).T

    # ------------------------------------------------------------------ #
    def _initial_positions(self) -> np.ndarray:
        """Random points on the unit *d*‑sphere (half‑sphere optional)."""
        z0 = self._rng.normal(size=(self.n, self.d))
        z0 /= np.linalg.norm(z0, axis=1, keepdims=True)
        if self.half_sph and self.d == 3:
            z0[z0[:, 2] < 0] *= -1
        return z0


# ====================================================================== #
#  SUBCLASS (IMMOBILE ANCHOR)
# ====================================================================== #
class AnchorSimulator(Simulator):
    """Simulator with an additional immobile anchor particle."""

    def __init__(
        self,
        *,
        anchor: Union[np.ndarray, list, tuple],
        anchor_weight: Optional[float] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)

        self.anchor = np.asarray(anchor, dtype=float)
        assert self.anchor.ndim == 1 and self.anchor.size == self.d, (
            f"Anchor must be 1‑D of length {self.d}"
        )
        self.anchor /= np.linalg.norm(self.anchor)

        self.anchor_weight = anchor_weight if anchor_weight is not None else self.n

    # ------------------------------------------------------------------ #
    def simulate(self) -> Tuple[np.ndarray, np.ndarray]:
        print(
            f"Calling simulate_particles with (anchor version):\n"
            f"    free_n={self.n}, T={self.T}, dt={self.dt}, d={self.d}, beta={self.beta},"
            f" half_sph={self.half_sph}, seed={self.seed}"
        )
        print(f"Anchor point: {self.anchor}, weight: {self.anchor_weight}")

        num_steps = int(self.T / self.dt) + 1
        t_grid = np.linspace(0.0, self.T, num_steps)

        total_n = self.n + 1  # free particles + anchor
        z = np.zeros((total_n, num_steps, self.d))

        z[:-1, 0] = self._initial_positions()  # free particles
        z[-1, 0] = self.anchor                # anchor

        for i in range(num_steps - 1):
            dz = self._anchored_step(z[:, i])
            z[:-1, i + 1] = z[:-1, i] + self.dt * dz[:-1]
            z[:-1, i + 1] /= np.linalg.norm(z[:-1, i + 1], axis=1, keepdims=True)
            z[-1, i + 1] = self.anchor  # keep anchor fixed

        return z, t_grid

    # ------------------------------------------------------------------ #
    def _anchored_step(self, z_slice: np.ndarray) -> np.ndarray:
        V, A = np.eye(self.d), np.eye(self.d)
        z4Attn = z_slice.copy()
        z4Attn[-1] *= self.anchor_weight  # emphasise anchor in attention
        Az = (A @ z4Attn.T).T
        attn = np.exp(self.beta * (Az @ Az.T))
        attn /= attn.sum(axis=1, keepdims=True)
        return attn @ (V @ z_slice.T).T


# ====================================================================== #
#  LEGACY PROCEDURAL WRAPPERS
# ====================================================================== #

def simulate_particles(**kwargs):
    """Drop‑in replacement for the old free‑swarm function."""
    return Simulator(**kwargs).simulate()


def simulate_particles_with_anchor(*, anchor, anchor_weight=None, **kwargs):
    """Drop‑in replacement for the old anchored‑swarm function."""
    return AnchorSimulator(anchor=anchor, anchor_weight=anchor_weight, **kwargs).simulate()


# ====================================================================== #
#  QUICK SELF‑TEST (exactly the user's snippet)
# ====================================================================== #
if __name__ == "__main__":
    myseed = 12

    # first run – free swarm
    z, t_grid = simulate_particles(seed=myseed)

    # derive anchor from final positions
    pre_anchor = np.mean(z[:, -1, :], axis=0)
    anchor = pre_anchor / np.linalg.norm(pre_anchor)

    # second run – anchored swarm
    anchor_weight = 1.0
    T = 15
    z2, t_grid2 = simulate_particles_with_anchor(
        seed=myseed,
        anchor=anchor,
        T=T,
        anchor_weight=anchor_weight,
    )

    # confirmation output
    print("z2[-5:, -1, :] ==>\n", z2[-5:, -1, :])
