from __future__ import annotations

"""
particle_simulators.py  — population‑aware & numerically stable
---------------------------------------------------------------

* **List trajectory** (z_list[t]  → (N_t, d)), same as previous revision.
* **Stable soft‑max** to avoid overflow when anchor_weight is large.
* **Self‑test** passes anchor_weight = 1.0, so results match the old tensor code.
"""

from typing import List, Optional, Tuple, Union, Callable
import torch
import pdb

__all__ = [
    "Simulator",
    "AnchorSimulator",
    "MobileAnchorSimulator",
    "InterJectorSimulator",
    "simulate_particles",
    "simulate_particles_with_anchor",
    "stack_trajectory",
]

dtype_default = torch.float64  # NumPy と同じ既定精度に揃える

# ----------------------------------------------------------------------
#  Utility: numerically stable row‑softmax
# ----------------------------------------------------------------------

def _row_softmax(x: torch.Tensor) -> torch.Tensor:
    x_shift = x - x.max(dim=1, keepdim=True).values  # subtract row‑wise max
    e = torch.exp(x_shift)
    return e / e.sum(dim=1, keepdim=True)


# ----------------------------------------------------------------------
#  Helper: convert list‑trajectory back to dense tensor
# ----------------------------------------------------------------------

def stack_trajectory(z_list: List[torch.Tensor]) -> torch.Tensor:
    n0 = z_list[0].shape[0]
    if not all(f.shape[0] == n0 for f in z_list):
        raise ValueError("Cannot stack – particle count varies across frames")
    return torch.stack(z_list, dim=1)  # (N, T, d)


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
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = dtype_default,
    ) -> None:
        self.n0 = n
        self.T = T
        self.dt = dt
        self.d = d
        self.beta = beta
        self.half_sph = half_sph
        self.seed = seed
        self.device = torch.device(device)
        self.dtype = dtype
        # RNG
        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)

    # ------------------------------------------------------------------
    def simulate(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        print(
            f"Calling simulate_particles with (population mode):\n"
            f"    n0={self.n0}, T={self.T}, dt={self.dt}, d={self.d}, beta={self.beta},"
            f" half_sph={self.half_sph}, seed={self.seed}"
        )
        num_steps = int(self.T / self.dt) + 1
        t_grid = torch.linspace(0.0, self.T, num_steps, device=self.device, dtype=self.dtype)
        z_list: List[torch.Tensor] = []

        z_curr = self._initial_positions(self.n0)

        for _ in range(num_steps):
            z_list.append(z_curr.clone())
            dz = self._step(z_curr)
            z_next = z_curr + self.dt * dz
            z_next = _normalize_rows(z_next)
            z_curr = z_next
        return z_list, t_grid

    # ------------------------------------------------------------------
    def _step(self, z_slice: torch.Tensor) -> torch.Tensor:
        V = torch.eye(self.d, device=self.device, dtype=self.dtype)
        A = torch.eye(self.d, device=self.device, dtype=self.dtype)
        Az = (A @ z_slice.T).T
        attn = _row_softmax(self.beta * (Az @ Az.T))
        return attn @ (V @ z_slice.T).T

    # ------------------------------------------------------------------
    def _initial_positions(self, n: int) -> torch.Tensor:
        z0 = torch.randn((n, self.d), generator=self._gen, device=self.device, dtype=self.dtype)
        z0 = _normalize_rows(z0)
        if self.half_sph and self.d == 3:
            z0[z0[:, 2] < 0] *= -1
        return z0


# ======================================================================
#  SUBCLASS – MULTIPLE IMMOBILE ANCHORS
# ======================================================================
class AnchorSimulator(Simulator):
    """Swarm with one or more immobile anchors."""

    def __init__(
        self,
        *,
        anchor: Union[torch.Tensor, List, Tuple, "numpy.ndarray"],
        anchor_weight: Optional[Union[float, List, "numpy.ndarray", torch.Tensor]] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)

        # ---------- accept single anchor or stack of anchors ------------
        anchor = torch.as_tensor(anchor, dtype=self.dtype, device=self.device)
        if anchor.ndim == 1:
            anchor = anchor.unsqueeze(0)           # (d,) → (1,d)
        if anchor.ndim != 2 or anchor.shape[1] != self.d:
            raise ValueError(f"`anchor` must have shape (M, {self.d}) or ({self.d},)")
        anchor = _normalize_rows(anchor)
        self.anchors = anchor                      # (M,d)
        self.M = anchor.shape[0]

        # ---------- anchor weights --------------------------------------
        if anchor_weight is None:
            anchor_weight = self.n0 / self.M
        anchor_weight = torch.as_tensor(anchor_weight, dtype=self.dtype, device=self.device)
        if anchor_weight.ndim == 0:
            anchor_weight = anchor_weight.repeat(self.M)
        if anchor_weight.shape != (self.M,):
            raise ValueError("`anchor_weight` must be a scalar or shape (M,)")
        self.anchor_weight = anchor_weight

    # ------------------------------------------------------------------
    def simulate(self, verbose: bool = True) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if verbose:
            print(
                """Calling simulate_particles with (multi-anchor, population mode):
                free_n0={self.n0}, M={self.M}, T={self.T}, dt={self.dt}, d={self.d}, beta={self.beta},
                half_sph={self.half_sph}, seed={self.seed}"""
            )
            print(f"Anchors (rows):\n {self.anchors}\n weights: {self.anchor_weight}")

        num_steps = int(self.T / self.dt) + 1
        t_grid = torch.linspace(0.0, self.T, num_steps, device=self.device, dtype=self.dtype)
        z_list: List[torch.Tensor] = []

        # concatenate initial free particles + anchors
        z_curr = torch.cat([self._initial_positions(self.n0), self.anchors], dim=0)

        for _ in range(num_steps):
            z_list.append(z_curr.clone())

            dz = self._anchored_step(z_curr)
            z_next_free = z_curr[:-self.M] + self.dt * dz[:-self.M]
            z_next_free = _normalize_rows(z_next_free)

            # anchors remain fixed
            z_curr = torch.cat([z_next_free, self.anchors], dim=0)

        return z_list, t_grid

    # ------------------------------------------------------------------
    def _anchored_step(self, z_slice: torch.Tensor) -> torch.Tensor:
        """Apply anchor weights only to the last M rows."""
        V = torch.eye(self.d, device=self.device, dtype=self.dtype)
        A = torch.eye(self.d, device=self.device, dtype=self.dtype)

        z4 = z_slice.clone()
        z4[-self.M:] *= self.anchor_weight.view(-1, 1)

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
        anchor_time_series: Optional["numpy.ndarray" | torch.Tensor] = None,  # (T,M,d)
        anchor_ode: Optional[Callable[[float, torch.Tensor], torch.Tensor]] = None,
        anchor_weight: Optional[Union[float, List, "numpy.ndarray", torch.Tensor]] = None,
        **base_kwargs,
    ) -> None:

        if (anchor_time_series is None) == (anchor_ode is None):
            raise ValueError("Provide exactly one of anchor_time_series or anchor_ode.")

        if anchor_time_series is not None:
            ts = torch.as_tensor(anchor_time_series, dtype=base_kwargs.get("dtype", dtype_default), device=base_kwargs.get("device", "cpu"))
            if ts.ndim != 3:
                raise ValueError("anchor_time_series must be (T,M,d)")
            anchor_init = ts[0]
            self._anchor_ts = ts
            self._use_ts = True
        else:
            # placeholder; real initial anchor will come from ODE at t=0
            d = base_kwargs.get("d", 3)
            anchor_init = torch.zeros((1, d), dtype=base_kwargs.get("dtype", dtype_default), device=base_kwargs.get("device", "cpu"))
            self._anchor_ode = anchor_ode
            self._use_ts = False

        super().__init__(anchor=anchor_init, anchor_weight=anchor_weight, **base_kwargs)

    # ------------------------------------------------------------------
    def simulate(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        num_steps = int(self.T / self.dt) + 1
        t_grid = torch.linspace(0.0, self.T, num_steps, device=self.device, dtype=self.dtype)
        z_list: List[torch.Tensor] = []

        free = self._initial_positions(self.n0)      # (n0,d)
        anc = self.anchors.clone()                   # (M,d)
        state = torch.cat([free, anc], dim=0)

        for step in range(num_steps):
            z_list.append(state.clone())

            # ------ free particles step --------------------------------
            attn_state = state.clone()
            attn_state[-self.M:] *= self.anchor_weight.view(-1, 1)
            dz = self._step(attn_state)
            free_next = state[:-self.M] + self.dt * dz[:-self.M]
            free_next = _normalize_rows(free_next)

            # ------ anchor update --------------------------------------
            if self._use_ts:
                anc_next = self._anchor_ts[step]
            else:
                anc_next = self._anchor_ode(float(t_grid[step]), state[-self.M:])
            anc_next = _normalize_rows(anc_next)

            state = torch.cat([free_next, anc_next], dim=0)

        return z_list, t_grid


# ======================================================================
#  SUBCLASS – INTERJECTOR (growing, weighted anchor history)
# ======================================================================
class InterJectorSimulator(Simulator):
    """Free block interacts with a growing FIFO history of anchor slices."""

    def __init__(
        self,
        *,
        anchor_time_series: "numpy.ndarray" | torch.Tensor,  # (T,K,d)
        anchor_weight: Union[float, "numpy.ndarray", torch.Tensor] = 1.0,
        max_history: Optional[int] = None,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)

        ts = torch.as_tensor(anchor_time_series, dtype=self.dtype, device=self.device)
        if ts.ndim != 3 or ts.shape[2] != self.d:
            raise ValueError("anchor_time_series must have shape (T, K, d)")
        self._ts = ts                      # (T,K,d)
        self.K = ts.shape[1]
        self.max_history = max_history

        w = torch.as_tensor(anchor_weight, dtype=self.dtype, device=self.device)
        if w.ndim == 0:
            w = w.repeat(ts.shape[0])      # scalar → (T,)
        if w.shape != (ts.shape[0],):
            raise ValueError("anchor_weight must be scalar or shape (T,)")
        self._w = w

    # ------------------------------------------------------------------
    def simulate(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        T_steps = self._ts.shape[0]
        if T_steps != int(self.T / self.dt) + 1:
            raise ValueError("anchor_time_series length must equal num time steps")

        t_grid = torch.linspace(0.0, self.T, T_steps, device=self.device, dtype=self.dtype)
        z_list: List[torch.Tensor] = []

        free = self._initial_positions(self.n0)            # (n0,d)
        anchors: List[torch.Tensor] = []                   # list of (K,d)
        masses:  List[torch.Tensor] = []                   # parallel list

        # ---------- record initial frame (free only) -------------------
        z_list.append(free.clone())

        # iterate over steps 0 .. T_steps-2  (since slice t+1 appended)
        for t in range(T_steps - 1):
            # ─── build state & weighted copy for attention ────────────
            state = torch.cat([free] + anchors, dim=0) if anchors else free.clone()
            weighted = state.clone()
            if masses:
                start = self.n0
                for mass, block in zip(masses, anchors):
                    weighted[start:start+self.K] *= mass
                    start += self.K

            # ─── Euler step -------------------------------------------
            dz = self._step(weighted)
            free = free + self.dt * dz[:self.n0]
            free = _normalize_rows(free)

            # update existing anchors in‑place
            off = self.n0
            for i in range(len(anchors)):
                anchors[i] = anchors[i] + self.dt * dz[off:off+self.K]
                anchors[i] = _normalize_rows(anchors[i])
                off += self.K

            # ─── append new slice (unit norm) --------------------------
            new_anchor = _normalize_rows(self._ts[t+1].clone())
            anchors.append(new_anchor)
            masses.append(self._w[t+1])

            # FIFO cap
            if self.max_history is not None and len(anchors) > self.max_history:
                anchors.pop(0)
                masses.pop(0)

            # record frame after append
            z_list.append(torch.cat([free] + anchors, dim=0))

        return z_list, t_grid


# ======================================================================
#  Helper function(s)
# ======================================================================

def _normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x, dim=1, keepdim=True)


# ======================================================================
#  LEGACY WRAPPERS
# ======================================================================

def simulate_particles(**kwargs):
    return Simulator(**kwargs).simulate()


def simulate_particles_with_anchor(*, anchor, anchor_weight=None, verbose=True, **kwargs):
    return AnchorSimulator(anchor=anchor, anchor_weight=anchor_weight, **kwargs).simulate(verbose=verbose)


# ======================================================================
#  SELF‑TEST  — matches original demo (values will differ due to float64)
# ======================================================================
if __name__ == "__main__":
    myseed = 12

    # free swarm
    z_list, _ = simulate_particles(seed=myseed)

    # build anchor from final frame
    pre_anchor = z_list[-1].mean(dim=0)
    anchor = pre_anchor / torch.linalg.norm(pre_anchor)

    # anchored swarm with weight 1.0 (to match old code)
    z2_list, _ = simulate_particles_with_anchor(
        seed=myseed,
        anchor=anchor,
        anchor_weight=1.0,
    )

    print("z2_list[-1][-5:] ==>\n", z2_list[-1][-5:])
