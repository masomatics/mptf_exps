import os, numpy as np, matplotlib.pyplot as plt, imageio
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange
import random # for reproducibility
# import simulators as sim 
import particle_simulators as psim 

import pdb


# # ──────────────────────────────────────────────────────────────────────────
# # 2) RENDER   (builds its own interpolators; no extra args)
# # ──────────────────────────────────────────────────────────────────────────
# def render(
#         z, d, integration_time,
#         *,                           # keyword-only from here
#         rootdrivepath = '/content/drive/MyDrive/figs',
#         color  = '#3658bf',
#         movie         = True,
#         fps           = 10
# ):
#     n, num_steps, _ = z.shape
#     dt = integration_time[1] - integration_time[0]

#     # build cubic splines (local to render)
#     interp_x = [interp1d(integration_time, z[i,:,0], 'cubic') for i in range(n)]
#     interp_y = [interp1d(integration_time, z[i,:,1], 'cubic') for i in range(n)] if d>1 else None
#     interp_z = [interp1d(integration_time, z[i,:,2], 'cubic') for i in range(n)] if d==3 else None

#     # output paths
#     now       = datetime.now().strftime('%H-%M-%S')
#     dir_path  = os.path.join(rootdrivepath, 'circle' if d==2 else 'sphere', 'beta=1')
#     os.makedirs(dir_path, exist_ok=True)
#     prefix    = os.path.join(dir_path, now)
#     gif_path  = prefix + '_movie.gif'

#     # axis limits + 10 % margin
#     x_min, x_max = z[:,:,0].min(), z[:,:,0].max(); pad = .1*(x_max-x_min); x_min-=pad; x_max+=pad
#     if d>1:
#         y_min, y_max = z[:,:,1].min(), z[:,:,1].max(); pad=.1*(y_max-y_min); y_min-=pad; y_max+=pad
#     if d==3:
#         z_min, z_max = z[:,:,2].min(), z[:,:,2].max(); pad=.1*(z_max-z_min); z_min-=pad; z_max+=pad

#     imgs=[]
#     for t in trange(num_steps, desc='render'):
#         if d==2:
#             fig, ax = plt.subplots(figsize=(5,5))
#             ax.axis('off'); ax.set_aspect('equal')
#             ax.set(xlim=(x_min,x_max), ylim=(y_min,y_max))
#             ax.set_title(f'$t={t*dt:.2f}$')
#             ax.scatter([fx(integration_time)[t] for fx in interp_x],
#                        [fy(integration_time)[t] for fy in interp_y],
#                        s=30,c=color,edgecolors='black')
#             if t:
#                 for i in range(n):
#                     ax.plot(interp_x[i](integration_time)[:t+1],
#                             interp_y[i](integration_time)[:t+1],
#                             c=color,lw=.4,ls='dashed')
#         else:
#             fig = plt.figure()
#             ax  = fig.add_subplot(111, projection='3d')
#             ax.axis('off')
#             ax.set(xlim=(x_min,x_max), ylim=(y_min,y_max), zlim=(z_min,z_max))
#             ax.set_title(f'$t={t*dt:.2f}$')
#             ax.scatter([fx(integration_time)[t] for fx in interp_x],
#                        [fy(integration_time)[t] for fy in interp_y],
#                        [fz(integration_time)[t] for fz in interp_z],
#                        s=25,c=color,edgecolors='black')
#             if t:
#                 for i in range(n):
#                     ax.plot(interp_x[i](integration_time)[:t+1],
#                             interp_y[i](integration_time)[:t+1],
#                             interp_z[i](integration_time)[:t+1],
#                             c=color,lw=.4,ls='dashed')
#             ax.view_init(elev=10, azim=30+30*(t/num_steps-.5))

#         png_path=f'{prefix}_{t}.png'
#         plt.savefig(png_path,dpi=300,bbox_inches='tight'); plt.close()
#         if movie: imgs.append(imageio.imread(png_path))
#         os.remove(png_path)

#     if movie:
#         imageio.mimsave(gif_path, imgs, fps=fps)
#         print('GIF saved →', gif_path)

# ──────────────────────────────────────────────────────────────────────────
# RENDER  – flexible visualiser for 2-D / 3-D trajectories
#   • free block (0…m-1) gets cubic-spline trails
#   • anchor history (m…end) plotted step-by-step
#   • variable population supported (list of frames)
# ──────────────────────────────────────────────────────────────────────────
import os, numpy as np, matplotlib.pyplot as plt, imageio
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange


def render(
    z, d, integration_time,
    *,
    rootdrivepath='./figs',
    color='#3658bf',
    color_rest='#d94f4f',
    m=None,                    # size of free block
    movie=True,
    fps=10,
    interpolate=True,
    title=None,
):
    # ---------- input normalisation (same logic as before) --------------
    is_list = isinstance(z, list)

    if is_list and m is None:
        raise ValueError("For a growing list you must provide `m`.")

    if is_list and all(f.shape[0] == z[0].shape[0] for f in z):
        z = np.stack(z, axis=1)
        is_list = False

    if not is_list:
        n, num_steps, _ = z.shape
        x_min, x_max = z[:, :, 0].min(), z[:, :, 0].max()
        if d > 1: y_min, y_max = z[:, :, 1].min(), z[:, :, 1].max()
        if d == 3: z_min, z_max = z[:, :, 2].min(), z[:, :, 2].max()
    else:
        num_steps = len(z)
        all_xyz   = np.concatenate(z, axis=0)
        x_min, x_max = all_xyz[:, 0].min(), all_xyz[:, 0].max()
        if d > 1: y_min, y_max = all_xyz[:, 1].min(), all_xyz[:, 1].max()
        if d == 3: z_min, z_max = all_xyz[:, 2].min(), all_xyz[:, 2].max()

    pad = .1 * (x_max - x_min); x_min -= pad; x_max += pad
    if d > 1:
        pad = .1 * (y_max - y_min); y_min -= pad; y_max += pad
    if d == 3:
        pad = .1 * (z_max - z_min); z_min -= pad; z_max += pad
    dt = integration_time[1] - integration_time[0]

    # ---------- colour palette -----------------------------------------
    def palette(n_rows):
        p = np.full(n_rows, color_rest)
        p[:m] = color
        return p

    # ---------- splines for free block ---------------------------------
    if interpolate:
        if not is_list:
            free_n = m if m is not None else z.shape[0]
            interp_x = [interp1d(integration_time, z[i, :, 0], 'cubic') for i in range(free_n)]
            interp_y = [interp1d(integration_time, z[i, :, 1], 'cubic') for i in range(free_n)] if d > 1 else None
            interp_z = [interp1d(integration_time, z[i, :, 2], 'cubic') for i in range(free_n)] if d == 3 else None
        else:
            free_block = np.stack([frame[:m] for frame in z], axis=1)
            interp_x = [interp1d(integration_time, free_block[i, :, 0], 'cubic') for i in range(m)]
            interp_y = [interp1d(integration_time, free_block[i, :, 1], 'cubic') for i in range(m)] if d > 1 else None
            interp_z = [interp1d(integration_time, free_block[i, :, 2], 'cubic') for i in range(m)] if d == 3 else None

    # ---------- paths / folders ----------------------------------------
    now = datetime.now().strftime('%H-%M-%S') + (f'_{title}' if title else '')
    folder = os.path.join(rootdrivepath, 'circle' if d == 2 else 'sphere')
    os.makedirs(folder, exist_ok=True)
    prefix = os.path.join(folder, now)
    gif_path = prefix + '_movie.gif'

    imgs, prev_frame = [], None
    for t in trange(num_steps, desc='render'):
        # positions -----------------------------------------------------
        if not is_list:
            xs = z[:, t, 0]; ys = z[:, t, 1] if d > 1 else None
            zs = z[:, t, 2] if d == 3 else None
        else:
            frame = z[t]
            xs, ys, zs = frame[:, 0], None, None
            if d > 1: ys = frame[:, 1]
            if d == 3: zs = frame[:, 2]

        cols = palette(len(xs))

        # plotting ------------------------------------------------------
        if d == 2:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.axis('off'); ax.set_aspect('equal')
            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))
            ax.set_title(f'$t={t*dt:.2f}$')
            ax.scatter(xs, ys, s=30, c=cols, edgecolors='black')

            # spline trails for free block
            if t and interpolate:
                for i in range(m):
                    ax.plot(interp_x[i](integration_time)[:t+1],
                            interp_y[i](integration_time)[:t+1],
                            c=cols[i], lw=.4, alpha=.35)
            # straight segments for anchors (guard on length mismatch)
            if prev_frame is not None:
                shared = min(len(prev_frame), len(xs))
                for i in range(m, shared):
                    ax.plot([prev_frame[i, 0], xs[i]],
                            [prev_frame[i, 1], ys[i]],
                            c=cols[i], lw=.4, alpha=.35)

        else:  # 3-D
            fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
            ax.axis('off')
            ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max), zlim=(z_min, z_max))
            ax.set_title(f'$t={t*dt:.2f}$')
            ax.scatter(xs, ys, zs, s=25, c=cols, edgecolors='black')

            if t and interpolate:
                for i in range(m):
                    ax.plot(interp_x[i](integration_time)[:t+1],
                            interp_y[i](integration_time)[:t+1],
                            interp_z[i](integration_time)[:t+1],
                            c=cols[i], lw=.4, alpha=.35)
            if prev_frame is not None:
                shared = min(len(prev_frame), len(xs))
                for i in range(m, shared):
                    ax.plot([prev_frame[i, 0], xs[i]],
                            [prev_frame[i, 1], ys[i]],
                            [prev_frame[i, 2], zs[i]],
                            c=cols[i], lw=.4, alpha=.35)
            ax.view_init(elev=10, azim=30 + 30 * (t / num_steps - .5))

        # save frame ----------------------------------------------------
        png = f'{prefix}_{t}.png'
        plt.savefig(png, dpi=300, bbox_inches='tight')
        plt.close()
        if movie:
            imgs.append(imageio.imread(png))
        os.remove(png)

        prev_frame = np.column_stack([xs, ys, zs]) if d == 3 else np.column_stack([xs, ys])

    if movie:
        imageio.mimsave(gif_path, imgs, fps=fps)
        print('GIF saved →', gif_path)


if __name__ == '__main__':
    myseed = 12
    # z, t_grid = sim.simulate_particles(seed=myseed)
    z, t_grid = psim.simulate_particles(seed=myseed)


    pre_anchor = np.mean(z[:, -1, :], axis=0)
    anchor = pre_anchor / np.linalg.norm(pre_anchor) 

    anchor_weight = 1.0
    T= 15
    # z2, t_grid2 = sim.simulate_particles_with_anchor(seed=myseed, anchor=anchor, T=T, anchor_weight=anchor_weight )
    z2, t_grid2 = psim.simulate_particles_with_anchor(seed=myseed, anchor=anchor, T=T, anchor_weight=anchor_weight )

    
    print(z2[-5:, -1, :])

    render(z2, 3, t_grid2,rootdrivepath='./figs',
        color='#3658bf',
        movie=True,
        fps=10,
        m=64,
        color_rest='#d94f4f',
        interpolate=True,
        title=f"anchor_weight={anchor_weight}")    

