import os, numpy as np, matplotlib.pyplot as plt, imageio
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d
from datetime import datetime
from tqdm import trange
import random # for reproducibility
import simulators as sim 


# ──────────────────────────────────────────────────────────────────────────
# 2) RENDER   (builds its own interpolators; no extra args)
# ──────────────────────────────────────────────────────────────────────────
def render(
        z, d, integration_time,
        *,                           # keyword-only from here
        rootdrivepath = '/content/drive/MyDrive/figs',
        color  = '#3658bf',
        movie         = True,
        fps           = 10
):
    n, num_steps, _ = z.shape
    dt = integration_time[1] - integration_time[0]

    # build cubic splines (local to render)
    interp_x = [interp1d(integration_time, z[i,:,0], 'cubic') for i in range(n)]
    interp_y = [interp1d(integration_time, z[i,:,1], 'cubic') for i in range(n)] if d>1 else None
    interp_z = [interp1d(integration_time, z[i,:,2], 'cubic') for i in range(n)] if d==3 else None

    # output paths
    now       = datetime.now().strftime('%H-%M-%S')
    dir_path  = os.path.join(rootdrivepath, 'circle' if d==2 else 'sphere', 'beta=1')
    os.makedirs(dir_path, exist_ok=True)
    prefix    = os.path.join(dir_path, now)
    gif_path  = prefix + '_movie.gif'

    # axis limits + 10 % margin
    x_min, x_max = z[:,:,0].min(), z[:,:,0].max(); pad = .1*(x_max-x_min); x_min-=pad; x_max+=pad
    if d>1:
        y_min, y_max = z[:,:,1].min(), z[:,:,1].max(); pad=.1*(y_max-y_min); y_min-=pad; y_max+=pad
    if d==3:
        z_min, z_max = z[:,:,2].min(), z[:,:,2].max(); pad=.1*(z_max-z_min); z_min-=pad; z_max+=pad

    imgs=[]
    for t in trange(num_steps, desc='render'):
        if d==2:
            fig, ax = plt.subplots(figsize=(5,5))
            ax.axis('off'); ax.set_aspect('equal')
            ax.set(xlim=(x_min,x_max), ylim=(y_min,y_max))
            ax.set_title(f'$t={t*dt:.2f}$')
            ax.scatter([fx(integration_time)[t] for fx in interp_x],
                       [fy(integration_time)[t] for fy in interp_y],
                       s=30,c=color,edgecolors='black')
            if t:
                for i in range(n):
                    ax.plot(interp_x[i](integration_time)[:t+1],
                            interp_y[i](integration_time)[:t+1],
                            c=color,lw=.4,ls='dashed')
        else:
            fig = plt.figure()
            ax  = fig.add_subplot(111, projection='3d')
            ax.axis('off')
            ax.set(xlim=(x_min,x_max), ylim=(y_min,y_max), zlim=(z_min,z_max))
            ax.set_title(f'$t={t*dt:.2f}$')
            ax.scatter([fx(integration_time)[t] for fx in interp_x],
                       [fy(integration_time)[t] for fy in interp_y],
                       [fz(integration_time)[t] for fz in interp_z],
                       s=25,c=color,edgecolors='black')
            if t:
                for i in range(n):
                    ax.plot(interp_x[i](integration_time)[:t+1],
                            interp_y[i](integration_time)[:t+1],
                            interp_z[i](integration_time)[:t+1],
                            c=color,lw=.4,ls='dashed')
            ax.view_init(elev=10, azim=30+30*(t/num_steps-.5))

        png_path=f'{prefix}_{t}.png'
        plt.savefig(png_path,dpi=300,bbox_inches='tight'); plt.close()
        if movie: imgs.append(imageio.imread(png_path))
        os.remove(png_path)

    if movie:
        imageio.mimsave(gif_path, imgs, fps=fps)
        print('GIF saved →', gif_path)


# ──────────────────────────────────────────────────────────────────────────
# QUICK DEMO
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    z, t_grid = sim.simulate_particles()        # you’ll edit only this
    render(z, d=3, integration_time=t_grid) # nothing else to touch
