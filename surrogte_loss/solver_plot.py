import numpy as np
import matplotlib.pyplot as plt 
import scipy.io 

root_dir = './results/solver'
mu, nu = 0.2, 4.0 

plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True

fig, axes = plt.subplots(2, 4, figsize=(12, 5.6))

for i, nz in enumerate([64*6, 128*6]):
    for j, nx in enumerate([8, 16, 32, 64]):
        data = scipy.io.loadmat(f'{root_dir}/MonLipNet-nx{nx}-nz{nz}-mu{mu:.1f}-nu{nu:.1f}.mat')
        ax = axes[i, j]
        ax.semilogy(data['xfwm'][0,:], label=r'FSM$(\mu/\nu^2)$')
        ax.semilogy(data['xdys1.0'][0,:], label=r'DYS$(\mu/\gamma)$')
        ax.semilogy(data['xdys0.7'][0,:], label=r'DYS$(0.7\mu/\gamma)$')
        ax.semilogy(data['xdys0.5'][0,:], label=r'DYS$(0.5\mu/\gamma)$')
        ax.semilogy(data['xdys0.3'][0,:], label=r'DYS$(0.3\mu/\gamma)$')
        # ax.set_xlim(0, 2500)
        # ax.set_ylim(1e-4, 5e2)
        ax.grid()
        if i == 0:
            ax.set_title(f'input_dim={nx}')
        if i == 1:
            ax.set_xlabel('Steps')
        if j == 0:
            if i == 0:
                ax.set_ylabel(r'$\|x-x^\star\|/\|x\|$ (hid_dim=384)')
            else:
                ax.set_ylabel(r'$\|x-x^\star\|/\|x\|$ (hid_dim=768)')
        if i == 0 and j == 0:
            ax.legend(loc=0)

plt.subplots_adjust(wspace=0, hspace=0)
fig.tight_layout()
fig.savefig(f'{root_dir}/solver.pdf')