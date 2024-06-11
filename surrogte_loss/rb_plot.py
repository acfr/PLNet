import numpy as np 
import matplotlib.pyplot as plt
import scipy.io 


f_lvl, e_lvl = 35, 10
lw = 1.5

name = "MLP"
depth = 3 
mlp_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}/data.mat')['dat']
mlp_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}/data.mat')['dat']

name = "ICNN"
depth = 8 
icnn_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}/data.mat')['dat']
icnn_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}/data.mat')['dat']
emax = np.max(np.abs(icnn_rr['zh'][0,0]-icnn_rr['zt'][0,0]))

mu, nu = 0.04, 16.0

name = "i-ResNet"
depth = 5 
resnet_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']
resnet_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']

name = "i-DenseNet"
depth = 5 
densenet_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']
densenet_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']

name = "BiLipNet"
depth = 2 
bln16_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']
bln16_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']

mu, nu = 0.25, 4.0
name = "BiLipNet"
depth = 2 
bln4_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']
bln4_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']

mu, nu = 0.5, 2.0
name = "BiLipNet"
depth = 2 
bln2_rb = scipy.io.loadmat(f'./results/rosenbrock/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']
bln2_rr = scipy.io.loadmat(f'./results/rosenbrock-sine/{name}-{depth}-mu{mu:.2f}-nu{nu:.1f}/data.mat')['dat']

x = mlp_rb['x'][0, 0]
y = mlp_rb['y'][0, 0]
zt_rb = mlp_rb['zt'][0, 0]
zmax = np.max(zt_rb)
zt_rr = mlp_rr['zt'][0, 0]
fz = 18

plt.rcParams['font.size'] = 14
plt.rcParams['text.usetex'] = True
fig, axes = plt.subplots(7, 4, figsize=(12.4, 17))
# MLP ------------------------

ax = axes[0, 0]
cs = ax.contourf(x, y, zt_rb, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('True',fontsize=fz)
ax.set_title('Rosenbrock',fontsize=fz)

axes[0, 1].axis('off')

ax = axes[0, 2]
cs = ax.contourf(x, y, zt_rr, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Rosenbrock + Sine',fontsize=fz)

axes[0, 3].axis('off')

row=1
zh = mlp_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('MLP',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Error')
zh = mlp_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Error')

# ICNN ------------------------
row=2
zh = icnn_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('ICNN',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
zh = icnn_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])

# i-DenseNet ------------------------
row = 3
zh = densenet_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$(0.04, 16)$-i-DenseNet',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
zh = densenet_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])

# BLN16 ------------------------
row = 5
zh = bln16_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$(0.04, 16)$-BiLipNet',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
zh = bln16_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig(f'./results/rosenbrock-sine-v1.pdf')

plt.rcParams['font.size'] = 14
plt.rcParams['text.usetex'] = True
fig, axes = plt.subplots(4, 4, figsize=(11, 10))
fz = 18
# MLP ------------------------
x = mlp_rb['x'][0, 0]
y = mlp_rb['y'][0, 0]
zt_rb = mlp_rb['zt'][0, 0]
zmax = np.max(zt_rb)
zt_rr = mlp_rr['zt'][0, 0]

ax = axes[0, 0]
cs = ax.contourf(x, y, zt_rb, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('True',fontsize=fz)
ax.set_title('Rosenbrock',fontsize=fz)

axes[0, 1].axis('off')

ax = axes[0, 2]
cs = ax.contourf(x, y, zt_rr, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Rosenbrock + Sine',fontsize=fz)

axes[0, 3].axis('off')


# i-ResNet ------------------------
row=1
zh = resnet_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$(0.04, 16)$-i-ResNet',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
zh = resnet_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])


# BLN4 ------------------------
row = 2
zh = bln4_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$(0.25, 4)$-BiLipNet',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
zh = bln4_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])

# BLN4 ------------------------
row = 3
zh = bln2_rb['zh'][0, 0]
ax = axes[row, 0]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel(r'$(0.5, 2)$-BiLipNet',fontsize=fz)
ax = axes[row, 1]
cs = ax.contourf(x, y, np.abs(zh-zt_rb), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
zh = bln2_rr['zh'][0, 0]
ax = axes[row, 2]
cs = ax.contourf(x, y, zh, levels=f_lvl, cmap="RdYlGn_r", vmax=zmax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax = axes[row, 3]
cs = ax.contourf(x, y, np.abs(zh-zt_rr), levels=e_lvl, vmax=emax)
cbar = fig.colorbar(cs, ax=ax)
ax.set_xticks([])
ax.set_yticks([])

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig(f'./results/rosenbrock-sine-v1.pdf')
plt.close()
