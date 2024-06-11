import numpy as np 
import matplotlib.pyplot as plt
import scipy.io 

mu = 0.1
nu = 10.0
root_dir = "./results/toy1d"

plt.rcParams['font.family'] = 'monospace'
# plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True

fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.7))

loss_resnet, loss_dense = 1e2, 1e2
for depth in [3, 5, 7, 9]:
    data = scipy.io.loadmat(f"{root_dir}/i-Resnet-dp{depth}-mu{mu:.1f}-nu{nu:.1f}/data.mat")
    if data['val_loss'][0,-1] <= loss_resnet:
        loss_resnet = data['val_loss'][0,-1]
        resnet = data 
    data = scipy.io.loadmat(f"{root_dir}/i-Densenet-dp{depth}-mu{mu:.1f}-nu{nu:.1f}/data.mat")
    if data['val_loss'][0,-1] <= loss_dense:
        loss_dense = data['val_loss'][0,-1]
        dense = data 

xtrue = np.array([-2., 0., 0., 2.])
ytrue = np.array([-2., -2., 2., 2.])
ax.plot(xtrue, ytrue, '--', color='black', label='Target fn.')

ax.plot(resnet['xtest'], resnet['yh'], label='i-ResNet')
print(f"iResnet: Lip. {resnet['lipmin'][0,-1]:.2f}, {resnet['lipmax'][0,-1]:.2f} Loss: {loss_resnet:.4f}")

ax.plot(dense['xtest'], dense['yh'], '-.', linewidth=2, label='i-Densenet')
print(f"iResnet: Lip. {dense['lipmin'][0,-1]:.2f}, {dense['lipmax'][0,-1]:.2f} Loss: {loss_dense:.4f}")

data = scipy.io.loadmat(f'{root_dir}/MonLipNet-dp1-mu{mu:.1f}-nu{nu:.1f}/data.mat')
xa = data['xa'][0, 0]
ya = nu * xa 
ax.plot(data['xtest'], data['yh'], color='blue', label='BiLipNet')
xopt = np.array([-2., -xa, xa, 2.])
yopt = np.array([-ya+mu*(-2+xa), -ya, ya, ya+mu*(2-xa)])
ax.plot(xopt, yopt, '-.', linewidth=2, color='red', label='Optimal fit')
print(f"MonLipNet: Lip. {data['lipmin'][0,-1]:.2f}, {data['lipmax'][0,-1]:.2f} Loss: {data['val_loss'][0,-1]:.4f}")
print(f"Optimal: Loss: {data['lopt'][0,0]:.4f}")
ax.set_xlim(-2., 2.)
ax.set_ylim(-3, 3)
# ax.grid()
ax.legend(loc=4, handlelength=2)
fig.tight_layout()
fig.savefig(f'{root_dir}/step-fit.pdf')