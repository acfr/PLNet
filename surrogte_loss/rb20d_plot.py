import numpy as np 
import matplotlib.pyplot as plt
import scipy.io 

root_dir = './results/'
batch = 50
tau = [2, 4, 5, 8, 10, 20, 50, 70, 100]
M = len(tau)
vloss = np.zeros((M))
tloss = np.zeros((M))

for j, t in enumerate(tau):
    dat = scipy.io.loadmat(f'{root_dir}/rosenbrock-dim20-batch50/BiLipNet-2-tau{t:.0f}/data.mat')
    vloss[j] = dat['eval_loss'][0,-1]
    tloss[j] = dat['train_loss'][0,-1]

tau = np.array(tau)

name = ["i-ResNet", "i-DenseNet"]
tau1 = [2, 5,10, 40, 60, 80, 100]
vloss1 = np.zeros((2, 7))
tloss1 = np.zeros((2, 7))
for i, s in enumerate(name):
    for j, t in enumerate(tau1):
        dat = scipy.io.loadmat(f'{root_dir}/rosenbrock-dim20-batch50/{s}-5-tau{t:.0f}/data.mat')
        vloss1[i, j] = dat['eval_loss'][0,-1]
        tloss1[i, j] = dat['train_loss'][0,-1]

plt.rcParams['font.size'] = 12
plt.rcParams['text.usetex'] = True
fig, axes = plt.subplots(1, 2, figsize=(6, 3))

ax = axes[0]
ax.loglog(tau, tloss, 'o-', label='BiLipNet')
for i, s in enumerate(name):
    if i == 0:
        ax.loglog(tau1, tloss1[i, :], 'o-', label=f'{s}')
    else:
        ax.loglog(tau1, tloss1[i, :], 'x--', label=f'{s}')
ax.legend(loc=0, handlelength=1)
ax.set_xlabel(r'model distortion $\tau$')
ax.set_ylabel(r'$\ell_2$ loss')
ax.set_title('Train')

ax = axes[1]
ax.loglog(tau, vloss, 'o-', label='BiLipNet')
for i, s in enumerate(name):
    if i == 0:
        ax.loglog(tau1, vloss1[i, :], 'o-', label=f'{s} ')
    else:
        ax.loglog(tau1, vloss1[i, :], 'x--', label=f'{s} ')
ax.legend(loc=0, handlelength=1)
ax.set_xlabel(r'model distortion $\tau$')
ax.set_title('Test')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.savefig(f'./results/rosenbrock20-model.pdf')