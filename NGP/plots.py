import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
from matplotlib.gridspec import SubplotSpec
import scipy.io 
from sklearn.metrics import roc_auc_score

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

# plt.rcParams['figure.figsize'] = (15,6)
plt.rcParams['font.size'] = 14
# plt.rcParams['text.usetex'] = True

snbs = [0.1, 0.9]
L=len(snbs)
rows = 2 
cols = 2
fig, axs = plt.subplots(rows, cols, figsize=(7, 6))

root_dir = "./results/NGP-lr0.0001-epoch600"
bins = np.linspace(0., 1.0, 51)
names = ["SGNP", "BiLipNet"]
ylabels = ["SNGP", "BiLipNet"]

for i, name in enumerate(names):
    for j, snb in enumerate(snbs):
        data = scipy.io.loadmat(f'{root_dir}/{name}-{snb:.1f}.mat')
        score = np.concatenate([data['train_uncert'], data['ood_uncert']], axis=-1)
        score = np.reshape(score, (2000,))
        label = np.concatenate([np.zeros((1000,)), np.ones((1000,))], axis=0)
        ood_auc = roc_auc_score(label, score)

        ax = axs[i, j]
        ax.set_ylim(DEFAULT_Y_RANGE)
        ax.set_xlim(DEFAULT_X_RANGE)
        pcm = ax.imshow(
            np.reshape(data['test_uncert'], [DEFAULT_N_GRID, DEFAULT_N_GRID]),
            origin="lower",
            extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
            vmin=DEFAULT_NORM.vmin,
            vmax=DEFAULT_NORM.vmax,
            interpolation='bicubic',
            aspect='auto')
        ax.scatter(
            data['train_examples'][:, 0], 
            data['train_examples'][:, 1], 
            c=data['train_labels'], 
            cmap=DEFAULT_CMAP, alpha=0.5)
        ax.scatter(
            data['ood_examples'][:, 0], 
            data['ood_examples'][:, 1], 
            c="red", alpha=0.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f'AUROC {ood_auc*100:.1f}%')
        cbar = fig.colorbar(pcm, ax=ax, ticks=[0.0, 0.5, 1.0])
        # cbar.ax.tick_params(labelsize=16)
        if i == 0 and j == 0:
            ax.set_title("c=0.1")
            ax.set_ylabel("SNGP")
        if i == 0 and j == 1:
            ax.set_title("c=0.9")
        if i == 1 and j == 0:
            ax.set_ylabel("BiLipNet")

plt.rcParams.update({'font.size': 14})
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig(f'{root_dir}/snb.pdf')
plt.close()

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    "Sign sets of subplots with title"
    row = fig.add_subplot(grid)
    # the '\n' is important
    row.set_title(f'{title}\n', fontweight='semibold')
    # hide subplot
    row.set_frame_on(False)
    row.axis('off')

DEFAULT_X_RANGE = (-3.5, 3.5)
DEFAULT_Y_RANGE = (-2.5, 2.5)
DEFAULT_CMAP = colors.ListedColormap(["#377eb8", "#ff7f00"])
DEFAULT_NORM = colors.Normalize(vmin=0, vmax=1,)
DEFAULT_N_GRID = 100

# plt.rcParams['figure.figsize'] = (15,6)
plt.rcParams['font.size'] = 16
# plt.rcParams['text.usetex'] = True

snbs = [0.3, 0.5, 0.7]
L=len(snbs)
rows = 2 
cols = 2*L
# fig, axs = plt.subplots(rows, cols, figsize=(22, 7))
fig, axs = plt.subplots(rows, cols, figsize=(7*L+1, 7))
grid = plt.GridSpec(rows, cols)
for i, c in enumerate(snbs):
    create_subtitle(fig, grid[0, 2*i:2*(i+1)], f'Lip. [{(1-c)**3:.3f}, {(1+c)**3:.3f}] (c={c:.1f})')
fig.set_facecolor('w')

root_dir = "./results/NGP"
bins = np.linspace(0., 1.0, 51)
names = ["SGNP", "BiLipNet"]
ylabels = ["SNGP", "Ours"]

for i, name in enumerate(names):
    for j, snb in enumerate(snbs):
        data = scipy.io.loadmat(f'{root_dir}/{name}-{snb:.1f}.mat')
        score = np.concatenate([data['train_uncert'], data['ood_uncert']], axis=-1)
        score = np.reshape(score, (2000,))
        label = np.concatenate([np.zeros((1000,)), np.ones((1000,))], axis=0)
        ood_auc = roc_auc_score(label, score)
        print(f'{name}-snb{snb}: ood_auc: {ood_auc:.2f}')

        ax = axs[i, 2*j]
        ax.hist(data['ood_uncert'][0,:],bins=bins, label='OOD', color='red')
        ax.hist(data['train_uncert'][0, :], bins=bins, label='Train', color='blue')
        ax.legend(loc=9)
        if j == 0:
            ax.set_ylabel(ylabels[i])
        if i == 1:
            ax.set_xlabel("Predictive uncertainty")

        ax = axs[i, 2*j+1]
        ax.set_ylim(DEFAULT_Y_RANGE)
        ax.set_xlim(DEFAULT_X_RANGE)
        pcm = ax.imshow(
            np.reshape(data['test_uncert'], [DEFAULT_N_GRID, DEFAULT_N_GRID]),
            origin="lower",
            extent=DEFAULT_X_RANGE + DEFAULT_Y_RANGE,
            vmin=DEFAULT_NORM.vmin,
            vmax=DEFAULT_NORM.vmax,
            interpolation='bicubic',
            aspect='auto')
        ax.scatter(
            data['train_examples'][:, 0], 
            data['train_examples'][:, 1], 
            c=data['train_labels'], 
            cmap=DEFAULT_CMAP, alpha=0.5)
        ax.scatter(
            data['ood_examples'][:, 0], 
            data['ood_examples'][:, 1], 
            c="red", alpha=0.1)
        cbar = fig.colorbar(pcm, ax=ax, ticks=[0.0, 0.5, 1.0])
        # cbar.ax.tick_params(labelsize=16)
        if i == 1:
            ax.set_xlabel("Uncertainty surface")

plt.rcParams.update({'font.size': 16})
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.savefig(f'{root_dir}/snb-357.pdf')
plt.close()
