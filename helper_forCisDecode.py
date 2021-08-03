import numpy as np
import matplotlib.pyplot as plt

def heatmap(heatmap, cmap="seismic", interpolation="none", colorbar=False, M=None):
    if M is None:
        M = np.abs(heatmap).max()
        if M == 0:
            M = 1
    plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, interpolation=interpolation)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar:
        plt.colorbar()

        
def heatmap_optional(heatmap, interpolation="none", colorbar=False, r=0.1, alp=1):
    Max = heatmap.max()
    Min = Max * r 
    plt.imshow(heatmap, cmap="Reds", vmax=Max, vmin=Min, interpolation=interpolation, alpha=alp)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar:
        plt.colorbar()
