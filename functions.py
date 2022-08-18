import matplotlib.pyplot as plt
import numpy as np
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colorbar

def plot_2D(ax, xx, yy, zz, xlabel='', ylabel='', title='',
            xlim=None, ylim=None, vmin=None, vmax=None,
            savepath=None, grid=False, cmap="RdBu_r", show_colorbar=False):
    # fig, ax = plt.subplots(1, figsize=(9, 6), constrained_layout=True)
    ax.set_title(title)

    zmax = np.max(zz) if vmax is None else vmax
    zmin = np.min(zz) if vmin is None else vmin

    step_X = xx[0, 1] - xx[0, 0]
    step_Y = yy[1, 0] - yy[0, 0]
    extent = [xx[0, 0] - 1 / 2 * step_X, xx[0, -1] + 1 / 2 * step_X,
              yy[0, 0] - 1 / 2 * step_Y, yy[-1, 0] + 1 / 2 * step_Y]
    ax_map = ax.imshow(zz, origin='lower', cmap=cmap,
                       aspect='auto', vmax=zmax, vmin=zmin, extent=extent)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(grid)
    if show_colorbar:
        cax, kw = colorbar.make_axes(ax)
        plt.colorbar(ax_map, cax=cax)
    return ax_map

def plot_sigmas(trace1, trace2, sm1re, sm1im, sm2re, sm2im):
    fig, axs = plt.subplots(3, 1, figsize=(9, 5), tight_layout=True)
    axs[0].plot(trace1, label="qubit 1")
    axs[0].plot(trace2, label="qubit 2")
    axs[0].legend(loc=1)
    axs[0].set_title("Ground state population")
    axs[1].plot(sm1re, label="Re")
    axs[1].plot(sm1im, label="Im")
    axs[1].set_title("Q1 Sigma minus")
    axs[1].legend(loc=1)
    ax1_d = make_axes_locatable(axs[1])
    ax1_fft = ax1_d.append_axes("right", size="50%", pad=0.2)
    ax1_fft.plot(np.log(np.abs(np.fft.fftshift(np.fft.fft(sm1re + 1j * sm1im)))))
    ax1_fft.set_xlim(len(sm1re)/2-20, len(sm1re)/2 + 20)
    ax1_fft.set_yticks([])
    axs[2].plot(sm2re, label="Re")
    axs[2].plot(sm2im, label="Im")
    axs[2].set_title("Q2 Sigma minus")
    axs[2].legend(loc=1)
    ax2_d = make_axes_locatable(axs[2])
    ax2_fft = ax2_d.append_axes("right", size="50%", pad=0.2)
    ax2_fft.plot(np.log(np.abs(np.fft.fftshift(np.fft.fft(sm2re + 1j * sm2im)))))
    ax2_fft.set_xlim(len(sm2re)/2-20, len(sm2re)/2 + 20)
    ax2_fft.set_yticks([])
    plt.show()

def plot_maps(ffts):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    x = np.linspace(-2, 2, 100, endpoint=False)
    xx, yy = np.meshgrid(x, x)
    plot_2D(ax[0, 0], xx, yy, 10 * np.log10(np.abs(ffts[0, :, :])), cmap="inferno", show_colorbar=True)
    plot_2D(ax[0, 1], xx, yy, 10 * np.log10(np.abs(ffts[1, :, :])), cmap="inferno", show_colorbar=True)
    plot_2D(ax[1, 0], xx, yy, 10 * np.log10(np.abs(ffts[2, :, :])), cmap="inferno", show_colorbar=True)
    plot_2D(ax[1, 1], xx, yy, 10 * np.log10(np.abs(ffts[3, :, :])), cmap="inferno", show_colorbar=True)
    fig.show()