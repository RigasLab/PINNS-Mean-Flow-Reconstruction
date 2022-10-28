"""This script plots true (DNS, time-averaged) u,v,p,uu,uv,vv fields.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import deepcopy
import scipy.io
import sys
import math
import numpy as np


import os


import matplotlib.pyplot as plt


fs = 20


def set_font(fs=20):
    plt.rc('font', size=fs)  # controls default text size
    plt.rc('axes', titlesize=fs)  # fontsize of the title
    plt.rc('axes', labelsize=fs)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fs)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=fs)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=fs)  # fontsize of the legend


set_font()


def read_data():
    """Read data from .mat files.
    """
    data = scipy.io.loadmat("./Data/unsteadyCylinder_full_field.mat")
    data_no_airfoil = scipy.io.loadmat(
        "./Data/unsteadyCylinder_no_cylinder.mat")
    x = data["x_data"].T
    y = data["y_data"].T
    x_no_airfoil = data_no_airfoil["x_data"].T
    y_no_airfoil = data_no_airfoil["y_data"].T
    u = data["u_data"].T
    v = data["v_data"].T
    p = data["p_data"].T
    uu = data["uu_data"].T
    uv = data["uv_data"].T
    vv = data["vv_data"].T
    return x, y, u, v, p, uu, uv, vv, x_no_airfoil, y_no_airfoil


if __name__ == "__main__":

    # Read data
    x_data, y_data, u_data, v_data, p_data, uu_data, uv_data, vv_data, x_domain, y_domain = read_data()

    # Find indices with points inside cylinder
    zero_index = (x_data < 0) & (x_data > 0)
    zero_index = zero_index | ((u_data == 0) & (v_data == 0))
    no_data_index = zero_index

    # Define cylinder surface points for plotting
    theta = np.linspace(0, 2*math.pi, 100)[:, None]
    R = 0.5
    cylinder_array = np.hstack((R*np.cos(theta), R*np.sin(theta)))

    # Domain vertices
    v_ld = [-1, -1.5]
    v_ru = [3, 1.5]

    # Point-width of the domain
    Nx = int((v_ru[0]-v_ld[0])*500)+1
    Ny = int((v_ru[1]-v_ld[1])*500)+1
    print('Nx', Nx, 'Ny', Ny)

    # Defining points withing cylinder for cylinder plotting
    zero_reshaped = deepcopy(zero_index)
    zero_reshaped = zero_reshaped.reshape(Nx, Ny).T

    x_reshaped = deepcopy(x_data)
    x_reshaped = x_reshaped.reshape(Nx, Ny).T

    y_reshaped = deepcopy(y_data)
    y_reshaped = y_reshaped.reshape(Nx, Ny).T

    x_ar = []
    y_ar = []
    offset = 8
    for i in range(offset, Nx-offset):
        for j in range(offset, Ny-offset):
            if zero_reshaped[j, i-offset] and zero_reshaped[j, i+offset] and zero_reshaped[j-offset, i] and zero_reshaped[j+offset, i]:
                x_ar.append(x_reshaped[j, i])
                y_ar.append(y_reshaped[j, i])

    # Using scatter on (x_ar, y_ar) points draws the inside of the cylinder.

    # Define figsize
    figsize = (8, 10)

    # Reading data
    predictions = scipy.io.loadmat("./Data/unsteadyCylinder_full_field.mat")

    # Plotting

    # Setting x and y arrays for plotting
    x_plot = np.linspace(v_ld[0], v_ru[0], Nx)
    y_plot = np.linspace(v_ld[1], v_ru[1], Ny)

    X, Y = np.meshgrid(x_plot, y_plot)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    set_font(12)

    # Setting stride for colorplots
    stride = 5

    # Plot true fields
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True,
                            sharey=True, figsize=(10, 7))

    to_plot = deepcopy(predictions['u_data']).T
    to_plot[no_data_index] = to_plot[no_data_index]*0
    to_plot = to_plot.T.reshape(Nx, Ny).T
    fig.subplots_adjust(left=0.05, bottom=0.06,
                        right=0.95, top=0.94, wspace=0.08, hspace=0.1)
    im0 = axs[0, 0].pcolor(X[::stride, ::stride], Y[::stride,
                                                    ::stride], to_plot[::stride, ::stride])

    fig.colorbar(im0, ax=axs[0, 0],
                 orientation='horizontal', shrink=0.9, pad=0.03)
    axs[0, 0].scatter(x_ar, y_ar, s=0.02, c='white')
    axs[0, 0].plot(cylinder_array[:, 0],
                   cylinder_array[:, 1], lw=1.5, c='black')

    to_plot = deepcopy(predictions['v_data']).T
    to_plot[no_data_index] = to_plot[no_data_index]*0
    to_plot = to_plot.T.reshape(Nx, Ny).T
    im1 = axs[0, 1].pcolor(X[::stride, ::stride], Y[::stride,
                                                    ::stride], to_plot[::stride, ::stride])

    fig.colorbar(im1, ax=axs[0, 1],
                 orientation='horizontal', shrink=0.9, pad=0.03)
    axs[0, 1].scatter(x_ar, y_ar, s=0.05, c='white')
    axs[0, 1].plot(cylinder_array[:, 0],
                   cylinder_array[:, 1], lw=1.5, c='black')

    to_plot = deepcopy(predictions['p_data']).T
    to_plot[no_data_index] = to_plot[no_data_index]*0
    to_plot = to_plot.T.reshape(Nx, Ny).T
    im2 = axs[0, 2].pcolor(X[::stride, ::stride], Y[::stride,
                                                    ::stride], to_plot[::stride, ::stride])

    fig.colorbar(im2, ax=axs[0, 2],
                 orientation='horizontal', shrink=0.9, pad=0.03)
    axs[0, 2].scatter(x_ar, y_ar, s=0.05, c='white')
    axs[0, 2].plot(cylinder_array[:, 0],
                   cylinder_array[:, 1], lw=1.5, c='black')

    to_plot = deepcopy(predictions['uu_data']).T
    to_plot[no_data_index] = to_plot[no_data_index]*0
    to_plot = to_plot.T.reshape(Nx, Ny).T

    im20 = axs[1, 0].pcolor(X[::stride, ::stride], Y[::stride,
                                                     ::stride], to_plot[::stride, ::stride])

    fig.colorbar(im20, ax=axs[1, 0],
                 orientation='horizontal', shrink=0.9, pad=0.1)
    axs[1, 0].scatter(x_ar, y_ar, s=0.05, c='white')
    axs[1, 0].plot(cylinder_array[:, 0],
                   cylinder_array[:, 1], lw=1.5, c='black')

    to_plot = deepcopy(predictions['uv_data']).T
    to_plot[no_data_index] = to_plot[no_data_index]*0
    to_plot = to_plot.T.reshape(Nx, Ny).T
    im21 = axs[1, 1].pcolor(X[::stride, ::stride], Y[::stride,
                                                     ::stride], to_plot[::stride, ::stride])

    fig.colorbar(im21, ax=axs[1, 1],
                 orientation='horizontal', shrink=0.9, pad=0.1)
    axs[1, 1].scatter(x_ar, y_ar, s=0.05, c='white')
    axs[1, 1].plot(cylinder_array[:, 0],
                   cylinder_array[:, 1], lw=1.5, c='black')

    to_plot = deepcopy(predictions['vv_data']).T
    to_plot[no_data_index] = to_plot[no_data_index]*0
    to_plot = to_plot.T.reshape(Nx, Ny).T
    im22 = axs[1, 2].pcolor(X[::stride, ::stride], Y[::stride,
                                                     ::stride], to_plot[::stride, ::stride])

    fig.colorbar(im22, ax=axs[1, 2],
                 orientation='horizontal', shrink=0.9, pad=0.1)
    axs[1, 2].scatter(x_ar, y_ar, s=0.05, c='white')
    axs[1, 2].plot(cylinder_array[:, 0],
                   cylinder_array[:, 1], lw=1.5, c='black')

    axs[0, 0].set_title(r"$\overline{u}_{true}$")
    axs[0, 1].set_title(r"$\overline{v}_{true}$")
    axs[0, 2].set_title(r"$\overline{p}_{true}$")
    axs[1, 0].set_title(r"$\overline{u'u'}_{true}$")
    axs[1, 1].set_title(r"$\overline{u'v'}_{true}$")
    axs[1, 2].set_title(r"$\overline{v'v'}_{true}$")

    plt.savefig('plots/true.png', dpi=400)
    plt.close()
