from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import deepxde as dde
from copy import deepcopy
import scipy.io
import sys
import math
import numpy as np


import os
import matplotlib.pyplot as plt

from utilities import set_directory, plot_train_points
from equations import RANSReStresses2D, func_zeros


fs = 20
plt.rc('font', size=fs)  # controls default text size
plt.rc('axes', titlesize=fs)  # fontsize of the title
plt.rc('axes', labelsize=fs)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=fs)  # fontsize of the x tick labels
plt.rc('ytick', labelsize=fs)  # fontsize of the y tick labels
plt.rc('legend', fontsize=fs)  # fontsize of the legend


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


def generate_domain_points(x, y, geometry):
    """Generate collocation points within the domain randomly.
    """
    points = []
    centre_x = 0
    centre_y = 0
    r = np.sqrt((x-centre_x)**2 + ((y-centre_y)*7)**2)
    r = r/(np.max(r)*1)
    r = r**0.3
    r = 1-r
    for i in range(x.shape[0]):
        tmp_u = np.random.random()
        tmp_r = np.random.random()
        if (tmp_r < r[i, 0]) and (tmp_u < 0.05) and geometry.inside([x[i, 0], y[i, 0]]):
            points.append([x[i, 0], y[i, 0]])
    print(f'Generated {len(points)} points in the domain')
    return points


def generate_PIV_points(x, y, outputs, x_stride, y_stride, v_ld, v_ru, geometry):
    """ Generate PIV points for training.
    """
    x_p = deepcopy(x)
    y_p = deepcopy(y)
    outputs_p = deepcopy(outputs)
    x_p = x_p.reshape(2001, 1501).T
    y_p = y_p.reshape(2001, 1501).T
    for i in range(len(outputs_p)):
        outputs_p[i] = outputs_p[i].reshape(2001, 1501).T

    start_ind_x = int((x_p.shape[1] % x_stride)/2)
    start_ind_y = int((x_p.shape[0] % y_stride)/2)
    start_ind_y = 0

    x_p = x_p[start_ind_y::y_stride, start_ind_x::x_stride]
    y_p = y_p[start_ind_y::y_stride, start_ind_x::x_stride]
    for i in range(len(outputs_p)):
        outputs_p[i] = outputs_p[i][start_ind_y::y_stride,
                                    start_ind_x::x_stride]

    x_p = x_p.T.reshape(-1, 1)
    y_p = y_p.T.reshape(-1, 1)
    for i in range(len(outputs_p)):
        outputs_p[i] = outputs_p[i].T.reshape(-1, 1)

    X = []
    for i in range(x_p.shape[0]):
        if geometry.inside([x_p[i, 0], y_p[i, 0]]) \
                and x_p[i, 0] >= v_ld[0] and x_p[i, 0] <= v_ru[0] \
                and y_p[i, 0] >= v_ld[1] and y_p[i, 0] <= v_ru[1]:
            items = [x_p[i, 0], y_p[i, 0]]
            for j in range(len(outputs_p)):
                items.append(outputs_p[j][i, 0])
            X.append(items)
    X = np.array(X)
    return np.hsplit(X, 2+len(outputs_p))


def main(train=True, test=True):
    # Case name
    case_name = "unCylinder_2nd_order_superresolutions_with_pressure_anchor_0.05"
    case_name_title = r'PIV superresolution second order 0.05 by 0.05'

    set_directory(case_name)

    x_data, y_data, u_data, v_data, p_data, uu_data, uv_data, vv_data, x_domain, y_domain = read_data()

    # Domain vertices
    v_ld = [-1, -1.5]
    v_ru = [3, 1.5]
    figsize = (10, 10*(v_ru[1]-v_ld[1])/(v_ru[0]-v_ld[0]))
    figsize = (8, 5)

    Nx = int((v_ru[0]-v_ld[0])*500)+1
    Ny = int((v_ru[1]-v_ld[1])*500)+1
    print('Nx', Nx, 'Ny', Ny)

    # Geometry specification
    geom1 = dde.geometry.Disk(0, 0.5)
    geom2 = dde.geometry.Rectangle(v_ld, v_ru)
    geom = geom2 - geom1

    # PIV point definition
    [x_piv, y_piv, u_piv, uu_piv, v_piv, vv_piv, uv_piv] = \
        generate_PIV_points(x_data, y_data, [u_data, uu_data, v_data, vv_data, uv_data],
                            25, 25, v_ld, v_ru, geom)
    piv_points = np.hstack((x_piv, y_piv))

    # Pressure point definition
    for i in range(x_data.shape[0]):
        if x_data[i, 0] == 0 and y_data[i, 0] == 0.5:
            p1 = p_data[i, 0]
            print(p1)
        elif x_data[i, 0] == 0 and y_data[i, 0] == -0.5:
            p2 = p_data[i, 0]
            print(p2)

    pressure_coors = np.array([[0, 0.5], [0, -0.5]])
    pressure_vals = np.array([[p1], [p2]])

    # BC specification

    # Boundary indicator functions
    def boundary(x, on_boundary):
        return on_boundary and not (
            np.isclose(x[0], v_ld[0])
            or np.isclose(x[0], v_ru[0])
            or np.isclose(x[1], v_ld[1])
            or np.isclose(x[1], v_ru[1])
        )

    # BC objects
    bc_wall_u = dde.DirichletBC(geom, func_zeros, boundary, component=0)
    bc_wall_v = dde.DirichletBC(geom, func_zeros, boundary, component=1)
    bc_wall_uu = dde.DirichletBC(geom, func_zeros, boundary, component=3)
    bc_wall_uv = dde.DirichletBC(geom, func_zeros, boundary, component=4)
    bc_wall_vv = dde.DirichletBC(geom, func_zeros, boundary, component=5)

    u_piv_points = dde.PointSetBC(piv_points, u_piv, component=0)
    uu_piv_points = dde.PointSetBC(piv_points, uu_piv, component=3)
    v_piv_points = dde.PointSetBC(piv_points, v_piv, component=1)
    vv_piv_points = dde.PointSetBC(piv_points, vv_piv, component=5)
    uv_piv_points = dde.PointSetBC(piv_points, uv_piv, component=4)

    pressure_points = dde.PointSetBC(
        pressure_coors, pressure_vals, component=2)

    # Custom domain points
    domain_points = generate_domain_points(x_domain, y_domain, geometry=geom)

    # Pde and physics compilation
    pde = RANSReStresses2D(150)
    if train:
        data = dde.data.PDE(
            geom,
            pde,
            [bc_wall_u, bc_wall_v, bc_wall_uu, bc_wall_uv, bc_wall_vv, u_piv_points,
                uu_piv_points, v_piv_points, vv_piv_points, uv_piv_points, pressure_points],
            100,
            1600,
            solution=None,
            num_test=100,
            train_distribution="custom",
            custom_train_points=domain_points,
        )
        plot_train_points(data, [2, 5, 10, 11], ["u,v wall BC", "uu,uv,vv wall BC", "PIV data", "surface pressure"],
                          case_name, title=case_name_title, figsize=(10, 5))
    else:
        data = dde.data.PDE(
            geom,
            pde,
            [bc_wall_u, bc_wall_v, bc_wall_uu, bc_wall_uv, bc_wall_vv, u_piv_points,
                uu_piv_points, v_piv_points, vv_piv_points, uv_piv_points, pressure_points],
            100,
            100,
            solution=None,
            num_test=100
        )

    # NN model definition
    layer_size = [2] + [100] * 7 + [6]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    # PINN definition
    model = dde.Model(data, net)

    if train:
        # Adam optimization
        loss_weights = [1, 1, 1, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        model.compile("adam", lr=0.001, loss_weights=loss_weights)
        checkpointer = dde.callbacks.ModelCheckpoint(
            f"{case_name}/models/model_{case_name}.ckpt",
            verbose=1,
            save_better_only=True,
        )

        loss_update = dde.callbacks.LossUpdateCheckpoint(
            momentum=0.7,
            verbose=1, period=1, report_period=100,
            base_range=[0, 1, 2, 3],
            update_range=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        )
        print('Training for 20000 epochs')
        losshistory, train_state = model.train(
            epochs=20000, callbacks=[checkpointer, loss_update], display_every=100
        )

        model.save(f"{case_name}/models/model-adam-last")

        # L-BFGS-B optimization
        model.compile("L-BFGS-B", loss_weights=loss_weights)
        losshistory, train_state = model.train()
        model.save(f"{case_name}/models/model-bfgs-last")

    if test:
        model.compile("adam", lr=0.001)
        model.compile("L-BFGS-B")
        last_epoch = model.train_state.epoch
        if not train:
            last_epoch = 80001
        model.restore(f"{case_name}/models/model-bfgs-last-{last_epoch}")

        x_plot = np.linspace(v_ld[0], v_ru[0], Nx)
        y_plot = np.linspace(v_ld[1], v_ru[1], Ny)

        # Domain data
        x_data = x_data.reshape(2001, 1501).T
        y_data = y_data.reshape(2001, 1501).T
        u_data = u_data.reshape(2001, 1501).T
        v_data = v_data.reshape(2001, 1501).T
        p_data = p_data.reshape(2001, 1501).T
        x_dom = np.linspace(-1, 3, 2001)
        y_dom = np.linspace(-1.5, 1.5, 1501)
        x_min = np.argmin(np.abs(x_dom-v_ld[0]))
        x_max = np.argmin(np.abs(x_dom-v_ru[0]))
        y_min = np.argmin(np.abs(y_dom-v_ld[1]))
        y_max = np.argmin(np.abs(y_dom-v_ru[1]))

        x_data = x_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1, 1)
        y_data = y_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1, 1)
        u_data = u_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1, 1)
        v_data = v_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1, 1)
        p_data = p_data[y_min:y_max+1, x_min:x_max+1].T.reshape(-1, 1)

        # Define points for predicting fields using PINN
        z = np.array([np.array([i, j]) for i in x_plot for j in y_plot])

        # Predict fields
        y = model.predict(z)
        u_star = y[:, 0][:, None]
        v_star = y[:, 1][:, None]
        p_star = y[:, 2][:, None]
        uu_star = y[:, 3][:, None]
        uv_star = y[:, 4][:, None]
        vv_star = y[:, 5][:, None]

        # Save predictions to a .mat file
        data_dict = {
            "x_data": x_data,
            "y_data": y_data,
            "u_star": u_star,
            "v_star": v_star,
            "p_star": p_star,
            "uu_star": uu_star,
            "uv_star": uv_star,
            "vv_star": vv_star,
        }
        scipy.io.savemat(f"{case_name}/results.mat", data_dict)

        # Find indices corresponding to points inside the cylinder
        zero_index = (x_data < 0) & (x_data > 0)
        zero_index = zero_index | ((u_data == 0) & (v_data == 0))
        no_data_index = zero_index

        # Plotting predicted (regressed) fields
        u_star_data = deepcopy(u_star)
        v_star_data = deepcopy(v_star)
        p_star_data = deepcopy(p_star)
        uu_star_data = deepcopy(uu_star)
        uv_star_data = deepcopy(uv_star)
        vv_star_data = deepcopy(vv_star)
        u_star_data[no_data_index] = u_star[no_data_index]*0
        v_star_data[no_data_index] = v_star[no_data_index]*0
        p_star_data[no_data_index] = p_star[no_data_index]*0
        uu_star_data[no_data_index] = uu_star[no_data_index]*0
        uv_star_data[no_data_index] = uv_star[no_data_index]*0
        vv_star_data[no_data_index] = vv_star[no_data_index]*0

        u_star_data = u_star_data.reshape(Nx, Ny).T
        v_star_data = v_star_data.reshape(Nx, Ny).T
        p_star_data = p_star_data.reshape(Nx, Ny).T
        uu_star_data = uu_star_data.reshape(Nx, Ny).T
        uv_star_data = uv_star_data.reshape(Nx, Ny).T
        vv_star_data = vv_star_data.reshape(Nx, Ny).T

        X, Y = np.meshgrid(x_plot, y_plot)
        plt.figure(figsize=figsize)
        # plt.title(f'regressed u field for {case_name_title}')
        plt.pcolor(X, Y, u_star_data)
        plt.colorbar(label='u')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'u_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed v field for {case_name_title}')
        plt.pcolor(X, Y, v_star_data)
        plt.colorbar(label='v')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'v_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed p field for {case_name_title}')
        plt.pcolor(X, Y, p_star_data)
        plt.colorbar(label='p')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'p_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed uu field for {case_name_title}')
        plt.pcolor(X, Y, uu_star_data)
        plt.colorbar(label='uu')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'uu_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed uv field for {case_name_title}')
        plt.pcolor(X, Y, uv_star_data)
        plt.colorbar(label='uv')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'uv_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'regressed vv field for {case_name_title}')
        plt.pcolor(X, Y, vv_star_data)
        plt.colorbar(label='vv')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'vv_plot.png'), dpi=400)
        plt.close()

        # Plotting error fields
        u_star_data = deepcopy(u_star)
        v_star_data = deepcopy(v_star)
        p_star_data = deepcopy(p_star)
        u_star_data[no_data_index] = u_star[no_data_index]*0
        v_star_data[no_data_index] = v_star[no_data_index]*0
        p_star_data[no_data_index] = p_star[no_data_index]*0

        u_star_data = u_star_data.reshape(Nx, Ny).T
        v_star_data = v_star_data.reshape(Nx, Ny).T
        p_star_data = p_star_data.reshape(Nx, Ny).T

        u_true = None
        v_true = None
        p_true = None

        u_true = deepcopy(u_data)
        v_true = deepcopy(v_data)
        p_true = deepcopy(p_data)

        u_true = u_true.reshape(Nx, Ny).T
        v_true = v_true.reshape(Nx, Ny).T
        p_true = p_true.reshape(Nx, Ny).T
        u_err = np.abs(u_true-u_star_data)
        v_err = np.abs(v_true-v_star_data)
        p_err = np.abs(p_true-p_star_data)

        plt.figure(figsize=figsize)
        # plt.title(f'u field abs error for {case_name_title}')
        plt.pcolor(X, Y, u_err)
        plt.colorbar(label='u')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'u_err_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'v field abs error for {case_name_title}')
        plt.pcolor(X, Y, v_err)
        plt.colorbar(label='v')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'v_err_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'p field abs error for {case_name_title}')
        plt.pcolor(X, Y, p_err)
        plt.colorbar(label='p')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'p_err_plot.png'), dpi=400)
        plt.close()

        # Finding pde residuals
        e = model.predict(z, operator=pde)

        e_mass = e[0]
        e_u_momentum = e[1]
        e_v_momentum = e[2]

        data_dict.update({
            "e_mass": e_mass,
            "e_u_momentum": e_u_momentum,
            "e_v_momentum": e_v_momentum,
        })
        scipy.io.savemat(f"{case_name}/results.mat", data_dict)

        # Plotting pde residuals
        e_mass[no_data_index] = e_mass[no_data_index] * 0
        e_u_momentum[no_data_index] = e_u_momentum[no_data_index] * 0
        e_v_momentum[no_data_index] = e_v_momentum[no_data_index] * 0
        e_mass = e_mass.reshape(Nx, Ny).T
        e_u_momentum = e_u_momentum.reshape(Nx, Ny).T
        e_v_momentum = e_v_momentum.reshape(Nx, Ny).T

        plt.figure(figsize=figsize)
        # plt.title(f'mass conservation residual for {case_name_title}')
        plt.pcolor(X, Y, e_mass, vmin=-1, vmax=1)
        plt.colorbar(label='e_mass')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(f'{case_name}',
                                 'plots', 'e_mass_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'u momentum conservation residual for {case_name_title}')
        plt.pcolor(X, Y, e_u_momentum, vmin=-1, vmax=1)
        plt.colorbar(label='e_u_momentum')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(
            f'{case_name}', 'plots', 'e_u_momentum_plot.png'), dpi=400)
        plt.close()
        plt.figure(figsize=figsize)
        # plt.title(f'v momentum conservation residual for {case_name_title}')
        plt.pcolor(X, Y, e_v_momentum, vmin=-1, vmax=1)
        plt.colorbar(label='e_v_momentum')
        plt.xlabel('x/c')
        plt.ylabel('y/c')
        axes = plt.gca()
        axes.set_aspect(1)
        plt.tight_layout()
        plt.savefig(os.path.join(
            f'{case_name}', 'plots', 'e_v_momentum_plot.png'), dpi=400)
        plt.close()


if __name__ == "__main__":
    train = True
    test = True
    if "train" in sys.argv and "test" not in sys.argv:
        train = True
        test = False
    if "train" not in sys.argv and "test" in sys.argv:
        train = False
        test = True
    main(train, test)
