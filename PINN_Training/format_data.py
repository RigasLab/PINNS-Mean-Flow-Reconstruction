"""This script transforms .dat files into .mat files with dictionaries containing fields"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from copy import deepcopy
import os

from sklearn.metrics import max_error

# Forcings
data = np.loadtxt('Data/stresses-1st-grad.dat',
                  skiprows=4, usecols=(5, 7, 8, 10), max_rows=3003501)

print(data.shape)

f_x = -data[:, 0]-data[:, 2]
f_y = -data[:, 1]-data[:, 3]


data_dict = {
    "fx_data": f_x.reshape(-1, 1),
    "fy_data": f_y.reshape(-1, 1)
}
scipy.io.savemat(f"Data/forcings.mat", data_dict)


# Pressure
data = np.loadtxt('Data/pressure.dat',
                  skiprows=3, usecols=(2, 3, 4), max_rows=3003501)

print(data.shape)
print(data[0, :])

p = data[:, 0]
p_x = data[:, 1]
p_y = data[:, 2]


data_dict = {
    "px_data": p_x.reshape(-1, 1),
    "py_data": p_y.reshape(-1, 1)
}
scipy.io.savemat(f"Data/pressure.mat", data_dict)


# Curlf
data = np.loadtxt('Data/stresses-2nd-grad.dat',
                  skiprows=3, usecols=(3, 5, 7, 9), max_rows=3003501)

print(data.shape)
print(data[0, :])

uu_x_y = data[:, 0]
uv_x_x = data[:, 1]
uv_y_y = data[:, 2]
vv_x_y = data[:, 3]

curlf = -uv_x_x-vv_x_y + uv_y_y + uu_x_y


data_dict = {
    "curlf_data": curlf.reshape(-1, 1)
}
scipy.io.savemat(f"Data/curlf.mat", data_dict)


# Velocities
data = np.loadtxt('Data/velocities.dat',
                  skiprows=3, usecols=(0, 1, 2, 3, 4, 5, 6, 7), max_rows=3003501)

print(data.shape)
print(data[0, :])

x = data[:, 0]
y = data[:, 1]
u = data[:, 2]
v = data[:, 3]
u_x = data[:, 4]
u_y = data[:, 5]
v_x = data[:, 6]
v_y = data[:, 7]


data_dict = {
    "u_data": u.reshape(-1, 1),
    "v_data": v.reshape(-1, 1),
    "ux_data": u_x.reshape(-1, 1),
    "uy_data": u_y.reshape(-1, 1),
    "vx_data": v_x.reshape(-1, 1),
    "vy_data": v_y.reshape(-1, 1)
}
scipy.io.savemat(f"Data/velocities.mat", data_dict)

# Re stresses derivatives
data = np.loadtxt('Data/stresses-1st-grad.dat',
                  skiprows=3, usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10), max_rows=3003501)

print(data.shape)
print(data[0, :])

uu = data[:, 0]
uv = data[:, 1]
vv = data[:, 2]
uu_x = data[:, 3]
uu_y = data[:, 4]
uv_x = data[:, 5]
uv_y = data[:, 6]
vv_x = data[:, 7]
vv_y = data[:, 8]


data_dict = {
    "uu_x_data": uu_x.reshape(-1, 1),
    "uu_y_data": uu_y.reshape(-1, 1),
    "uv_x_data": uv_x.reshape(-1, 1),
    "uv_y_data": uv_y.reshape(-1, 1),
    "vv_x_data": vv_x.reshape(-1, 1),
    "vv_y_data": vv_y.reshape(-1, 1)
}
scipy.io.savemat(f"Data/Data/stress_derivatives.mat", data_dict)


# PINN training data
data_all = np.zeros(shape=(data.shape[0], 8))
data_all[:, 0] = x[:]
data_all[:, 1] = y[:]
data_all[:, 2] = u[:]
data_all[:, 3] = v[:]
data_all[:, 4] = p[:]
data_all[:, 5] = uu[:]
data_all[:, 6] = uv[:]
data_all[:, 7] = vv[:]

data_dict = {
    "x_data": data_all[:, 0],
    "y_data": data_all[:, 1],
    "u_data": data_all[:, 2],
    "v_data": data_all[:, 3],
    "p_data": data_all[:, 4],
    "uu_data": data_all[:, 5],
    "uv_data": data_all[:, 6],
    "vv_data": data_all[:, 7],
}

scipy.io.savemat("Data/unsteadyCylinder_full_field.mat", data_dict)


# Without cylinder region (points inside cylinder should not be fed into PINN)
data = []
for i in range(data_all.shape[0]):
    data.append(data_all[i, :])

new_data = [data[0]]
for i in range(1, len(data)-1):
    if data[i][2] != 0 or data[i][3] != 0 \
            or data[i+1][2] != 0 or data[i+1][3] != 0 \
            or data[i-1][2] != 0 or data[i-1][3] != 0:
        new_data.append(data[i])
new_data.append(data[-1])
new_data = np.array(new_data)

data_dict = {
    "x_data": new_data[:, 0],
    "y_data": new_data[:, 1],
    "u_data": new_data[:, 2],
    "v_data": new_data[:, 3],
    "p_data": new_data[:, 4],
    "uu_data": new_data[:, 5],
    "uv_data": new_data[:, 6],
    "vv_data": new_data[:, 7],
}

scipy.io.savemat("Data/unsteadyCylinder_no_cylinder.mat", data_dict)
