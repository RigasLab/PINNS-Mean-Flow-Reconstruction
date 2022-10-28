import os

import matplotlib.pyplot as plt
import numpy as np


def set_directory(case_name):
    """Creating directories for the PINN case.
    """
    names = [f"{case_name}", f"{case_name}/plots", f"{case_name}/models"]
    for foldername in names:
        try:
            os.mkdir(foldername)
        except FileExistsError:
            pass


def plot_train_points(data, bc_ranges, bc_labels, case_name, title, figsize):
    """Plotting training points.
    """
    plt.figure(figsize=figsize)
    plt.title(f"Training points for {title}")
    plt.scatter(
        data.train_x[np.sum(data.num_bcs):, 0],
        data.train_x[np.sum(data.num_bcs):, 1],
        label="collocation points", s=0.15
    )
    bc_ranges = [0] + bc_ranges
    print('BC ranges', bc_ranges)
    print('Num bcs', data.num_bcs)
    for i in range(1, 3):
        plt.scatter(
            data.train_x[
                int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
                    data.num_bcs[: bc_ranges[i]]
                ),
                0,
            ],
            data.train_x[
                int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
                    data.num_bcs[: bc_ranges[i]]
                ),
                1,
            ],
            label=bc_labels[i - 1], s=1,
        )
    # for i in range(3, len(bc_ranges)):
    #     plt.scatter(
    #         data.train_x[
    #             int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
    #                 data.num_bcs[: bc_ranges[i]]
    #             ),
    #             0,
    #         ],
    #         data.train_x[
    #             int(np.sum(data.num_bcs[: bc_ranges[i - 1]])): np.sum(
    #                 data.num_bcs[: bc_ranges[i]]
    #             ),
    #             1,
    #         ],
    #         label=bc_labels[i - 1], s=1.5
    #     )
    plt.legend(loc=(1.05, 0.45))
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    axes = plt.gca()
    axes.set_aspect(1)
    plt.tight_layout()
    plt.savefig(f"{case_name}/training_points_{case_name}.png", dpi=400)
    plt.close()
