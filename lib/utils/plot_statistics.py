# Copyright (c) Ingmar Nitze and Konrad Heidler

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import matplotlib.pyplot as plt

def channel_histograms(data):
    data = data.cpu().numpy()
    C = data.shape[1]
    COLS = 4
    ROWS = int(math.ceil(C / COLS))
    fig, ax = plt.subplots(ROWS, COLS, figsize=(3*COLS, 3*ROWS))
    ax = ax.reshape(-1)
    for i in range(C):
        ax[i].hist(data[:, i].reshape(-1))
    plt.show()
