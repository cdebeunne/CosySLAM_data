import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()

if __name__ == '__main__':
    data_campbell = np.load('plot_data_campbell.npz')
    x_campbell = data_campbell['X_det']
    y_campbell = data_campbell['trans_err']

    data_legrand = np.load('plot_data_legrand.npz')
    x_legrand = data_legrand['X_det']
    y_legrand = data_legrand['trans_err']

    data_switch = np.load('plot_data_switch.npz')
    x_switch = data_switch['X_det']
    y_switch = data_switch['trans_err']

    r_list = np.linspace(0.3, 0.8, 100)
    N = 40
    plt.plot(r_list[:N], y_campbell[:N], label='ycbv-4')
    plt.plot(r_list[:N], y_legrand[:N], label='tless-23')
    plt.plot(r_list[:N], y_switch[:N], label='tless-26')
    plt.xlabel(r'$r$ (m)', fontsize = 15)
    plt.ylabel(r'$||e_t|| (m)$', fontsize = 15)
    plt.legend(fontsize = 15)
    plt.show()
