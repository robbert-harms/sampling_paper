from mot.mcmc_diagnostics import minimum_multivariate_ess
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn

def set_matplotlib_font_size(font_size):
    import matplotlib.pyplot as plt
    plt.rc('font', size=font_size)  # controls default text sizes
    plt.rc('axes', titlesize=font_size)  # fontsize of the axes title
    plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=font_size)  # legend fontsize
    plt.rc('figure', titlesize=font_size)


set_matplotlib_font_size(28)

colors = ['r--', 'b:', 'g', 'k', 'y-.']

f, axarr = plt.subplots(2, 2)
f.subplots_adjust(hspace=0.35, wspace=0.25)

f.suptitle(r'Theoretical minimum amount of ESS ($W(p, \alpha, \epsilon)$) for varying confidence levels $\alpha$, relative precision $\epsilon$ and parameters $p$')
for x, y in itertools.product(range(2), range(2)):
    axarr[x, y].set_ylabel(r'$W(p, \alpha, \epsilon)$')

parameters = [4, 8, 10, 16]

x = np.arange(0.0001, 0.15, 0.0001)
for color_ind, nmr_params in enumerate(parameters):
    axarr[0, 0].plot(x, [minimum_multivariate_ess(nmr_params, xi, 0.10) for xi in x], colors[color_ind], label=r'$p={}$'.format(nmr_params))
axarr[0, 0].set_title(r'$\epsilon=0.1$')
axarr[0, 0].set_xlabel(r'$\alpha$')
axarr[0, 0].legend(loc='upper right')


x = np.arange(0.01, 0.2, 0.0001)
for color_ind, nmr_params in enumerate(parameters):
    axarr[0, 1].plot(x, [minimum_multivariate_ess(nmr_params, 0.05, xi) for xi in x], colors[color_ind], label=r'$p={}$'.format(nmr_params))
axarr[0, 1].set_title(r'$\alpha=0.05$')
axarr[0, 1].set_xlabel(r'$\epsilon$')
axarr[0, 1].legend(loc='upper right')

x = np.arange(0.0001, 0.15, 0.0001)
for color_ind, epsilon in enumerate([0.05, 0.1, 0.15, 0.2]):
    axarr[1, 0].plot(x, [minimum_multivariate_ess(10, xi, epsilon) for xi in x], colors[color_ind], label=r'$\epsilon={}$'.format(epsilon))
axarr[1, 0].set_title(r'$p=10$')
axarr[1, 0].set_xlabel(r'$\alpha$')
axarr[1, 0].legend(loc='upper right')

x = np.arange(0.05, 0.2, 0.0001)
for color_ind, alpha in enumerate([0.01, 0.05, 0.1, 0.15]):
    axarr[1, 1].plot(x, [minimum_multivariate_ess(10, alpha, xi) for xi in x], colors[color_ind], label=r'$\alpha={}$'.format(alpha))
axarr[1, 1].set_title(r'$p=10$')
axarr[1, 1].set_xlabel(r'$\epsilon$')
axarr[1, 1].legend(loc='upper right')

plt.show()
