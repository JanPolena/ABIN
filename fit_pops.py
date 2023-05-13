# -*- coding: utf-8 -*-

# The script for fitting specific electronic state population extracted with analyze_pops.sh or analyzeSH.sh
# Created by Jan Polena

# ===== USER INPUT =====

# FITTING PROCEDURE
nstates = 5         # Number of states
spin_multi = 1      # 1 - All states are singlets
                    # 2 - The ground state is singlet, the excited states are doublets
                    # 3 - The ground state is singlet, the excited states are triplets
fit_state = 1       # 0 - Ground state, 1 - 1st excited state, 2 - 2nd excited state, ...
fit_type = 1        # 0 - No fit, 1 - JJ fit, 2 - Exponential fit, 3 - Bi-exponential fit, 4 - Bi-exp. JJ Master fit
                        # In specfific cases of Bi-exponential fitting procedures, the bounds for decay components may
                        # be changed in the code in Fitting section

# PATH TO THE DATA
data = "D:\\Documents\Programs\Development\S1_SH_AS66.dat"                  # Data fetch (populations)

# FIGURE OPTIONS
show_details = True     # True - Print fitting details within the figure, False - Fitting details remain hidden
fig_linewidth = 1.8     # Linewidth of plots
fig_fontsize = 18.0     # Fontsize of labels and text
fig_size = (16, 9)      # Size of the figure: (4, 3), (8, 6), (12, 9), (16, 9), ...
x_length = 1000         # Length of horizontal axis in femtoseconds

# PRINTING THE FIGURE
print_fig = 0           # 0 - No printing, 1 - Print the figure to 'output' destination
fig_format = 'png'      # Options: .png, .pdf, .svg, ...
fig_dpi = 'figure'      # Options: figure, 72 (optimal screen DPI), 300 (min. printing high resolution figure DPI), ...
output = 'D:\\Documents\Programs\Development\S1_SH_AS66.%s' % fig_format    # Path to a saved figure

# =============================================================================
# LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import subplot
from scipy.optimize import curve_fit
import sys

# =============================================================================
# ERROR CHECKS
while spin_multi not in (1, 2, 3):
    sys.exit("Please, choose a propriate spin multiplicity in the head of the code:\n"
             "1 - All states are singlets\n"
             "2 - The ground state is singlet, the excited states are doublets\n"
             "3 - The ground state is singlet, the excited states are triplets\n")

while fit_type not in (0, 1, 2, 3, 4):
    sys.exit("Please, choose a propriate type of a fit function in the head of the code:\n"
             "0 - No fit\n"
             "1 - JJ fit\n"
             "2 - Exponential\n"
             "3 - Bi-exponential fit\n"
             "4 - Bi-exponential fit (JJ Master fit)")

if nstates != int(nstates):
    sys.exit("Please, choose a number of states as an integer type value.")

if nstates < 1:
    sys.exit("Please, choose a positive number of states in the head of the code.")

if show_details and fit_type == 0:
    sys.exit("You chose not to fit the data, and yet you would like to show the details of a fitting procedure. Please,"
             "refrain from this behaviour.")

# =============================================================================
# DATA FETCH
pops = np.transpose(np.loadtxt(data))
state = [[] for i in range(nstates)]
nstep = int(pops.size / len(pops) / nstates)    # Number of steps in the simulation

for j in range(nstates):
    state[j] = pops[:, j * nstep: (j + 1) * nstep]

t = state[0][0]

# =============================================================================
# PLOTTING THE DATA
plt.figure(figsize=fig_size)
ax = subplot(111)

if spin_multi == 1:
    for k in range(nstates):
        ax.plot(state[k][0], state[k][1], label="$\mathregular{S_%d}$" % k, linewidth=fig_linewidth, color='C%d' % k)
        ax.fill_between(state[k][0], state[k][1] + state[k][2], state[k][1] - state[k][2], alpha=0.2, color='C%d' % k)
elif spin_multi == 2:
    ax.plot(state[0][0], state[0][1], label="$\mathregular{S_0}$", linewidth=fig_linewidth, color='C0')
    ax.fill_between(state[0][0], state[0][1] + state[0][2], state[0][1] - state[0][2], alpha=0.2, color='C0')
    for k in range(1, nstates):
        ax.plot(state[k][0], state[k][1], label="$\mathregular{D_%d}$" % k, linewidth=fig_linewidth, color='C%d' % k)
        ax.fill_between(state[k][0], state[k][1] + state[k][2], state[k][1] - state[k][2], alpha=0.2, color='C%d' % k)
elif spin_multi == 3:
    ax.plot(state[0][0], state[0][1], label="$\mathregular{S_0}$", linewidth=fig_linewidth, color='C0')
    ax.fill_between(state[0][0], state[0][1] + state[0][2], state[0][1] - state[0][2], alpha=0.2, color='C0')
    for k in range(1, nstates):
        ax.plot(state[k][0], state[k][1], label="$\mathregular{T_%d}$" % k, linewidth=fig_linewidth, color='C%d' % k)
        ax.fill_between(state[k][0], state[k][1] + state[k][2], state[k][1] - state[k][2], alpha=0.2, color='C%d' % k)

# =============================================================================
# FITTING SECTION
if fit_type == 0:
    print("No fit type was set.")

# JJ FIT
elif fit_type == 1:
    def objective(t, t0, tau):
        return np.piecewise(t, [((t >= 0) & (t <= t0)), t > t0], [1, lambda t: np.exp(-(t - t0) / tau)])


    fit, _ = curve_fit(objective, t, state[fit_state][1], bounds=((0, 0), (np.inf, np.inf)))
    t0, tau = fit
    total = t0 + tau
    print('Vibrational relaxation = %d fs\nExcited state lifetime = %d fs\nTotal decay = %d fs' % (t0, tau, total))
    x_line = np.arange(min(t), max(t), 1)
    y_line = objective(x_line, t0, tau)
    if spin_multi == 1 or fit_state == 0:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{S_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 2:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{D_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 3:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{T_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    fit_info = (r'$\bf{Fit}$ $\bf{details:}$''\n\n'
                r'$t \leq t_0:$ $f(x)=1$''\n'
                r'$t > t_0:$ $f(x)=e^{\frac{-(t-t_0)}{\tau}}$''\n\n'
                r'$t_0 = %d$ fs''\n'r'$\tau = %d$ fs''\n'r'$\tau_\mathrm{total} = %d$ fs' % (t0, tau, total))

# EXPONENTIAL FIT
elif fit_type == 2:
    def objective(t, tau):
        return np.exp(-t / tau)


    fit, _ = curve_fit(objective, t, state[fit_state][1], bounds=(0, np.inf))
    tau = fit
    print('Excited state lifetime = %d fs' % tau)
    x_line = np.arange(min(t), max(t), 1)
    y_line = objective(x_line, tau)
    if spin_multi == 1 or fit_state == 0:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{S_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 2:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{D_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 3:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{T_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    fit_info = (r'$\bf{Fit}$ $\bf{details:}$''\n'
                r'$f(x)=e^{\frac{-t}{\tau}}$''\n\n'
                r'$\tau = %d$ fs' % tau)

# BI-EXPONENTIAL FIT 
elif fit_type == 3:
    def objective(t, t0, tau1, tau2, w):
        return np.piecewise(t, [((t >= 0) & (t <= t0)), (t > t0)],
                            [1, lambda t: w * np.exp(-(t - t0) / tau1) + (1 - w) * np.exp(-(t - t0) / tau2)])


    fit, _ = curve_fit(objective, t, state[fit_state][1], maxfev=100000,
                       bounds=((0, 0, 0, 0), (np.inf, np.inf, np.inf, 1.0)))
    t0, tau1, tau2, w = fit
    print('Vibrational relaxation = %d fs\n1st excited state lifetime component = %d fs\n2nd excited state '
          'lifetime component = %d fs\nWeight of 1st component = %1.3f' % (t0, tau1, tau2, w))
    x_line = np.arange(min(t), max(t), 1)
    y_line = objective(x_line, t0, tau1, tau2, w)
    if spin_multi == 1 or fit_state == 0:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{S_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 2:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{D_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 3:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{T_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    fit_info = (r'$\bf{Fit}$ $\bf{details:}$''\n\n'
                r'$t \leq t_0:$ $f(x)=1$''\n'
                r'$t > t_0:$ $f(x)=W\cdot e^{\frac{-(t-t_0)}{\tau_1}} +(1-W)\cdot e^{\frac{-(t-t_0)}{\tau_2}}$''\n\n'
                r'$t_0 = %d$ fs''\n'r'$\tau_1 = %d$ fs''\n'r'$\tau_2 = %d$ fs''\n'r'$W = %1.3f$' % (t0, tau1, tau2, w))

# BI-EXPONENTIAL FIT – JJ MASTER FIT
elif fit_type == 4:
    def objective(t, a, b, c):
        return np.piecewise(t, [((t >= 0) & (t <= a)), (t > a)],
                            [lambda t: np.exp(-t / b), lambda t: (np.exp(-a / b)) * np.exp(-(t - a) / c)])


    fit, _ = curve_fit(objective, t, state[fit_state][1], maxfev=100000,
                       bounds=((0, 0, 0), (np.inf, np.inf, np.inf)))
    a, b, c = fit
    total = a + c
    print('The 1st decay component = %d fs\nParameter = %d fs\nThe 2nd decay component = %d fs\nTotal decay = %d fs'
          % (a, b, c, total))
    x_line = np.arange(min(t), max(t), 1)
    y_line = objective(x_line, a, b, c)
    if spin_multi == 1 or fit_state == 0:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{S_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 2:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{D_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    elif spin_multi == 3:
        plt.plot(x_line, y_line, '--', label='fit – $\mathregular{T_%d}$' % fit_state, color='black',
                 linewidth=fig_linewidth)
    fit_info = (r'$\bf{Fit}$ $\bf{details:}$''\n\n'
                r'$t \leq t_0:$ $f(x)=e^{\frac{-t}{b}}$''\n'
                r'$t \leq t_0:$ $f(x)=e^{\frac{-a}{b}}\cdot e^{\frac{-(t-a)}{c}}$''\n\n'
                r'$a = %d$ fs''\n'r'$b = %d$ fs''\n'r'$c = %d$ fs' % (a, b, c))

# =============================================================================
# FIGURE CONSTRUCTION
ax.set_xlabel("Time / fs", fontsize=fig_fontsize)
ax.set_ylabel("Population", fontsize=fig_fontsize)
ax.tick_params(axis='x', labelsize=fig_fontsize)
ax.tick_params(axis='y', labelsize=fig_fontsize)
ax.legend(prop={"size": fig_fontsize}, loc=7)
ax.set_xlim([0, x_length])
ax.set_ylim([0, 1.01])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2.0)
plt.tight_layout()

if show_details:
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    ax.text(0.5, 0.5, fit_info, transform=ax.transAxes, fontsize=fig_fontsize, verticalalignment='center', bbox=props)

# =============================================================================
# PRINTING THE FIGURE
if print_fig == 1:
    plt.savefig(output, format=fig_format, dpi=fig_dpi)

plt.show()
