import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.metrics import mean_squared_error
from model import load_from_NCVS, load_from_PC, load_Region_from_PC
import pandas as pd

SLIDER_VISIBLE = False
AUTO_TUNE_MODEL_PARAMS = True

# initial and guessed model params
_N = 122000
_I0, _R0 = 450, 0
_S0 = _N - _I0 - _R0
BETA, GAMMA = 0.2417, 0.037
DAYS = 160
DATE_TO_START = '2020-02-24'


def main():
    N, I0, S0, R0, beta, gamma = _N, _I0, _S0, _R0, BETA, GAMMA
    t = np.arange(0, DAYS)
    #t_days = np.arange(DATE_TO_START, DAYS, dtype='datetime64[D]')

    # load data
    confirmed, deaths, recovered = load_from_PC(remote=True)
    #confirmed, deaths, recovered = load_Region_from_PC('Campania', remote=True)
    #confirmed, deaths, recovered = load_from_NCVS('US', '2/24/20', remote=True)

    recovered = recovered + deaths
    confirmed = confirmed - recovered

    # predict params
    if AUTO_TUNE_MODEL_PARAMS:
        beta, gamma, N, I0, R0 = train(I0, R0, S0, N, beta, gamma, t[:len(confirmed.values)], confirmed.values, recovered.values)
        S0 = N - R0 - I0

    S, I, R = compute_SIR(I0, R0, S0, N, beta, gamma, t)
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    if SLIDER_VISIBLE:
        plt.subplots_adjust(bottom=0.25)
    #S_plot, = ax.plot(t, S, 'b-+', alpha=0.5, label='S')
    I_plot, = ax.plot(t, I, 'r-+', alpha=0.5, label='I')
    R_plot, = ax.plot(t, R, 'g-+', alpha=0.5, label='R')
    ax.set_xlabel('Days since 24/02/20')
    ax.set_ylabel('Numbers of people')
    n_max = I.argmax()
    Max_plot,= plt.plot(t[n_max],I[n_max],'bx')
    Max_text_plot = ax.text(t[n_max],I[n_max], '({:.0f},{:.0f})'.format(t[n_max],I[n_max]))

    recovered_extended = np.concatenate((recovered.values, [None] * (DAYS - len(recovered.values))))
    infected_extended = np.concatenate((confirmed.values, [None] * (DAYS - len(confirmed.values))))
    ax.plot(t, infected_extended, 'ro', alpha=1, label='I Observed', mfc='none')
    ax.plot(t, recovered_extended, 'go', alpha=1, label='R Observed', mfc='none')

    ax.set_ylim(0, I[n_max]+I[n_max]*0.035)
    ax.set_xlim(0, 60)

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)


    axBeta = plt.axes([0.25, 0.15, 0.65, 0.03], visible= SLIDER_VISIBLE)
    axGamma = plt.axes([0.25, 0.1, 0.65, 0.03], visible= SLIDER_VISIBLE)
    axN = plt.axes([0.25, 0.05, 0.65, 0.03], visible= SLIDER_VISIBLE)
    axI0 = plt.axes([0.25, 0.00, 0.65, 0.03], visible= SLIDER_VISIBLE)

    sBeta = Slider(axBeta, 'beta', 0.001, 1, valinit=beta, valstep=0.0001, valfmt='%1.4f')
    sGamma = Slider(axGamma, 'gamma', 0.001, 1, valinit=gamma, valstep=0.001, valfmt='%1.4f')
    sN = Slider(axN, 'N', 1000, 1000000, valinit=N, valstep=1000, valfmt='%1d')
    sI0 = Slider(axI0, 'I(0)', -100, 1000, valinit=I0, valstep=1, valfmt='%1d')

    def update(val):
        N = sN.val
        I0 = sI0.val
        R0 = 0
        S0 = N - I0 - R0
        beta = sBeta.val
        gamma = sGamma.val

        # predict params
        if AUTO_TUNE_MODEL_PARAMS:
            beta, gamma, N, I0, R0 = train(I0, R0, S0, N, beta, gamma, t[:len(confirmed.values)], confirmed.values,
                                           recovered.values)
            S0 = N - R0 - I0
        S, I, R = compute_SIR(I0, R0, S0, N, beta, gamma, t)
        # S_plot.set_ydata(S)
        I_plot.set_ydata(I)
        R_plot.set_ydata(R)
        n_max = I.argmax()
        Max_plot.set_data(t[n_max], I[n_max])
        Max_text_plot.set_position((t[n_max], I[n_max]))
        Max_text_plot.set_text('({:.0f},{:.0f})'.format(t[n_max], I[n_max]))
        fig.canvas.draw_idle()
        print('MSE I:{:.0f}'.format(loss(I[:len(confirmed)], R[:len(confirmed)], confirmed, recovered)))

    if SLIDER_VISIBLE:
        sBeta.on_changed(update)
        sGamma.on_changed(update)
        sN.on_changed(update)
        sI0.on_changed(update)
    plt.show()


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def compute_SIR(I0, R0, S0, N, beta, gamma, t):
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R


def objective(input, t, infected, recovered):
    beta, gamma, N, I0, R0 = input
    S0 = N - R0 - I0
    S, I, R = compute_SIR(I0, R0, S0, N, beta, gamma, t)
    return loss(I, R, infected, recovered)


def loss(I, R, infected, recovered):
    a = 0.7
    #weights = np.linspace(0, 1, len(infected))
    #weights = np.ones(len(infected))
    # use exponential weights
    weights = np.logspace(0, 2, len(infected))
    #weights[:weights.size-14] = 0
    weights_norm = weights/np.sum(weights)
    #plt.plot(weights_norm)
    #plt.show()

    return a * mean_squared_error(infected, I, sample_weight=weights_norm) + (1 - a) * mean_squared_error(recovered, R, sample_weight=weights_norm)


def train(I0, R0, S0, N, beta, gamma, t, infected, recovered):

    # method = 'Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B'
    optimal = minimize(objective, [beta, gamma, N, I0, R0], args=(t, infected, recovered), method='Powell',
             bounds=[(0.01, 0.4), (0.01, 0.4), (100000, 900000), (0, 600), (0, 600)])
    S0 = N - R0 - I0
    print(optimal)
    return optimal.x


if __name__ == "__main__":
    main()