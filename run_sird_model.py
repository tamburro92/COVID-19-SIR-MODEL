import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.ticker as ticker
from sklearn.metrics import mean_squared_error
import datetime as dt
from model import load_from_NCVS, load_from_PC, load_Region_from_PC
from model import compute_SIRD, train_SIRD

AUTO_TUNE_MODEL_PARAMS = True

# initial and guessed model params
_N = 122000
_I0, _R0, _D0 = 450, 0, 0
_S0 = _N - _I0 - _R0
BETA, GAMMA, ALPHA = 0.2417, 0.037, 0.037
DAYS = 160
DATE_TO_START = '2020-02-24'


def main():
    N, I0, S0, R0, D0, beta, gamma, alpha = _N, _I0, _S0, _R0, _D0, BETA, GAMMA, ALPHA
    t = np.arange(0, DAYS)
    #t_days = np.arange(DATE_TO_START, DAYS, dtype='datetime64[D]')

    # load data
    confirmed, deaths, recovered = load_from_PC(remote=True)
    #confirmed, deaths, recovered = load_Region_from_PC('Campania', remote=True)
    #confirmed, deaths, recovered = load_from_NCVS('US', '2/24/20', remote=True)

    confirmed = confirmed - recovered - deaths

    # predict params
    if AUTO_TUNE_MODEL_PARAMS:
        beta, gamma, alpha, N, I0, R0, D0 = train_SIRD(S0, I0, R0, D0, N, beta, gamma, alpha, t[:len(confirmed.values)], confirmed.values, recovered.values, deaths.values, error_function= error_function)
        S0 = N - R0 - I0 - D0

    S, I, R, D = compute_SIRD(S0, I0, R0, D0, N, beta, gamma, alpha, t)
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    #S_plot, = ax.plot(t, S, 'b-+', alpha=0.5, label='S')
    I_plot, = ax.plot(t, I, 'r-+', alpha=0.5, label='I')
    R_plot, = ax.plot(t, R, 'g-+', alpha=0.5, label='R')
    D_plot, = ax.plot(t, D, 'b-+', alpha=0.5, label='D')

    ax.set_ylabel('Numbers of people')
    n_max = I.argmax()
    Max_plot,= plt.plot(t[n_max],I[n_max],'bx')
    Max_text_plot = ax.text(t[n_max],I[n_max], '({:.0f},{:.0f})'.format(t[n_max],I[n_max]))

    recovered_extended = np.concatenate((recovered.values, [None] * (DAYS - len(recovered.values))))
    infected_extended = np.concatenate((confirmed.values, [None] * (DAYS - len(confirmed.values))))
    deaths_extended = np.concatenate((deaths.values, [None] * (DAYS - len(deaths.values))))

    ax.plot(t, infected_extended, 'ro', alpha=1, label='I Observed', mfc='none')
    ax.plot(t, recovered_extended, 'go', alpha=1, label='R Observed', mfc='none')
    ax.plot(t, deaths_extended, 'bo', alpha=1, label='D Observed', mfc='none')


    ax.set_ylim(0, I[n_max]+I[n_max]*0.04)
    ax.set_xlim(0, t[n_max+10])

    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)


    def todate(x, pos):
        return (dt.datetime.strptime(DATE_TO_START, '%Y-%m-%d') + dt.timedelta(days=x)).strftime('%d/%m')

    fmt = ticker.FuncFormatter(todate)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_tick_params(rotation=45)
    plt.show()

def error_function(I, R, D, infected, recovered, deaths):
    a, b, c = 0.7, 0.15, 0.15
    n_windows = len(infected)
    #weights = np.linspace(0, 1, len(infected))
    #weights = np.ones(len(infected))
    # use exponential weights
    #weights = np.logspace(0, 2, len(infected))
    weights = np.concatenate((np.zeros(len(infected)-n_windows), np.logspace(0, 3, n_windows)))
    weights_norm = weights/np.sum(weights)
    #plt.plot(weights_norm)
    #plt.show()

    return a * mean_squared_error(infected, I, sample_weight=weights_norm) + a * mean_squared_error(recovered, R, sample_weight=weights_norm) + + a * mean_squared_error(deaths, D, sample_weight=weights_norm)


if __name__ == "__main__":
    main()
