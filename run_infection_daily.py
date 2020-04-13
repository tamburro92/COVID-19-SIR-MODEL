import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from model import load_from_PC_increment_daily, load_from_PC_delta_infected_daily
import matplotlib.ticker as ticker
import datetime as dt
from datetime import date
from sklearn.metrics import mean_squared_error



DATE_TO_START = '2020-02-24'
DAYS = 80

def main():
    t = np.arange(0, DAYS)

    # ----------------------------#
    # ----- INFECTION DAILY ----- #
    infection = load_from_PC_increment_daily(remote=True)
    infection_extended = np.concatenate((infection.values, [None] * (DAYS - len(infection.values))))

    x = np.arange(0, len(infection))
    # LOGISTIC FIT
    mean_est = sum(x * infection.values) / sum(infection.values)
    (a, c, d), _ = opt.curve_fit(logistic_function_pdf, x, infection.values, p0=[173084, 0.13, mean_est])
    print(a, c, d)
    infection_fit_logistic = logistic_function_pdf(t, a, c, d)

    # GAUSSIAN FIT
    mean_est = sum(x * infection.values) / sum(infection.values)
    sigma_est = sum(infection.values * (x - mean_est) ** 2) / sum(infection.values)
    (amplitude, mu, std), _ = opt.curve_fit(gaussian_function_pdf, x, infection.values, p0=[max(infection.values), mean_est, sigma_est])
    print(amplitude, mu, std)
    infection_fit_normal = gaussian_function_pdf(t, amplitude, mu, std)

    # GOMPERTZ FIT
    (a,b,n), pcov = opt.curve_fit(shifted_gompertz_function_pdf, x, infection.values, p0=[1000, 1, 0.1])
    print(a,b,n)
    infection_fit_gompertz = shifted_gompertz_function_pdf(t, a, b, n)

    # --------------------------- #
    # -- DELTA INFECTION DAILY -- #
    delta_infection = load_from_PC_delta_infected_daily(remote=True)
    infection_delta_extended = np.concatenate((delta_infection.values, [None] * (DAYS - len(delta_infection.values))))

    x = np.arange(0, len(delta_infection))
    # LOGISTIC FIT
    mean_est = sum(x * delta_infection.values) / sum(delta_infection.values)
    (a, c, d), pcov = opt.curve_fit(logistic_function_pdf, x, delta_infection.values, p0=[173084, 0.13, mean_est])
    print(a, c, d)
    infection_delta_fit_logistic = logistic_function_pdf(t, a, c, d)
    print('MSE: {:.0f}'.format(mean_squared_error(infection_delta_fit_logistic[:len(delta_infection.values)], delta_infection.values)))

    # GAUSSIAN FIT
    mean_est = sum(x * delta_infection.values) / sum(delta_infection.values)
    sigma_est = sum(delta_infection.values * (x - mean_est) ** 2) / sum(delta_infection.values)
    (amplitude, mu, std), pcov = opt.curve_fit(gaussian_function_pdf, x, delta_infection.values, p0=[max(delta_infection.values), mean_est, sigma_est])
    print(amplitude, mu, std)
    infection_delta_fit_normal = gaussian_function_pdf(t, amplitude, mu, std)
    print('MSE: {:.0f}'.format(mean_squared_error(infection_delta_fit_normal[:len(delta_infection.values)], delta_infection.values)))

    # GOMPERTZ FIT
    (a,b,n), pcov = opt.curve_fit(shifted_gompertz_function_pdf, x, delta_infection.values, p0=[114691, 0.09, 10])
    print(a,b,n)
    infection_delta_fit_gompertz = shifted_gompertz_function_pdf(t, a, b, n)
    print('MSE: {:.0f}'.format(mean_squared_error(infection_delta_fit_gompertz[:len(delta_infection.values)], delta_infection.values)))

    # --------------------------- #
    # ------- PLOT RESULTS ------ #
    def todate(x, pos):
        return (dt.datetime.strptime(DATE_TO_START, '%Y-%m-%d') + dt.timedelta(days=x)).strftime('%d/%m')
    fmt = ticker.FuncFormatter(todate)
    # subplot 1
    ax = plt.subplot(211)
    ax.plot(t, infection_extended, 'ro', alpha=1,  mfc='none', label='Infected')
    ax.plot(t, infection_fit_logistic, label='Logistic fit')
    ax.plot(t, infection_fit_normal, label='Normal fit')
    ax.plot(t, infection_fit_gompertz, label='infection_fit_gompertz')
    ax.set_ylabel('Infected Daily')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_tick_params(rotation=30)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True)

    # subplot 2
    ax2 = plt.subplot(212)
    ax2.plot(t, infection_delta_extended, 'ro', alpha=1,  mfc='none', label='Infected')
    ax2.plot(t, infection_delta_fit_logistic, label='Logistic fit')
    ax2.plot(t, infection_delta_fit_normal, label='Normal fit')
    ax2.plot(t, infection_delta_fit_gompertz, label='Gompertz fit')
    ax2.set_ylabel('Infected Delta Daily')
    legend = ax2.legend()
    legend.get_frame().set_alpha(0.5)
    ax2.xaxis.set_major_formatter(fmt)
    ax2.xaxis.set_tick_params(rotation=45)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax2.yaxis.set_tick_params(length=0)
    ax2.xaxis.set_tick_params(length=0)
    ax2.grid(b=True)

    plt.savefig('forecast_daily/COVID_forecast_{}'.format(date.today().strftime('%d_%m')))
    plt.show()



def logistic_function_cdf(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b

def logistic_function_pdf(x, a, c, d):
    return a * c * np.exp(-c * (x - d)) / (1. + np.exp(-c * (x - d)))**2

def gaussian_function_pdf(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

def shifted_gompertz_function_pdf(x, amplitude, b, n):
    return amplitude * b * np.exp(-b * x - n * np.exp(-b*x)) * (1 + n * (1 - np.exp(-b * x)))

if __name__ == '__main__':
    main()
