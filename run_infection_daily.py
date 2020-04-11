import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from model import load_from_PC_increment_daily
import matplotlib.ticker as ticker
import datetime as dt
from datetime import date
from scipy.stats import logistic, norm


DATE_TO_START = '2020-02-24'
DAYS = 100

def main():
    t = np.arange(0, DAYS)

    infection = load_from_PC_increment_daily(remote=True)

    x = np.arange(0, len(infection))

    # LOGISTIC FIT
    mean_est = sum(x * infection.values) / sum(infection.values)
    (a, c, d), _ = opt.curve_fit(logistic_function_pdf, x, infection.values, p0=[173084, 0.13, mean_est])
    print(a, c, d)
    infection_fit_logistic = logistic_function_pdf(t, a, c, d)


    loc, scale = logistic.fit(infection)
    print(loc, scale)
    infection_fit_logistic2 = logistic.pdf(t, loc, scale)

    # GAUSSIAN FIT
    mean_est = sum(x * infection.values) / sum(infection.values)
    sigma_est = sum(infection.values * (x - mean_est) ** 2) / sum(infection.values)
    (amplitude, mu, std), _ = opt.curve_fit(gaussian_function_pdf, x, infection.values, p0=[max(infection.values), mean_est, sigma_est])
    print(amplitude, mu, std)
    infection_fit_normal = gaussian_function_pdf(t, amplitude, mu, std)

    infection_extended = np.concatenate((infection.values, [None] * (DAYS - len(infection.values))))

    ax = plt.subplot(111)
    ax.plot(t, infection_extended, 'ro', alpha=1,  mfc='none', label='Infected')
    ax.plot(t, infection_fit_logistic, label='Logistic fit')
    #ax.plot(t, infection_fit_logistic2, label='Logistic2')
    ax.plot(t, infection_fit_normal, label='Normal fit')



    ax.set_ylabel('New Infected Daily')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)


    def todate(x, pos):
        return (dt.datetime.strptime(DATE_TO_START, '%Y-%m-%d') + dt.timedelta(days=x)).strftime('%d/%m')

    fmt = ticker.FuncFormatter(todate)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_tick_params(rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True)

    plt.savefig('forecast_daily/COVID_forecast_{}'.format(date.today().strftime('%d_%m')))
    plt.show()



def logistic_function_cdf(x, a, b, c, d):
    return a / (1. + np.exp(-c * (x - d))) + b

def logistic_function_pdf(x, a, c, d):
    return a * c * np.exp(-c * (x - d)) / (1. + np.exp(-c * (x - d)))**2

def gaussian_function_pdf(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

if __name__ == '__main__':
    main()
