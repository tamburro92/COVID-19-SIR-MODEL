import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def compute_SIRD(S0, I0, R0, D0, N, beta, gamma, alpha, t):
    y0 = S0, I0, R0, D0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, alpha))
    S, I, R, D = ret.T
    return S, I, R, D

def error_function_mse(I, R, D, infected, recovered, deaths):
    a, b, c = 1/3, 1/3, 1/3
    weights = np.ones(len(infected))
    weights_norm = weights/np.sum(weights)

    return a * mean_squared_error(infected, I, sample_weight=weights_norm) + a * mean_squared_error(recovered, R, sample_weight=weights_norm) + + a * mean_squared_error(deaths, D, sample_weight=weights_norm)

def train_SIRD(S0, I0, R0, D0, N, beta, gamma, alpha, t, infected, recovered, deaths, minimize_method = 'Powell', error_function=error_function_mse):

    # minimize_method = 'Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B'
    optimal = minimize(objective, [beta, gamma, alpha, N, I0, R0, D0], args=(t, infected, recovered, deaths, error_function), method=minimize_method,
             bounds=[(0.01, 0.4), (0.01, 0.4), (0.01, 0.4), (100000, 900000), (0, 600), (0, 600)])
    S0 = N - R0 - I0
    print(optimal)
    return optimal.x

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma, alpha):
    S, I, R, D = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I - alpha * I
    dRdt = gamma * I
    dDdt = alpha * I
    return dSdt, dIdt, dRdt, dDdt

def objective(input, t, infected, recovered, deaths, error_function):
    beta, gamma, alpha, N, I0, R0, D0 = input
    S0 = N - R0 - D0 - I0
    S, I, R, D = compute_SIRD(S0, I0, R0, D0, N, beta, gamma, alpha, t)
    return error_function(I, R, D, infected, recovered, deaths)



