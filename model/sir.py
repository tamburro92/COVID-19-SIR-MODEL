import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error


def compute_SIR(S0, I0, R0, N, beta, gamma, t):
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return S, I, R

def error_function_mse(I, R, infected, recovered):
    a = 0.5
    weights = np.ones(len(infected))
    weights_norm = weights/np.sum(weights)

    return a * mean_squared_error(infected, I, sample_weight=weights_norm) + (1 - a) * mean_squared_error(recovered, R, sample_weight=weights_norm)

def train_SIR(S0, I0, R0, N, beta, gamma, t, infected, recovered, minimize_method = 'Powell', error_function=error_function_mse):

    # minimize_method = 'Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B'
    optimal = minimize(objective, [beta, gamma, N, I0, R0], args=(t, infected, recovered, error_function), method=minimize_method,
             bounds=[(0.01, 0.4), (0.01, 0.4), (100000, 900000), (0, 600), (0, 600)])
    S0 = N - R0 - I0
    print(optimal)
    return optimal.x

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def objective(input, t, infected, recovered, error_function):
    beta, gamma, N, I0, R0 = input
    S0 = N - R0 - I0
    S, I, R = compute_SIR(S0, I0, R0, N, beta, gamma, t)
    return error_function(I, R, infected, recovered)



