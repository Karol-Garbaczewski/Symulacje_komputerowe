import numpy as np 
import matplotlib.pyplot as plt 
from lista_4_funkcje import plot_poisson, get_poisson_value
import scipy
import math

def example_func(x):
    return np.power(x+1,-1)

def NHPP_thining(M, T, func):
    """M >= max(func) otherwise it wont work """
    I=0
    S = []
    t = - 1/M * np.log(np.random.uniform())
    while t <=T:
        if np.random.uniform() <= func(t)/M:
            I += 1 
            S.append(t)
        t += - 1/M * np.log(np.random.uniform())
    return np.array(S) 

def numeric_integrate(func, t):
    integral,_ = scipy.integrate.quad(func,0,t)
    return integral 

def inverse_lambda(x):
    return np.log(x+1)

# SPOSÓB 1 
# INVERSE FUNCTION OF lambda - cummulated intenisty function
def m_t(x):
    return np.log(x+1)

# func - inversed 1-exp(integral(lambda(t))), lambda(t) from previous exercise
def NHPP_inverse(func, T):
    t=0
    I=0
    S =[]
    tau = func(np.random.uniform(), t)
    t += tau
    while t < T:
        I += 1
        S.append(t)
        tau = func(np.random.uniform(), t)
        t += tau
    return sorted(S)

def NHPP_inverse_2(func, T):
    """func - integral of lambda(t), S = inversed func(t)/func(T), t is argument"""
    N_T = np.random.poisson(func(T))
    S = np.power(np.e, np.random.uniform(size=N_T) * np.log(T+1)) -1
    return sorted(S)

def example_func_1(x):
    return np.sqrt(x)/100

def example_func_2(x):
    return np.power(x+1, -1)

def merged_func(x):
    return np.power(x+1, -1) + np.sqrt(x)/100

def empirical_cf(t_vec, data):
    """
    t_vec: wektor punktów t, dla których liczymy wartość funkcji
    data: jednowymiarowa tablica z próbką losową (X_1, ..., X_n)
    """
    # Używamy rozgłaszania (broadcasting) do policzenia macierzy t * X
    # t_vec[:, None] tworzy pionową kolumnę, data tworzy poziomy wiersz
    matrix = t_vec[:, None] * data
    
    # Liczymy średnią po osi danych (wierszach)
    ecf = np.mean(np.exp(1j * matrix), axis=1)
    return ecf

# 2. Definicje Funkcji Teoretycznych
def theoretical_cf_normal(t_vec):
    return np.exp(-0.5 * t_vec**2)

def theoretical_cf_exponential(t_vec, lam=1.0):
    return lam / (lam - 1j * t_vec)

def plot_compound_poisson(S,J, T):
    x = np.concatenate(([0], S, [T]))
    y = np.concatenate(([0], J, [J[-1]]))

    plt.figure(figsize=(10, 5))
    
    # 'where=post' -CADLAG
    plt.step(x, y, where='post', label='Trajektoria $N_t$', color='blue', linewidth=2)
    
    # points in jump places
    plt.scatter(x, y, color='red', zorder=3, label='Momenty skoków')

    plt.title(f"Trajektoria Procesu Poissona (Liczba skoków: {len(S)})")
    plt.xlabel("Czas (t)")
    plt.ylabel("N(t)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def NHPP_thining_normal(M, T, func, mean=0, std=1):
    """M >= max(func) otherwise it wont work """
    I=0
    S = []
    t = - 1/M * np.log(np.random.uniform())
    while t <=T:
        if np.random.uniform() <= func(t)/M:
            I += 1 
            S.append(t)
        t += - 1/M * np.log(np.random.uniform())
    jumps = np.cumsum(np.random.normal(loc=mean, scale=std, size=len(S)))
    return np.array(S), jumps

def NHPP_thining_cauchy(M, T, func):
    I=0
    S = []
    t = - 1/M * np.log(np.random.uniform())
    while t <=T:
        if np.random.uniform() <= func(t)/M:
            I += 1 
            S.append(t)
        t += - 1/M * np.log(np.random.uniform())
    jumps = np.cumsum(np.random.standard_cauchy(size=len(S)))
    return np.array(S), jumps

def NHPP_thining_expon(M, T, func, lamb=1):
    I=0
    S = []
    t = - 1/M * np.log(np.random.uniform())
    while t <=T:
        if np.random.uniform() <= func(t)/M:
            I += 1 
            S.append(t)
        t += - 1/M * np.log(np.random.uniform())
    jumps = np.cumsum(np.random.exponential(scale=1/lamb, size=len(S)))
    return np.array(S), jumps

def get_value_risk_process(S,jumps,t,r0,c):
    """Returns value of our process ad given (t) time."""
    return r0+ c*t - jumps[get_poisson_value(S, t)-1]


def calculate_trajectories_risk_process(lamb_func, lamb, T, M, r0, c, num_paths=1):
    t_grid = np.linspace(0, T, int(T*100))
    all_trajectories = []
    for _ in range(num_paths):
        S, J = NHPP_thining_expon(M, T, lamb_func, lamb=lamb)
        if len(S) == 0:
            S = np.array([0])
            J = np.array([0])
        else:
            S = np.concatenate(([0], S, [S[-1]]))
            J = np.concatenate(([0], J, [J[-1]]))
        all_trajectories.append(get_value_risk_process(S, J, t_grid,r0,c))
    return np.array(all_trajectories)


def plot_trajectories(lamb_func, lamb, T, M, r0, c, num_paths=1):
    t_grid = np.linspace(0, T, int(T*100))

    plt.figure(figsize=(12,5))
    paths = calculate_trajectories_risk_process(lamb_func, lamb, T, M, r0, c, num_paths=num_paths)

    for path in paths:
        plt.plot(t_grid, path, linewidth=1)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Trajektorie procesu ryzyka ($\\lambda(t)={lamb}$)")
    plt.xlabel("Czas (t)")
    plt.ylabel("$\\~{R}_t$")
    plt.grid(True, alpha=0.2)
    plt.show()

def probability_of_bankrupcy(lamb_func, lamb, T_list, M, r0, c, num_paths=100000):
    bankrupt_counts = {t: 0 for t in T_list}
    
    for _ in range(num_paths):
        # Generating trajectory for max T 
        max_T = max(T_list)
        S, J = NHPP_thining_expon(M, max_T, lamb_func, lamb=lamb)
        
        if len(S) > 0:
            # we can only become bankrupt at moments of jumps
            # calculate capital at jumps
            capital_at_jumps = r0 + c * S - J
            
            # check for each time period
            for t in T_list:
                # jumps to time t
                mask = S <= t
                jumps_before_t = capital_at_jumps[mask]
                
                # if in any of the moments capital < 0 -> bankrupt
                if len(jumps_before_t) > 0 and (jumps_before_t < 0).any():
                    bankrupt_counts[t] += 1
                    
    return {t: count / num_paths for t, count in bankrupt_counts.items()}

