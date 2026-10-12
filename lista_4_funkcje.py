import numpy as np 
import matplotlib.pyplot as plt
import scipy

def gen_poisson_process_1(lamb,T):
    """lamb: parametr rozkładu wykładniczego dla skoków T1, T2, ... """
    t=0
    I=0
    S = []
    while t<T:
        temp = np.random.random()
        t  = t - 1/lamb * np.log(temp)
        if t > T:
            break 
        S.append(t)
    return S

def generate_poisson_jumps(lamb, T):
    """Method 2: Generates jump moments using Poisson count and Uniform distribution."""
    n = np.random.poisson(lamb * T)
    if n > 0:
        # Uniform distribution on [0, T] sorted gives the jump times
        return np.sort(np.random.uniform(0, T, n))
    return np.array([])



def plot_poisson(S, T):

    x = np.concatenate(([0], S, [T]))
    y = np.concatenate(([0], np.arange(1, len(S) + 1), [len(S)]))
    
    # 'where=post' -CADLAG
    plt.step(x, y, where='post', label='Trajektoria $N_t$', color='blue', linewidth=2)
    
    # points in jump places
    plt.scatter(S, np.arange(1, len(S) + 1), color='red', zorder=3, label='Momenty skoków')

    plt.title(f"Trajektoria Procesu Poissona (Liczba skoków: {len(S)})")
    plt.xlabel("Czas (t)")
    plt.ylabel("N(t)")
    plt.grid(True, linestyle='--', alpha=0.7)

def get_poisson_value(jumps, t):
    """
    Returns N(t). 
    Supports both single float 't' and numpy arrays of 't'.
    """
    # np.searchsorted is the fast, vectorized equivalent of binary search
    return np.searchsorted(jumps, t, side='right') 

def run_simulation(lamb, T, t_checkpoints, num_sims=10000):
    # Dict for samples
    samples = {t: [] for t in t_checkpoints}
    
    for _ in range(num_sims):
        S = generate_poisson_jumps(lamb, T)
        
        # finding out what happend it t moments
        for t in t_checkpoints:
            nt_val = get_poisson_value(S, t)
            samples[t].append(nt_val)
            
    return samples

def empirical_mean_var(lamb,T,t_points, iterations):
    poisson_data = run_simulation(lamb, T, t_points, iterations)

    for t in t_points:
        data = poisson_data[t]
        empirical_mean = np.mean(data)
        empirical_variance = np.var(data)
        print(f"Wariancja i średnia liczby zdarzeń które się pojawiły do czasu {t}:")
        print(f"teoretyczna średnia: {(lamb*t)}, średnia empiryczna: {empirical_mean}")
        print(f"teoretyczna wariancja: {lamb*t}, wariancja empiryczna: {empirical_variance}")
        print("")
    return None



def get_compensated_value(jumps, lamb, t):
    """Returns the value of the compensated process: N(t) - lambda * tl
        works for t as float or np vector"""
    return get_poisson_value(jumps, t) - lamb * t

def plot_compensated_trajectories(lamb, T, num_paths=5):
    t_grid = np.linspace(0, T, 1000)
    plt.figure(figsize=(12, 5))
    
    for _ in range(num_paths):
        jumps = generate_poisson_jumps(lamb, T)
        # Vectorized call: get_poisson_value handles the whole t_grid at once
        path = get_compensated_value(jumps, lamb, t_grid)
        plt.plot(t_grid, path, linewidth=1)
        
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Trajektorie kompensowanego procesu Poissona ($\\lambda={lamb}$)")
    plt.xlabel("Czas (t)")
    plt.ylabel("$\\~{N}_t$")
    plt.grid(True, alpha=0.2)
    plt.show()


def estimate_moments(lamb, T, num_sims=20000):
    checkpoints = np.linspace(0, T, 11)
    results_matrix = np.zeros((num_sims, len(checkpoints)))
    
    for i in range(num_sims):
        jumps = generate_poisson_jumps(lamb, T)
        results_matrix[i, :] = get_compensated_value(jumps, lamb, checkpoints)
        
    means = np.mean(results_matrix, axis=0)
    variances = np.var(results_matrix, axis=0)
    
    print(f"{'t':>5} | {'Mean (Emp)':>10} | {'Mean (Theo)':>10} | {'Var (Emp)':>10} | {'Var (Theo)':>10}")
    print("-" * 65)
    for i, t in enumerate(checkpoints):
        print(f"{t:5.1f} | {means[i]:10.4f} | {0.0:10.1f} | {variances[i]:10.4f} | {lamb*t:10.1f}")

def check_martingale_property(lamb, T, s, t, num_sims=300000):
    n_s_values = []
    n_t_values = []
    
    for _ in range(num_sims):
        jumps = generate_poisson_jumps(lamb, T)
        n_s_values.append(get_compensated_value(jumps, lamb, s))
        n_t_values.append(get_compensated_value(jumps, lamb, t))
        
    n_s_values = np.array(n_s_values)
    n_t_values = np.array(n_t_values)
    
    # Round to avoid floating point issues during grouping
    unique_states = np.unique(np.round(n_s_values, 6))
    
    print(f"\n Sprawdzenie własności martyngałowej (s={s}, t={t})")
    print(f"{'Stan Ns':>10} | {'E[Nt | Ns]':>12} | {'Różnica':>10} | {'Liczba obserwacji':>8}")
    print("-" * 50)
    
    for state in unique_states:
        mask = np.round(n_s_values, 6) == state
        count = np.sum(mask)
        if count > 300: # Show only significant groups
            expected_nt = np.mean(n_t_values[mask])
            print(f"{state:10.4f} | {expected_nt:12.4f} | {abs(state-expected_nt):10.4f} | {count:8d}")

def poisson_gen_uni(size, lamb):
    L = np.full(size, np.power(np.e, -lamb))
    P =np.full(size, 1, dtype=float)
    k=np.full(size,0)

    mask = P > L
    while np.any(mask):
        active_count = np.sum(mask)
        random_values = np.random.uniform(0,1, size=active_count)
        P[mask] *= random_values
        k[mask] += 1
        mask = P > L
    return k - 1

#poisson_sample= poisson_gen_uni(1000000,40)

def poisson_gen_gamma(size, lamb):
    k = int(lamb - 2 * np.sqrt(lamb))
    if k < 1: k = int(lamb)
    s_k = np.random.gamma(k, 1/lamb, size=size)

    results = np.zeros(size, dtype=int)

    mask_early = s_k <= 1
    if np.any(mask_early):
        remaining_lamb = lamb * (1-s_k[mask_early])
        results[mask_early] = k + poisson_gen_uni(np.sum(mask_early), remaining_lamb)
    mask_late = ~mask_early
    if np.any(mask_late):
        results[mask_late] = np.random.binomial(k - 1, 1.0 / s_k[mask_late])

    return results 

def poisson_gen_(size, lamb):
    if lamb > 40:
        return poisson_gen_gamma(size, lamb)
    else:
        return poisson_gen_uni(size, lamb)
