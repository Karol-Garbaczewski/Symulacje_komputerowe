import numpy as np 
import matplotlib.pyplot as plt


def brown_1d(T_max, n=1000):
    """_summary_

    Args:
        T_max (_type_): max simulation time
        n (int, optional): discrete steps Defaults to 1000.
    """    
    N = np.random.randn(n)
    h = T_max/n # dt ?
    increses = np.concatenate(([0], N * np.sqrt(h)))
    B = np.cumsum(increses)
    T_grid = np.linspace(0,T_max, n+1)
    return B, T_grid

def brown_2d(T_max, n=1000):
    N_1 = np.random.randn(n)
    N_2 = np.random.randn(n)
    h = T_max/n # dt ?
    increses_N1 = np.concatenate(([0], N_1 * np.sqrt(h)))
    increses_N2 = np.concatenate(([0], N_2 * np.sqrt(h)))
    B_x = np.cumsum(increses_N1)
    B_y = np.cumsum(increses_N2)
    return B_x, B_y

def brown_bridge(T_max, trajectories=10, n=1000, plot=True):
    """_summary_

    Args:
        T_max (_type_): _description_
        trajectories (int, optional): _description_. Defaults to 10.
        n (int, optional): _description_. Defaults to 1000.
    """    
    results = np.zeros((trajectories, n + 1))
    for i in range(trajectories):
        B, T_grid = brown_1d(T_max, n=n)
        BB = B - T_grid / T_max * B[-1]
        results[i, :] = BB
        if plot:
            plt.plot(T_grid, BB)
            plt.xlabel("czas (t)")
            plt.ylabel("wartości ($B_{t}$)")
    
    return results, T_grid


def GBM(S0, mu, sigma, T_max, n, num_trajectories):
    trajectories = np.zeros((num_trajectories, n + 1))
    for i in range(num_trajectories):
        B, T_grid = brown_1d(T_max, n=n)
        S_t = S0 * np.exp((mu - np.power(sigma,2)/2) * T_grid + sigma * B)
        trajectories[i,:]= S_t

    return trajectories, T_grid