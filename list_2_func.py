import numpy as np 
import scipy 
import matplotlib.pyplot as plt 
import bisect
import random 



def binary_search_bisect(arr, x):
    i = bisect.bisect_left(arr, x)
    while len(arr) > i + 1 and arr[i + 1] == x:  # we define empirical cdf as P(X <= x) so we look for the last occurence of x not the first
        i += 1
    if x > arr[-1]:
         return i # we get the len() value
    else:
        return i + 1 # we are comparing index with len function so we have to add one bcs indexing starts at 0 

def demp(X,x):
        X = sorted(X)
        return binary_search_bisect(X,x) / len(X) 

def drawdemp(X, color = "purple"):
    X_sorted = sorted(X)
    y_values = np.zeros(len(X_sorted))

    for i in range(len(X_sorted)):
        y_values[i] = demp(X_sorted, X_sorted[i])
    plt.plot(X_sorted, y_values, label=f"ecdf for sample {len(X)}", color=color, alpha=0.8)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('Empirical CDF')

def drawdemp_discrete(X, color = "purple"):
    X = sorted(X)
    cdf = scipy.stats.ecdf(X)
    plt.scatter(cdf.cdf.quantiles, cdf.cdf.probabilities, color=color, s=10, label=f"ecdf {len(X)}")

def generate_expon_sample(size=1000, lamb = 1):
    sample = scipy.stats.expon.rvs(scale = 1/lamb, size=size) #scale = 1/lambda
    return sample

def compare_expon_samples(*sizes, lamb = 1):
    colors = plt.cm.viridis(np.linspace(0,1,len(sizes)))
    for i, el in enumerate(sizes):
        temp = generate_expon_sample(size=el, lamb=1)
        drawdemp(temp,color = colors[i])
    # the analytical cdf
    arguments = np.linspace(0.01, 13, 1000)
    cdf= scipy.stats.expon.cdf(arguments, scale = 1 / lamb)
    plt.plot(arguments, cdf, label = "cdf of exponential distribution", color="red", linestyle="--")
    plt.legend()

ex4_mi = 2
ex4_sigma = 2 
ex4_lambda = 4

def cdf_expon(x,lamb=1):
    return 1 - np.exp(-lamb * x)
def cdf_normal(x, mi=ex4_mi, sigma=ex4_sigma):
    return 0.5 * (1 + scipy.special.erf((x - mi) / (sigma * np.sqrt(2))))

def expon_theoretical_density(size=1000, lamb = ex4_lambda):
    arguments = np.linspace(0.01, 13 * (1/lamb), size)
    sample = lamb* np.exp(-lamb * arguments)
    return arguments, sample

def normal_theoretical_density(size=1000, mi=ex4_mi, sigma=ex4_sigma):
    arguments = np.linspace(mi - 12 * sigma, mi + 12 * sigma, size)
    sample = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((arguments - mi) / sigma) ** 2)
    return arguments, sample

def inverse_cdf_norm_vec(F, u,max_iter=1000,tol=1e-12, params = (ex4_mi,ex4_sigma)):
    """params: - used to define mi and sigma
    F - cdf function
    u - array of uniform random variables
        """
    mi, sigma = params
    u = np.asarray(u)
    # we cover almost all range of normal distr
    low = np.full_like(u, mi - 12 * sigma)
    high = np.full_like(u, mi+ 12 * sigma)

    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = F(mid, *params)
        
        # using boolean masks (np.where - if for numpy arrays)
        # If F(mid) < u, the target is in the upper half
        go_higher = f_mid < u
        low = np.where(go_higher, mid, low) # mid becomes the new low for go_higher values, otherwise low stays the same
        high = np.where(go_higher, high, mid) # mid becomes the new high for not go_higher values, otherwise high stays the same
        
        # Check for convergence across the whole array
        if np.max(high - low) < tol:
            break
            
    return (low + high) / 2

def inverse_cdf_expon_vec(F, u, dx=0.01,max_iter=1000,tol=1e-12, lamb = ex4_lambda):
    u = np.asarray(u)
    low = np.zeros_like(u)
    high = np.full_like(u, - np.log(1 - 0.999999) / lamb) # we cover almost all range of exponential distr

    for _ in range(max_iter):
        mid = (low + high) / 2
        f_mid = F(mid, lamb)
        
        go_higher = f_mid < u
        low = np.where(go_higher, mid, low)
        high = np.where(go_higher, high, mid)
        
        if np.max(high - low) < tol:
            break
            
    return (low + high) / 2

def drawqq(sample, distr):
    """sample - data sample, distr - theoretical distribution"""
    sample = np.sort(sample)
    length = len(sample)
    p_points = (np.arange(length) + 0.5) / length # quantile 0 < p < 1
    theoretical_quantiles = distr(p_points)
    plt.scatter(theoretical_quantiles, sample)
    plt.plot([theoretical_quantiles[0], theoretical_quantiles[-1]], [theoretical_quantiles[0], theoretical_quantiles[-1]], 'r--')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('Q-Q Plot')
    plt.show()

def inverse_expon_an(u, lamb = 1):
    return -np.log(u) / lamb

def inverse_norm_an(u, mi=0, sigma=1):
    return mi + sigma * np.sqrt(2) * scipy.special.erfinv(2 * u - 1)

def inverse_cdf_cauchy_an(u, x0=0, gamma=1):
    return x0 + gamma * np.tan(np.pi * (u - 0.5))

def inverse_cdf_geom_an(u, p):
    return np.ceil(np.log(u) / np.log(1 - p))

def inverse_poisson_cdf(u, lam):
    k = np.zeros_like(u, dtype=int)
    p = np.full_like(u, np.exp(-lam))
    cdf = p.copy()
    
    unfinished = u > cdf

    while np.any(unfinished):
        k[unfinished] += 1
        p[unfinished] *= lam / k[unfinished]
        cdf[unfinished] += p[unfinished]

        # updating mask 
        unfinished = u > cdf
    return k 

def finite_distr(np_numerator, np_denominator, sample_size = 1000, probabilities = True):
    """np_numerator - array of numerators, np_denominator - array of denominators"""
    if len(np_numerator) != len(np_denominator):
        raise ValueError("numerator and denominator arrays must have the same length")
    np_numerator = np.asarray(np_numerator)
    np_denominator = np.asarray(np_denominator)

    #finding the least common multiple of the denominators to get a common denominator
    common_denominator = np.lcm.reduce(np_denominator)

    #adjusting numerators to the common denominator
    scaled_counts = np_numerator * (common_denominator // np_denominator)

    # we create an array of zeros with the size of common denominator to store the counts of each possible value
    prob_table = np.zeros(common_denominator, dtype=int)
    current_pos = 0

    for index, value in enumerate(scaled_counts):
        prob_table[current_pos:current_pos + value] += index 
        current_pos += value
    # [0,0,0,1,2,2,3,3,3] - example of prob table for 4 values 
    # 0 - prob of 1/3, 1 - prob of 1/9, 2 - prob of 2/9, 3 - prob of 1/3
    if probabilities==True:
        # drawing random indices from 0 - len(prob_table) or common denominator 
        random_indices = np.random.randint(0, common_denominator, size=sample_size)

        outcomes = prob_table[random_indices] # number of indices occured in the random sample 

        unique, counts_elements = np.unique(outcomes, return_counts=True)

        final_counts = np.zeros(len(np_numerator) , dtype=int)
        final_counts[unique] = counts_elements

        return final_counts / sample_size 
    else:
        return prob_table[np.random.randint(0, common_denominator, size=sample_size)]
    
def drawqq_discrete(data, probs=None, values=None):

    n = len(data)
    X = np.sort(data)
    
    # calculating quantile ranks 
    q_i = (np.arange(1, n + 1) - 0.5) / n
    
    # theoretical quantiles
    if probs is not None:
        # we assume values [0, 1, ..., k]
        if values is None:
            values = np.arange(len(probs))
        
        F = np.cumsum(probs)
        
        theoretical_quantiles = []
        for q in q_i:
            idx = np.searchsorted(F, q)
            theoretical_quantiles.append(values[idx])
        Q_theo = np.array(theoretical_quantiles)
    else:
        raise ValueError("Należy podać wektor prawdopodobieństw (probs).")


    plt.figure(figsize=(8, 6))
    plt.scatter(Q_theo, X, color='blue', alpha=0.6, label='Dane vs Teoria')
    

    min_val = min(np.min(X), np.min(Q_theo))
    max_val = max(np.max(X), np.max(Q_theo))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')
    
    plt.xlabel('theoretical quantiles')
    plt.ylabel('empirical quantiles')
    plt.title('Q-Q Plot discrete')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

#drawqq_discrete(finite_sample_2, probs=finite_probabilities, values=values)