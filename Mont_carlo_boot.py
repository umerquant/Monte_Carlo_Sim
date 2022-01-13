import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

sp_returns = np.loadtxt("sp_returns2.csv", delimiter = ",", usecols = 1)
sp_returns.size

I0 = 1000000
sims = 10000
days = 260

def path_simul_bootstr(daily_returns, days, I0, sims = 10000, seed = 123):
    
    days = int(days)
    
    np.random.seed(seed)
    ret = np.random.choice(daily_returns, size = days * sims, replace = True).reshape(sims, days)
    
    paths = (ret + 1).cumprod(axis = 1) * I0
    paths = np.hstack((np.ones(sims).reshape(sims, 1) * I0, paths))
    
    return paths

paths = path_simul_bootstr(sp_returns, days = days, I0 = I0, sims = sims)
paths
paths.shape

plt.figure(figsize = (20, 12))
plt.plot(paths.T)
plt.ylabel("Portfolio Value", fontsize = 15)
plt.xlabel("Days", fontsize = 15 )
plt.title("Monte Carlo Simulation - Bootstrapping", fontsize = 20)
plt.show() 


final_b = paths[:, -1]

plt.figure(figsize = (20, 12))
plt.hist(final_b, bins = 1000, density = True, alpha = 0.5, color = "red")
plt.show()
