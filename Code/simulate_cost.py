import numpy as np

def simulate_cost(w, P, seed=None):
    """
    Monte Carlo simulation returning mean total cost.
    w    : weight vector [w0, w1, ..., w_{Kmax}] (should be length >= Kmax + 2)
    P    : dictionary with all parameters
    seed : optional RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    T = P['T']
    c_s = P['c_s']
    c_k = np.array(P['c_k'])
    iter_ = P['iter']
    Kmax = P['Kmax']
    delta = P['delta']
    mu = P['mu']
    sigma = P['sigma']
    increase = P['increase']
    sigmaeps = P['sigmaeps']
    incr = P['incr']
    
    assert len(w) >= Kmax + 2, "w too short"
    assert len(c_k) >= Kmax + 1, "c_k too short"
    
    # Demand
    # The thesis scales variance over time, so std dev scales with sqrt(.)
    sigmas = sigma * np.sqrt(increase ** np.arange(T))
    demands = np.exp(mu + rng.standard_normal((iter_, T)) * sigmas)
    
    # Demand always carries over fully, so each period's total demand equals
    # the current raw demand plus all demand accumulated from earlier periods.
    Dtot = np.cumsum(demands, axis=1)
        
    # Forecasts
    sigmaepses = np.zeros(Kmax + 1)
    if Kmax >= 1:
        sigmaepses[1:] = sigmaeps * np.sqrt(incr ** np.arange(1, Kmax + 1))
        
    E = np.full((iter_, T, Kmax + 1), np.nan)
    for k in range(Kmax + 1):
        if k >= T:
            break
        t_idx = np.arange(T - k)
        if k == 0:
            E[:, t_idx, 0] = Dtot[:, t_idx]
        else:
            s = sigmaepses[k]
            mu_k = -0.5 * (s ** 2)
            eps_draws = rng.lognormal(mean=mu_k, sigma=s, size=(iter_, len(t_idx)))
            E[:, t_idx, k] = Dtot[:, t_idx + k] * eps_draws
            
    # Decisions s_{t, k} and inventory balance
    s_alloc = np.zeros((iter_, T, Kmax + 1))
    
    for t in range(T):
        for k in range(Kmax + 1):
            if t + k >= T:
                continue
            eps_slice = E[:, t, :k+1]
            if k == 0:
                eps_slice = eps_slice.reshape(-1, 1)
            # w[0] + sum of (eps_slice * w[1:k+2])
            s_alloc[:, t, k] = w[0] + eps_slice @ w[1:k+2]
            s_alloc[:, t, k] = np.maximum(s_alloc[:, t, k], 0)
            
    I = np.zeros((iter_, T))
    shortfall = np.zeros((iter_, T))
    
    for t in range(T):
        arrivals = np.zeros(iter_)
        for tau in range(t + 1):
            kk = t - tau
            if kk <= Kmax:
                arrivals += s_alloc[:, tau, kk]
                
        if t == 0:
            I_prev = np.zeros(iter_)
        else:
            I_prev = I[:, t-1]
            
        available = (1 - delta) * I_prev + arrivals
        shortfall[:, t] = np.maximum(Dtot[:, t] - available, 0)
        served = Dtot[:, t] - shortfall[:, t]
        I[:, t] = available - served
        
    # Costs
    invCost = np.sum(np.sum(s_alloc * c_k.reshape(1, 1, -1), axis=2), axis=1)
    sfCost = c_s * np.sum(shortfall, axis=1)
    totalCost = invCost + sfCost
    meanCost = np.mean(totalCost)
    
    # Fill rate calculation
    totalDemand = np.sum(Dtot, axis=1)
    totalShort = np.sum(shortfall, axis=1)
    fillRatePath = 1 - totalShort / totalDemand
    fillRateOverall = np.mean(fillRatePath)
    fillRatePerPeriod = 1 - np.sum(shortfall, axis=0) / np.sum(Dtot, axis=0)
    
    probStockout = np.mean(np.any(shortfall > 1e-12, axis=1))
    cycleService = 1 - probStockout
    
    details = {
        'totalCost': totalCost,
        'invCost': invCost,
        'sfCost': sfCost,
        'p95': np.percentile(totalCost, 95),
        'I': I,
        'S_tot': np.sum(s_alloc, axis=2),
        'Dtot': Dtot,
        's': s_alloc,
        'fillRateOverall': fillRateOverall,
        'fillRatePerPeriod': fillRatePerPeriod,
        'fillRatePath': fillRatePath,
        'cycleService': cycleService
    }
    
    return meanCost, details
