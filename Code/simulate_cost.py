import numpy as np

def simulate_cost(w, P, seed=None):
    """
    Monte Carlo simulation returning mean total cost.
    w    : weight vector [w0, w1, ..., w_{Kmax}] (should be length >= Kmax + 2)
    P    : dictionary with all parameters
    seed : optional RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
        
    gamma = P['gamma']
    T = P['T']
    c_s = P['c_s']
    c_k = np.array(P['c_k'])
    iter_ = P['iter']
    Kmax = P['Kmax']
    min_lead = P.get('min_lead', 0)
    delta = P['delta']
    initial_inventory = float(P.get('initial_inventory', 0.0))
    assert 0 <= min_lead <= Kmax, "min_lead must be between 0 and Kmax"
    
    assert len(w) >= Kmax + 2, "w too short"
    assert len(c_k) >= Kmax + 1, "c_k too short"
    
    # Demand
    if 'demand_mean' in P and 'demand_rel_sigma' in P:
        demand_mean = np.asarray(P['demand_mean'], dtype=float)
        demand_rel_sigma = np.asarray(P['demand_rel_sigma'], dtype=float)
        assert demand_mean.shape == (T,), "demand_mean must have length T"
        assert demand_rel_sigma.shape == (T,), "demand_rel_sigma must have length T"

        demand_std = demand_mean * demand_rel_sigma
        sigma_ln_sq = np.log1p((demand_std / demand_mean) ** 2)
        mu_ln = np.log(demand_mean) - 0.5 * sigma_ln_sq
        sigmas = np.sqrt(sigma_ln_sq)
        demands = np.exp(mu_ln.reshape(1, -1) + rng.standard_normal((iter_, T)) * sigmas.reshape(1, -1))
    else:
        mu = P['mu']
        sigma = P['sigma']
        increase = P['increase']
        # Backward-compatible path: variance grows geometrically over time.
        sigmas = sigma * np.sqrt(increase ** np.arange(T))
        demands = np.exp(mu + rng.standard_normal((iter_, T)) * sigmas)
    
    Dtot = np.zeros((iter_, T))
    Dtot[:, 0] = demands[:, 0]
    for t in range(1, T):
        Dtot[:, t] = demands[:, t] + gamma * Dtot[:, t-1]
        
    # Forecasts
    E = np.full((iter_, T, Kmax + 1), np.nan)
    if 'forecast_rel_alpha' in P:
        forecast_rel_alpha = np.asarray(P['forecast_rel_alpha'], dtype=float)
        assert forecast_rel_alpha.shape == (Kmax + 1,), "forecast_rel_alpha must have length Kmax + 1"
        assert np.all((forecast_rel_alpha >= 0) & (forecast_rel_alpha < 1)), "forecast_rel_alpha values must be in [0, 1)"

        for k in range(Kmax + 1):
            if k >= T:
                break
            t_idx = np.arange(T - k)
            rel_noise = rng.uniform(-forecast_rel_alpha[k], forecast_rel_alpha[k], size=(iter_, len(t_idx)))
            E[:, t_idx, k] = Dtot[:, t_idx + k] * (1 + rel_noise)
    else:
        sigmaeps = P['sigmaeps']
        incr = P['incr']
        sigmaepses = np.zeros(Kmax + 1)
        if Kmax >= 1:
            sigmaepses[1:] = sigmaeps * np.sqrt(incr ** np.arange(1, Kmax + 1))

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
            
    s_alloc = np.zeros((iter_, T, Kmax + 1))

    I = np.zeros((iter_, T))
    shortfall = np.zeros((iter_, T))

    for t in range(T):
        if t == 0:
            I_prev = np.full(iter_, initial_inventory)
        else:
            I_prev = I[:, t-1]

        arrivals_from_past = np.zeros(iter_)
        for tau in range(t):
            kk = t - tau
            if min_lead <= kk <= Kmax:
                arrivals_from_past += s_alloc[:, tau, kk]

        def projected_available_at(target_period, max_current_lead_included):
            inventory = I_prev.copy()
            for m in range(t, target_period + 1):
                arrivals_m = np.zeros(iter_)
                for tau in range(m + 1):
                    kk = m - tau
                    if tau < t:
                        if min_lead <= kk <= Kmax:
                            arrivals_m += s_alloc[:, tau, kk]
                    elif tau == t:
                        if min_lead <= kk <= max_current_lead_included:
                            arrivals_m += s_alloc[:, tau, kk]

                available_m = (1 - delta) * inventory + arrivals_m
                if m == target_period:
                    return available_m

                demand_forecast = E[:, t, m - t]
                inventory = np.maximum(available_m - demand_forecast, 0)

            return inventory

        for k in range(min_lead, Kmax + 1):
            target_period = t + k
            if target_period >= T:
                continue

            eps_slice = E[:, t, :k+1]
            if k == 0:
                eps_slice = eps_slice.reshape(-1, 1)

            # Coverage for a future period should include capacity that arrived
            # earlier and remains available, not only capacity arriving exactly
            # in the target period.
            coverage = projected_available_at(target_period, k - 1)

            s_alloc[:, t, k] = w[0] + eps_slice @ w[1:k+2] - coverage
            s_alloc[:, t, k] = np.maximum(s_alloc[:, t, k], 0)

        arrivals = arrivals_from_past
        if min_lead == 0:
            arrivals += s_alloc[:, t, 0]

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
        'shortfall': shortfall,
        'totalShortfall': totalShort,
        'meanTotalShortfall': np.mean(totalShort),
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
