import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from simulate_cost import simulate_cost

# 1. Parameters
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)

demand_rel_sigma_scenarios = {
    'Low': np.array([0.05, 0.10, 0.15, 0.20]),
    'Med': np.array([0.10, 0.20, 0.30, 0.40]),
    'High': np.array([0.20, 0.40, 0.60, 0.80]),
}
forecast_rel_alpha_scenarios = {
    # We keep all forecast horizons uncertain and let the forecast range widen
    # the further into the future we look.
    'Low': np.array([0.05, 0.10, 0.15, 0.20]),
    'Med': np.array([0.10, 0.20, 0.30, 0.40]),
    'High': np.array([0.20, 0.40, 0.60, 0.80]),
}

capacity_cost_now = 0.80
period_discount_rate = 0.20
capacity_costs = (capacity_cost_now / (1 + period_discount_rate) ** np.arange(4)).tolist()

params = {
    'gamma': 1.0,
    'T': 4,
    'c_s': 1,
    'c_k': capacity_costs,
    'iter': 10000,
    'Kmax': 3,
    'min_lead': 1,
    'delta': 0,
    'initial_inventory': 5.0,
    'demand_mean': np.array([5.0, 5.0, 5.0, 5.0]),
    'demand_rel_sigma': demand_rel_sigma_scenarios['Med'].copy(),
    'forecast_rel_alpha': forecast_rel_alpha_scenarios['Med'].copy(),
    'estmu': 0,
}

# 2. Initial weights
# Start from a smooth, horizon-decaying policy: near-term signals matter most,
# but all forecasts retain some influence because long-lead capacity is cheaper.
w0 = np.array([
    0.12,  # baseline investment
    0.35,  # e_{t,0}
    0.10,  # e_{t,1}
    0.06,  # e_{t,2}
    0.035, # e_{t,3}
])

OPT_SEED = 12345

def sample_raw_demands(P, seed=None):
    rng = np.random.default_rng(seed)
    demand_mean = np.asarray(P['demand_mean'], dtype=float)
    demand_rel_sigma = np.asarray(P['demand_rel_sigma'], dtype=float)

    demand_std = demand_mean * demand_rel_sigma
    sigma_ln_sq = np.log1p((demand_std / demand_mean) ** 2)
    mu_ln = np.log(demand_mean) - 0.5 * sigma_ln_sq
    sigmas = np.sqrt(sigma_ln_sq)

    return np.exp(mu_ln.reshape(1, -1) + rng.standard_normal((P['iter'], P['T'])) * sigmas.reshape(1, -1))

def compute_capacity_paths(details, P):
    s_alloc = details['s']
    inventory = details['I']
    iter_, T, _ = s_alloc.shape
    min_lead = P.get('min_lead', 0)
    initial_inventory = float(P.get('initial_inventory', 0.0))
    delta = P['delta']

    arrivals = np.zeros((iter_, T))
    for service_t in range(T):
        for decision_t in range(T):
            lead = service_t - decision_t
            if min_lead <= lead <= P['Kmax']:
                arrivals[:, service_t] += s_alloc[:, decision_t, lead]

    available = np.zeros((iter_, T))
    for t in range(T):
        if t == 0:
            inventory_prev = np.full(iter_, initial_inventory)
        else:
            inventory_prev = inventory[:, t-1]
        available[:, t] = (1 - delta) * inventory_prev + arrivals[:, t]

    served = details['Dtot'] - details['shortfall']
    cum_capacity = initial_inventory + np.cumsum(arrivals, axis=1)
    cum_demand = np.cumsum(details['Dtot'], axis=1)
    cum_served = np.cumsum(served, axis=1)

    return {
        'arrivals': arrivals,
        'available': available,
        'served': served,
        'cum_capacity': cum_capacity,
        'cum_demand': cum_demand,
        'cum_served': cum_served,
    }

def style_bar_axis(ax, xticks):
    ax.set_xticks(xticks)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', zorder=0)

BAR_STYLE = {
    'edgecolor': 'black',
    'linewidth': 1.0,
    'zorder': 3,
}

def finalize_figure(fig, filename, rect=None):
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(FIGURES_DIR / filename, bbox_inches='tight')
    if 'agg' in plt.get_backend().lower():
        plt.close(fig)
    else:
        plt.show(block=False)

# 3. Objective handle
def obj(w):
    mc, _ = simulate_cost(w, params, seed=OPT_SEED)
    return mc

# 4. Optimise
bounds = [(0, None) for _ in range(len(w0))]
print('Optimising...')
# COBYQA is a derivative-free trust-region method for bound-constrained black-box problems.
res = minimize(obj, w0, bounds=bounds, method='COBYQA', options={'disp': True, 'maxiter': 1000})
w_best = res.x
fval = res.fun

print(f'Best mean cost: {fval:.2f}')
print('Best weights:')
print(w_best)

_, det = simulate_cost(w_best, params, seed=123)

# ----------------- Graphs & Tables -----------------

# Baseline performance
BASE_SEED = 12345
meanCost, det_base = simulate_cost(w_best, params, seed=BASE_SEED)

MTot = meanCost
MInv = np.mean(det_base['invCost'])
MSF = np.mean(det_base['sfCost'])
MShort = det_base['meanTotalShortfall']

print(f'\n=== BASELINE (seed={BASE_SEED}) ===')
print(f'Expected total cost      : {MTot:.4f}')
print(f'  Investment cost    : {MInv:.4f}')
print(f'  Shortfall cost     : {MSF:.4f}')
print(f'  Mean total shortfall: {MShort:.4f}')

print('Optimised weights (w_best):')
print(w_best)

# Investment profile by lead time
active_leads = np.arange(params['min_lead'], params['Kmax'] + 1)
avg_s_by_k = np.mean(det_base['s'], axis=(0, 1))[active_leads]
fig, ax = plt.subplots()
ax.bar(active_leads, avg_s_by_k, **BAR_STYLE)
ax.set_xlabel('Lead time k')
ax.set_ylabel('Avg s_{t,k} per period')
ax.set_title('Baseline investment profile by lead time')
style_bar_axis(ax, active_leads)
finalize_figure(fig, 'baseline_investment_profile.pdf')

# Figure 1: demand distributions by period
demands = sample_raw_demands(params, seed=BASE_SEED)
Taxis = np.arange(1, params['T'] + 1)

fig, ax = plt.subplots()
box = ax.boxplot(
    [demands[:, t] for t in range(params['T'])],
    tick_labels=Taxis,
    patch_artist=True,
    showfliers=False,
)
for patch in box['boxes']:
    patch.set(facecolor='#9ecae1', edgecolor='black', linewidth=1.0)
for median in box['medians']:
    median.set(color='black', linewidth=1.2)
for artist_group in ('whiskers', 'caps'):
    for artist in box[artist_group]:
        artist.set(color='black', linewidth=1.0)
ax.set_xlabel('Period t')
ax.set_ylabel('Raw demand $d_t$')
ax.set_title('Demand distributions by period')
style_bar_axis(ax, Taxis)
finalize_figure(fig, 'figure1.pdf')

# Sensitivity
REOPT = True
BASE_SEED = 12345

demand_scenarios = [demand_rel_sigma_scenarios[label] for label in ['Low', 'Med', 'High']]
forecast_scenarios = [forecast_rel_alpha_scenarios[label] for label in ['Low', 'Med', 'High']]
labels_d = ['Low', 'Med', 'High']
labels_e = ['Low', 'Med', 'High']

nD = len(demand_scenarios)
nE = len(forecast_scenarios)

meanCostMat = np.zeros((nD, nE))
profileMat = np.zeros((nD, nE, params['Kmax'] + 1))
investPeriodMat = np.zeros((nD, nE, params['T']))
invCostMat = np.zeros((nD, nE))
sfCostMat = np.zeros((nD, nE))
scenario_capacity = [[None for _ in range(nE)] for _ in range(nD)]

Wdim = len(w_best)
wMat = np.zeros((nD, nE, Wdim))

def evalScenarioSimple(P, w_start, REOPT, bounds, seed):
    if REOPT:
        P_opt = P.copy()
        P_opt['iter'] = min(P['iter'], 1000)
        def obj_inner(w):
            mc, _ = simulate_cost(w, P_opt, seed=seed)
            return mc
        res_inner = minimize(obj_inner, w_start, bounds=bounds, method='COBYQA', options={'maxiter': 1000})
        w_used = res_inner.x
        mc, det_cell = simulate_cost(w_used, P, seed=seed)
    else:
        w_used = w_start
        mc, det_cell = simulate_cost(w_used, P, seed=seed)
    return mc, det_cell, w_used

for i in range(nD):
    for j in range(nE):
        P = params.copy()
        P['iter'] = 50000
        P['demand_rel_sigma'] = demand_scenarios[i].copy()
        P['forecast_rel_alpha'] = forecast_scenarios[j].copy()
        
        mc, det_cell, w_used = evalScenarioSimple(P, w_best, REOPT, bounds, BASE_SEED)
        meanCostMat[i, j] = mc
        wMat[i, j, :] = w_used
        scenario_capacity[i][j] = compute_capacity_paths(det_cell, P)
        
        avg_s_by_k = np.mean(det_cell['s'], axis=(0, 1))
        profileMat[i, j, :] = avg_s_by_k
        
        avg_s_by_t = np.mean(np.sum(det_cell['s'], axis=2), axis=0)
        investPeriodMat[i, j, :] = avg_s_by_t
        
        invCostMat[i, j] = np.mean(det_cell['invCost'])
        sfCostMat[i, j] = np.mean(det_cell['sfCost'])

# Expected total cost heatmap
fig, ax = plt.subplots()
sns.heatmap(meanCostMat, annot=True, fmt='.2f', xticklabels=labels_e, yticklabels=labels_d, cmap='Blues', ax=ax)
ax.set_title('Expected Total Cost')
ax.set_xlabel('Forecast Uncertainty Scenario')
ax.set_ylabel('Demand Uncertainty Scenario')
finalize_figure(fig, 'figure3.pdf')

# Figure 4: average total investment by period
yMax2 = np.max(investPeriodMat) * 1.1
fig, axes = plt.subplots(nD, nE, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle('Capacity orders placed by decision period')
for i in range(nD):
    for j in range(nE):
        ax = axes[i, j]
        ax.bar(Taxis, investPeriodMat[i, j, :], width=0.8, **BAR_STYLE)
        style_bar_axis(ax, Taxis)
        ax.set_xlim([0.5, params['T'] + 0.5])
        ax.set_ylim([0, yMax2])
        ax.set_title(f'Demand {labels_d[i]} | Forecast {labels_e[j]}')
        if i == nD - 1:
            ax.set_xlabel('Period t')
        if j == 0:
            ax.set_ylabel('Avg orders placed')
finalize_figure(fig, 'figure4.pdf', rect=[0, 0, 1, 0.97])

# Figure 5: investment profile by lead time
yMax = np.max(profileMat) * 1.1
Kax = active_leads
fig, axes = plt.subplots(nD, nE, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle('Avg investment profile by lead time across uncertainty scenarios')
for i in range(nD):
    for j in range(nE):
        ax = axes[i, j]
        ax.bar(Kax, profileMat[i, j, active_leads], width=0.8, **BAR_STYLE)
        style_bar_axis(ax, Kax)
        ax.set_ylim([0, yMax])
        ax.set_title(f'Demand {labels_d[i]} | Forecast {labels_e[j]}')
        if i == nD - 1:
            ax.set_xlabel('Lead time k')
        if j == 0:
            ax.set_ylabel('Avg s_{t,k}')
finalize_figure(fig, 'figure5.pdf', rect=[0, 0, 1, 0.97])

# Absolute costs with % in parentheses
den = invCostMat + sfCostMat
shareINV = np.divide(100 * invCostMat, den, out=np.zeros_like(invCostMat), where=den > 0)
shareSF = np.divide(100 * sfCostMat, den, out=np.zeros_like(sfCostMat), where=den > 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Costs: absolute with % share')
sns.heatmap(invCostMat, annot=np.array([[f'{invCostMat[i,j]:.2f} ({shareINV[i,j]:.1f}%)' for j in range(nE)] for i in range(nD)]), fmt='', xticklabels=labels_e, yticklabels=labels_d, cmap='Blues', ax=axes[0])
axes[0].set_title('Investment Cost')
axes[0].set_xlabel('Forecast Uncertainty Scenario')
axes[0].set_ylabel('Demand Uncertainty Scenario')

sns.heatmap(sfCostMat, annot=np.array([[f'{sfCostMat[i,j]:.2f} ({shareSF[i,j]:.1f}%)' for j in range(nE)] for i in range(nD)]), fmt='', xticklabels=labels_e, yticklabels=labels_d, cmap='Blues', ax=axes[1])
axes[1].set_title('Shortfall Cost')
axes[1].set_xlabel('Forecast Uncertainty Scenario')
axes[1].set_ylabel('Demand Uncertainty Scenario')
finalize_figure(fig, 'figure7.pdf', rect=[0, 0, 1, 0.95])

# Cumulative Demand vs Installed Capacity
fig, axes = plt.subplots(nD, nE, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle('Cumulative demand vs cumulative capacity available')
cum_bar_width = 0.36
for i in range(nD):
    for j in range(nE):
        cap_paths = scenario_capacity[i][j]
        cumDemand_all = cap_paths['cum_demand']
        cumCapacity_all = cap_paths['cum_capacity']

        cumDemand_mean = np.mean(cumDemand_all, axis=0)
        cumDemand_std = np.std(cumDemand_all, axis=0, ddof=0)
        cumCapacity_mean = np.mean(cumCapacity_all, axis=0)
        cumCapacity_std = np.std(cumCapacity_all, axis=0, ddof=0)

        ax = axes[i, j]
        ax.bar(
            Taxis - cum_bar_width / 2,
            cumDemand_mean,
            width=cum_bar_width,
            color='tab:red',
            alpha=0.85,
            label='Cumulative demand',
            **BAR_STYLE,
        )
        ax.bar(
            Taxis + cum_bar_width / 2,
            cumCapacity_mean,
            width=cum_bar_width,
            color='tab:blue',
            alpha=0.85,
            label='Cumulative capacity available',
            **BAR_STYLE,
        )
        ax.errorbar(Taxis - cum_bar_width / 2, cumDemand_mean, yerr=cumDemand_std, fmt='none', ecolor='k', capsize=3)
        ax.errorbar(Taxis + cum_bar_width / 2, cumCapacity_mean, yerr=cumCapacity_std, fmt='none', ecolor='k', capsize=3)
        style_bar_axis(ax, Taxis)
        ax.set_title(f'Demand {labels_d[i]} | Forecast {labels_e[j]}')
        if i == nD - 1:
            ax.set_xlabel('Period t')
        if j == 0:
            ax.set_ylabel('Cumulative units')
        if i == 0 and j == 0:
            ax.legend(loc='upper left', fontsize=8)
finalize_figure(fig, 'figure6.pdf', rect=[0, 0, 1, 0.97])

# Weights Table (simple)
Wdim_exact = params['Kmax'] + 2
for i in range(nD):
    for j in range(nE):
        w = wMat[i, j, :Wdim_exact]
        weights_str = ', '.join([f'{x:.5f}' for x in w])
        print(f"Demand uncertainty: {labels_d[i].lower()}, Forecast uncertainty: {labels_e[j].lower()} {{{weights_str}}}")

# c_k figure
ck_active = np.array(params['c_k'])[active_leads]
y_pad = 0.08 * (ck_active.max() - ck_active.min())
fig, ax = plt.subplots()
ax.bar(active_leads, ck_active, color='steelblue', width=0.55, **BAR_STYLE)
ax.set_xlabel('Lead time k')
ax.set_ylabel('Unit cost')
ax.set_title('Unit costs of capacity by lead time')
style_bar_axis(ax, active_leads)
ax.set_ylim([ck_active.min() - y_pad, ck_active.max() + y_pad])
finalize_figure(fig, 'figure2.pdf')
