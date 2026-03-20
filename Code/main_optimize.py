import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize
from simulate_cost import simulate_cost

# 1. Parameters
params = {
    'T': 3,
    'c_s': 1,
    'iter': 10000,
    'Kmax': 2,
    # Use a simple 10% per-period discount rate to price future installations.
    'r': 0.1,
    'delta': 0,
    'mu': 1,
    'sigma': 1,
    'increase': 1.1,
    'sigmaeps': 1,
    'incr': 1.1,
    'estmu': 0
}

# With a 3-period horizon, only lead times k = 0, 1, 2 can arrive in time.
params['c_k'] = (1 / (1 + params['r']) ** np.arange(params['Kmax'] + 1)).tolist()

# 2. Initial weights
# Start from a smooth, short-horizon policy: near-term signals matter most,
# while preserving some sensitivity to the longer lead times that still fit
# inside the 3-period model.
w0 = np.array([
    0.12,   # baseline investment
    0.35,   # e_{t,0}
    0.10,   # e_{t,1}
    0.06    # e_{t,2}
])

if len(params['c_k']) != params['Kmax'] + 1:
    raise ValueError("c_k must have length Kmax + 1.")

if len(w0) != params['Kmax'] + 2:
    raise ValueError("w0 must have length Kmax + 2.")

OPT_SEED = 12345
OPT_METHOD = 'Powell'
FIGURES_DIR = Path(__file__).resolve().parents[1] / 'Figures'
FIGURES_DIR.mkdir(exist_ok=True)
HEATMAP_CMAP = 'Blues'

print(f'Saving figures to {FIGURES_DIR}')

# 3. Objective handle
def obj(w):
    mc, _ = simulate_cost(w, params, seed=OPT_SEED)
    return mc

# 4. Optimise
bounds = [(0, None) for _ in range(len(w0))]
print('Optimising...')
# Powell is a derivative-free method that supports bounds in the SciPy version
# available in this environment.
res = minimize(obj, w0, bounds=bounds, method=OPT_METHOD, options={'disp': True, 'maxiter': 1000})
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

print(f'\n=== BASELINE (seed={BASE_SEED}) ===')
print(f'Expected total cost      : {MTot:.4f}')
print(f'  Investment cost    : {MInv:.4f}')
print(f'  Shortfall cost     : {MSF:.4f}')

print('Optimised weights (w_best):')
print(w_best)

def draw_bar_chart(ax, x_values, heights, width=0.8):
    x_values = np.asarray(x_values)
    ax.bar(
        x_values,
        heights,
        width=width,
        edgecolor='black',
        linewidth=0.8,
        zorder=3,
    )
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', zorder=0)
    ax.set_xticks(x_values)
    if x_values.size > 0:
        ax.set_xlim([np.min(x_values) - 0.5, np.max(x_values) + 0.5])

def save_figure(fig, filename):
    fig.savefig(FIGURES_DIR / filename, format='pdf', bbox_inches='tight')

# Investment profile by lead time
avg_s_by_k = np.mean(det_base['s'], axis=(0, 1))
fig, ax = plt.subplots()
draw_bar_chart(ax, np.arange(params['Kmax'] + 1), avg_s_by_k)
ax.set_xlabel('Lead time k')
ax.set_ylabel('Avg s_{t,k} per period')
ax.set_title('Baseline investment profile by lead time')
save_figure(fig, 'baseline_investment_profile_by_lead_time.pdf')
plt.show(block=False)

# Bar chart: mean raw demand per period
rng = np.random.default_rng(BASE_SEED)
sigmas = params['sigma'] * np.sqrt(params['increase'] ** np.arange(params['T']))
demands = np.exp(params['mu'] + rng.standard_normal((params['iter'], params['T'])) * sigmas)
mu_raw = np.mean(demands, axis=0)
sd_raw = np.std(demands, axis=0, ddof=0)
Taxis = np.arange(1, params['T'] + 1)

fig, ax = plt.subplots()
draw_bar_chart(ax, Taxis, mu_raw)
ax.errorbar(Taxis, mu_raw, yerr=[np.zeros_like(sd_raw), sd_raw], fmt='none', ecolor='k', capsize=0, zorder=4)
ax.set_xlabel('Period t')
ax.set_ylabel('Demand (E[d_t])')
ax.set_title('Mean raw demand by period with standard deviation')
save_figure(fig, 'mean_raw_demand_by_period.pdf')
plt.show(block=False)

# Sensitivity
REOPT = True
BASE_SEED = 12345

sigma_factors = [0.7, 1.0, 1.3]
sigmaeps_factors = [0.7, 1.0, 1.3]
labels_d = ['Low', 'Med', 'High']
labels_e = ['Low', 'Med', 'High']

nD = len(sigma_factors)
nE = len(sigmaeps_factors)

meanCostMat = np.zeros((nD, nE))
profileMat = np.zeros((nD, nE, params['Kmax'] + 1))
investPeriodMat = np.zeros((nD, nE, params['T']))
invCostMat = np.zeros((nD, nE))
sfCostMat = np.zeros((nD, nE))

Wdim = len(w_best)
wMat = np.zeros((nD, nE, Wdim))

def evalScenarioSimple(P, w_start, REOPT, bounds, seed):
    if REOPT:
        P_opt = P.copy()
        P_opt['iter'] = min(P['iter'], 1000)
        def obj_inner(w):
            mc, _ = simulate_cost(w, P_opt, seed=seed)
            return mc
        res_inner = minimize(obj_inner, w_start, bounds=bounds, method=OPT_METHOD, options={'maxiter': 1000})
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
        P['sigma'] = params['sigma'] * np.sqrt(sigma_factors[i])
        P['sigmaeps'] = params['sigmaeps'] * np.sqrt(sigmaeps_factors[j])
        
        mc, det_cell, w_used = evalScenarioSimple(P, w_best, REOPT, bounds, BASE_SEED)
        meanCostMat[i, j] = mc
        wMat[i, j, :] = w_used
        
        avg_s_by_k = np.mean(det_cell['s'], axis=(0, 1))
        profileMat[i, j, :] = avg_s_by_k
        
        avg_s_by_t = np.mean(np.sum(det_cell['s'], axis=2), axis=0)
        investPeriodMat[i, j, :] = avg_s_by_t
        
        invCostMat[i, j] = np.mean(det_cell['invCost'])
        sfCostMat[i, j] = np.mean(det_cell['sfCost'])

# Expected total cost heatmap
fig, ax = plt.subplots()
sns.heatmap(meanCostMat, annot=True, xticklabels=labels_e, yticklabels=labels_d, cmap=HEATMAP_CMAP, ax=ax)
ax.set_title('Expected Total Cost')
ax.set_xlabel('Forecast Uncertainty ($\\xi_{k}^2$)')
ax.set_ylabel('Demand Uncertainty ($\\sigma_t^2$)')
save_figure(fig, 'expected_total_cost_heatmap.pdf')
plt.show(block=False)

# 3x3 matrix of investment profiles
yMax = np.max(profileMat) * 1.1
Kax = np.arange(params['Kmax'] + 1)
fig, axes = plt.subplots(nD, nE, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle('Avg investment profile by lead time across uncertainty scenarios')
for i in range(nD):
    for j in range(nE):
        ax = axes[i, j]
        draw_bar_chart(ax, Kax, profileMat[i, j, :])
        ax.set_ylim([0, yMax])
        ax.set_title(f'Demand {labels_d[i]} | Forecast {labels_e[j]}')
        if i == nD - 1:
            ax.set_xlabel('Lead time k')
        if j == 0:
            ax.set_ylabel('Avg s_{t,k}')
plt.tight_layout()
save_figure(fig, 'investment_profile_by_lead_time_matrix.pdf')
plt.show(block=False)

# 3x3 matrix of average total investment by period
yMax2 = np.max(investPeriodMat) * 1.1
fig, axes = plt.subplots(nD, nE, figsize=(10, 8), sharex=True, sharey=True)
fig.suptitle('Moment of investment across uncertainty scenarios')
for i in range(nD):
    for j in range(nE):
        ax = axes[i, j]
        draw_bar_chart(ax, Taxis, investPeriodMat[i, j, :])
        ax.set_ylim([0, yMax2])
        ax.set_title(f'Demand {labels_d[i]} | Forecast {labels_e[j]}')
        if i == nD - 1:
            ax.set_xlabel('Period t')
        if j == 0:
            ax.set_ylabel('Avg $\\Sigma_k s_{t,k}$')
plt.tight_layout()
save_figure(fig, 'investment_moment_by_period_matrix.pdf')
plt.show(block=False)

# Absolute costs with % in parentheses
den = invCostMat + sfCostMat
shareINV = np.divide(100 * invCostMat, den, out=np.zeros_like(invCostMat), where=den > 0)
shareSF = np.divide(100 * sfCostMat, den, out=np.zeros_like(sfCostMat), where=den > 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Costs: absolute with % share')
sns.heatmap(invCostMat, annot=np.array([[f'{invCostMat[i,j]:.2f} ({shareINV[i,j]:.1f}%)' for j in range(nE)] for i in range(nD)]), fmt='', xticklabels=labels_e, yticklabels=labels_d, cmap='Blues', ax=axes[0])
axes[0].set_title('Investment Cost')
axes[0].set_xlabel('Forecast Uncertainty ($\\xi_{k}^2$)')
axes[0].set_ylabel('Demand Uncertainty ($\\sigma_t^2$)')

sns.heatmap(sfCostMat, annot=np.array([[f'{sfCostMat[i,j]:.2f} ({shareSF[i,j]:.1f}%)' for j in range(nE)] for i in range(nD)]), fmt='', xticklabels=labels_e, yticklabels=labels_d, cmap='Blues', ax=axes[1])
axes[1].set_title('Shortfall Cost')
axes[1].set_xlabel('Forecast Uncertainty ($\\xi_{k}^2$)')
axes[1].set_ylabel('Demand Uncertainty ($\\sigma_t^2$)')
plt.tight_layout()
save_figure(fig, 'cost_share_heatmaps.pdf')
plt.show(block=False)

# Cumulative Demand vs Capacity
fig, axes = plt.subplots(nD, nE, figsize=(12, 10))
fig.suptitle('Cumulative Demand vs Capacity (bubble size = standard deviation of D^{tot})')
lightBlue = (0.6, 0.85, 1.0)
for i in range(nD):
    for j in range(nE):
        P = params.copy()
        P['iter'] = 50000
        P['sigma'] = params['sigma'] * np.sqrt(sigma_factors[i])
        P['sigmaeps'] = params['sigmaeps'] * np.sqrt(sigmaeps_factors[j])
        
        _, det_cell, _ = evalScenarioSimple(P, w_best, REOPT, bounds, BASE_SEED)
        
        cumDemand_all = np.cumsum(det_cell['Dtot'], axis=1)
        cumCapacity_all = np.cumsum(np.sum(det_cell['s'], axis=2), axis=1)
        
        cumDemand_mean = np.mean(cumDemand_all, axis=0)
        cumCapacity_mean = np.mean(cumCapacity_all, axis=0)
        cumDemand_std = np.std(cumDemand_all, axis=0, ddof=0)
        
        max_std = np.max(cumDemand_std)
        bubbleSize = np.zeros_like(cumDemand_std) if max_std == 0 else 1800 * (cumDemand_std / max_std)
        
        ax = axes[i, j]
        ax.scatter(cumCapacity_mean, cumDemand_mean, s=bubbleSize, color=lightBlue, edgecolor='k', alpha=0.7)
        for t in range(params['T']):
            ax.text(cumCapacity_mean[t], cumDemand_mean[t], str(t+1), va='center', ha='center', color='k', fontweight='bold')
            
        lineMin = min(np.min(cumCapacity_mean), np.min(cumDemand_mean))
        lineMax = max(np.max(cumCapacity_mean) + 50, np.max(cumDemand_mean) + 50)
        ax.plot([lineMin, lineMax], [lineMin, lineMax], '--k', linewidth=1.0)
        
        ax.grid(True)
        ax.set_title(f'Demand {labels_d[i]} | Forecast {labels_e[j]}')
        if i == nD - 1:
            ax.set_xlabel('S^{tot}')
        if j == 0:
            ax.set_ylabel('D^{tot}')
plt.tight_layout()
save_figure(fig, 'cumulative_demand_vs_capacity.pdf')
plt.show(block=False)

# Weights Table (simple)
Wdim_exact = params['Kmax'] + 2
for i in range(nD):
    for j in range(nE):
        w = wMat[i, j, :Wdim_exact]
        weights_str = ', '.join([f'{x:.5f}' for x in w])
        print(f"Demand uncertainty: {labels_d[i].lower()}, Forecast uncertainty: {labels_e[j].lower()} {{{weights_str}}}")

# c_k figure
fig, ax = plt.subplots()
draw_bar_chart(ax, Kax, params['c_k'])
ax.set_xlabel('Lead time k')
ax.set_ylabel('Cost')
ax.set_title('Unit costs of capacity installed k periods later c_k')
ax.set_ylim([0, max(params['c_k']) * 1.05])
save_figure(fig, 'unit_costs_by_lead_time.pdf')
plt.show()
