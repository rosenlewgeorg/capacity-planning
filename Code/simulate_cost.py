import os
import random

import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from matplotlib.colors import LinearSegmentedColormap


# Parameters
ITERATIONS = 1000
SHORTFALL_COST = 2.0  # c^s
UNCERTAINTY_LEVELS = {
    "none": 0.0,
    "low": 0.25,
    "med": 0.5,
    "high": 0.75,
}
SEED = 67

levels = list(UNCERTAINTY_LEVELS.keys())
total_cost_matrix = [[0.0 for _ in levels] for _ in levels]
estimate_samples_matrix = [[None for _ in levels] for _ in levels]

os.makedirs("Figures", exist_ok=True)


# Print a simple table header.
header = (
    f"{'Demand':<6} {'Estimate':<8} "
    f"{'Exp inv cost':>14} {'Exp short cost':>16} {'Exp total cost':>16}"
)
print(header)
print("-" * len(header))


# Go through every demand-uncertainty and estimate-uncertainty combination.
for demand_label, demand_sigma in UNCERTAINTY_LEVELS.items():
    for estimate_label, estimate_sigma in UNCERTAINTY_LEVELS.items():
        rng = random.Random(f"{SEED}:{demand_label}:{estimate_label}")

        # Step 1: simulate D and E and store them in lists.
        D = []
        E = []

        for _ in range(ITERATIONS):
            if demand_sigma == 0:
                demand = 1.0
            else:
                demand_mu = -0.5 * demand_sigma * demand_sigma
                demand = rng.lognormvariate(demand_mu, demand_sigma)

            if estimate_sigma == 0:
                estimate_noise = 1.0
            else:
                estimate_mu = -0.5 * estimate_sigma * estimate_sigma
                estimate_noise = rng.lognormvariate(estimate_mu, estimate_sigma)

            estimate = demand * estimate_noise

            D.append(demand)
            E.append(estimate)

        # Step 2: optimize S by minimizing the expected total cost formula.
        result = minimize_scalar(
            lambda S: S + SHORTFALL_COST * sum(max(e - S, 0.0) for e in E) / ITERATIONS,
            bounds=(0.0, max(E)),
            method="bounded",
        )

        S = result.x

        # Step 3: compute the final expected costs for this scenario.
        expected_investment_cost = S
        expected_shortfall_cost = SHORTFALL_COST * sum(max(e - S, 0.0) for e in E) / ITERATIONS
        expected_total_cost = expected_investment_cost + expected_shortfall_cost
        row_index = levels.index(demand_label)
        col_index = levels.index(estimate_label)
        total_cost_matrix[row_index][col_index] = expected_total_cost
        estimate_samples_matrix[row_index][col_index] = E.copy()

        # Step 4: print the final results.
        print(
            f"{demand_label:<6} "
            f"{estimate_label:<8} "
            f"{expected_investment_cost:>14.4f} "
            f"{expected_shortfall_cost:>16.4f} "
            f"{expected_total_cost:>16.4f}"
        )


# Step 5: create a matrix plot for expected total cost.
cmap = LinearSegmentedColormap.from_list("white_to_blue", ["#ffffff", "#6baed6"])

fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(total_cost_matrix, origin="lower", cmap=cmap)

ax.set_xticks(range(len(levels)))
ax.set_yticks(range(len(levels)))
ax.set_xticklabels(levels)
ax.set_yticklabels(levels)
ax.set_xlabel("Estimate uncertainty")
ax.set_ylabel("Demand uncertainty")
ax.set_title("Expected total cost by uncertainty scenario")

for row_index in range(len(levels)):
    for col_index in range(len(levels)):
        value = total_cost_matrix[row_index][col_index]
        ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center")

colorbar = plt.colorbar(image, ax=ax)
colorbar.set_label("Expected total cost")

plt.tight_layout()
plt.savefig("Figures/expected_total_cost_matrix.png", dpi=300, bbox_inches="tight")
plt.close()


# Step 6: create a matrix of histograms for the simulated E values.
all_E_values = []
for row in estimate_samples_matrix:
    for samples in row:
        all_E_values.extend(samples)

global_max_E = max(all_E_values)

fig, axes = plt.subplots(len(levels), len(levels), figsize=(12, 10), sharex=True, sharey=True)

for row_index in range(len(levels)):
    for col_index in range(len(levels)):
        plot_row = len(levels) - 1 - row_index
        ax = axes[plot_row][col_index]
        E_values = estimate_samples_matrix[row_index][col_index]

        ax.hist(
            E_values,
            bins=25,
            range=(0.0, global_max_E),
            color="#9ecae1",
            edgecolor="#3182bd",
        )
        ax.set_title(f"{levels[row_index]}/{levels[col_index]}", fontsize=10)

        if plot_row == len(levels) - 1:
            ax.set_xlabel("E")
        if col_index == 0:
            ax.set_ylabel("Count")

fig.suptitle("Distribution of E by uncertainty scenario", fontsize=14)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig("Figures/estimate_histogram_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
