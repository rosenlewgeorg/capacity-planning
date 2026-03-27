import random

from scipy.optimize import minimize_scalar


# Parameters you may want to change
ITERATIONS = 1000
SHORTFALL_COST = 2.0  # c^s
UNCERTAINTY_LEVELS = {
    "none": 0.0,
    "low": 0.25,
    "med": 0.5,
    "high": 0.75,
}
SEED = 67


# Draw one random lognormal value for a chosen uncertainty level.
def sample_lognormal(rng, sigma):
    if sigma == 0:
        return 1.0
    mu = -0.5 * sigma * sigma
    return rng.lognormvariate(mu, sigma)


# Simulate the estimate values used in one scenario.
def simulate_estimates(rng, sigma_demand, sigma_estimate):
    estimates = []
    for _ in range(ITERATIONS):
        demand = sample_lognormal(rng, sigma_demand)
        estimate = demand * sample_lognormal(rng, sigma_estimate)
        estimates.append(estimate)
    return estimates


# Compute expected investment, shortfall, and total cost for one S value.
def expected_costs(investment, estimates):
    expected_investment_cost = investment
    expected_shortfall_cost = SHORTFALL_COST * (
        sum(max(estimate - investment, 0.0) for estimate in estimates) / len(estimates)
    )
    expected_total_cost = expected_investment_cost + expected_shortfall_cost
    return expected_investment_cost, expected_shortfall_cost, expected_total_cost


# Solve one uncertainty scenario by finding the best investment level.
def solve_scenario(demand_label, demand_sigma, estimate_label, estimate_sigma):
    rng = random.Random(f"{SEED}:{demand_label}:{estimate_label}")
    estimates = simulate_estimates(rng, demand_sigma, estimate_sigma)

    result = minimize_scalar(
        lambda investment: expected_costs(investment, estimates)[2],
        bounds=(0.0, max(estimates)),
        method="bounded",
    )

    expected_investment_cost, expected_shortfall_cost, expected_total_cost = expected_costs(
        result.x,
        estimates,
    )

    return {
        "demand_level": demand_label,
        "estimate_level": estimate_label,
        "expected_investment_cost": expected_investment_cost,
        "expected_shortfall_cost": expected_shortfall_cost,
        "expected_total_cost": expected_total_cost,
    }


# Print the results in a simple table.
def print_results(results):
    header = (
        f"{'Demand':<6} {'Estimate':<8} "
        f"{'Exp inv cost':>14} {'Exp short cost':>16} {'Exp total cost':>16}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['demand_level']:<6} "
            f"{row['estimate_level']:<8} "
            f"{row['expected_investment_cost']:>14.4f} "
            f"{row['expected_shortfall_cost']:>16.4f} "
            f"{row['expected_total_cost']:>16.4f}"
        )


# Run all uncertainty combinations and show the final results.
def main():
    results = []

    for demand_label, demand_sigma in UNCERTAINTY_LEVELS.items():
        for estimate_label, estimate_sigma in UNCERTAINTY_LEVELS.items():
            results.append(
                solve_scenario(
                    demand_label,
                    demand_sigma,
                    estimate_label,
                    estimate_sigma,
                )
            )

    print_results(results)


if __name__ == "__main__":
    main()
