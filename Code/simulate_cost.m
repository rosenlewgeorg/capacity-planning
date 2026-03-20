function [meanCost, details] = simulate_cost(w, P, seed)
%SIMULATE_COST  Monte Carlo simulation returning mean total cost.
%   w    : weight vector [w0, w1, ..., w_{Kmax}]  
%   P    : struct with all parameters
%   seed : optional RNG seed for reproducibility

    if nargin == 3 && ~isempty(seed);
rng(seed);
end

T = P.T;
c_s = P.c_s;
c_k = P.c_k;
iter = P.iter;
Kmax = P.Kmax;
delta = P.delta;
mu = P.mu;
sigma = P.sigma;
increase = P.increase;
sigmaeps = P.sigmaeps;
incr = P.incr;
estmu = P.estmu;

assert(numel(w) >= Kmax + 2, 'w too short');
assert(numel(c_k) >= Kmax + 1, 'c_k too short');

% Demand sigmas = sigma * increase.^ (0 : T - 1);
demands = exp(mu + randn(iter, T).*sigmas);

Dtot = cumsum(demands, 2);

        % Forecasts sigmaepses = [ 0, sigmaeps *incr.^ (1
                                                        : Kmax) ];
    E = NaN(iter, T, Kmax + 1);
    for
      k = 0 : Kmax if k > T, break, end t_idx = 1 : (T - k);

    if k
      == 0 E( :, t_idx, 1) = Dtot( :, t_idx);
    else
      s = sigmaepses(k + 1);
    mu_k = -0.5 * s ^ 2;
    % ensures E[eps] = 1 eps_draws = lognrnd(mu_k, s, [ iter, numel(t_idx) ]);
    E( :, t_idx, k + 1) = Dtot( :, t_idx + k).*eps_draws;
    end

            end

        % Decisions s_{t, k} and inventory balance s = zeros(iter, T, Kmax + 1);

    % decide s
    for t = 1:T
        for k = 0:Kmax
            if t+k > T, continue;
    end eps_slice = squeeze(E( :, t, 1 : k + 1));
    if k
      == 0, eps_slice = eps_slice( :);
    end
            s(:, t, k+1) = w(1) + eps_slice * w(2:k+2)';
            s(:, t, k+1) = max(s(:, t, k+1), 0);
    end end

        % inventory after serving demand at t I = zeros(iter, T);
    shortfall = zeros(iter, T);

    for
      t = 1 : T % arrivals that materialize at t arrivals = zeros(iter, 1);
        for
          tau = 1 : t kk = t - tau;
        arrivals = arrivals + s( :, tau, kk + 1);
        end

            if t == 1 I_prev = zeros(iter, 1);
        else I_prev = I( :, t - 1);
        end

            available = (1 - delta) * I_prev + arrivals;
        % before demand at t shortfall( :, t) = max(Dtot( :, t) - available, 0);
        served = Dtot( :, t) - shortfall( :, t);
        I( :, t) = available - served;
        % carry to next period end

            % Costs invCost = sum(sum(s.*reshape(c_k(1
                                                     : Kmax + 1),
                                                 1, 1, []),
                                      3),
                                  2);
        sfCost = c_s * sum(shortfall, 2);
        totalCost = invCost + sfCost;
        meanCost = mean(totalCost);

        % Fill rate calculation totalDemand = sum(Dtot, 2);
        totalShort = sum(shortfall, 2);
        fillRatePath = 1 - totalShort./ totalDemand;

        % Overall expected fill rate fillRateOverall = mean(fillRatePath);

        % Period - wise average fill rate fillRatePerPeriod =
            1 - sum(shortfall, 1)./ sum(Dtot);

        % Probability of any stockout in the path probStockout =
            mean(any(shortfall > 1e-12, 2));
        cycleService = 1 - probStockout;

        details.totalCost = totalCost;
        details.invCost = invCost;
        details.sfCost = sfCost;
        details.p95 = prctile(totalCost, 95);
        details.I = I;
        % new : on - hand after demand details.S_tot = I;                   % kept for backward compat (now equals I)
    details.Dtot      = Dtot;
        details.s = s;
        details.fillRateOverall = fillRateOverall;
        details.fillRatePerPeriod = fillRatePerPeriod;
        details.fillRatePath = fillRatePath;
        details.cycleService = cycleService;

        end
