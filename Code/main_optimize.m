% 1. Parameters
params.T = 10;
params.c_s = 1;
params.iter = 10000;
params.Kmax = 9;
params.r = 0.05;
params.c_k = 1 ./ (1 + params.r).^(0:params.Kmax);
params.delta = 0;
params.mu = 1;
params.sigma = 1;
params.increase = 1.1;
params.sigmaeps = 1;
params.incr = 1.1;
params.estmu = 0;

% 2. Initial weights w0 =
    [0.177734375000000 0.862377929687500 0.00371093750000000 0.00761528015136719 0.0119873046875000 0.128125000000001 0.0312500000000000 0.000122070312500000 0.0156250000000000 0.000488281250000000 0];
% length Kmax + 1

                    % 3. Objective handle obj = @(w) simulate_cost(w, params);

% 4. Optimise lb = zeros(size(w0));
% no negatives ub = [];
% or set upper bound wanted opts =
    optimoptions('patternsearch', 'Display', 'iter', 'UseParallel', false);

[ w_best, fval ] = patternsearch(obj, w0, [], [], [], [], lb, ub, [], opts);

fprintf('Best mean cost: %.2f\n', fval);

format long disp('Best weights:');
disp(w_best);

[ ~, det ] = simulate_cost(w_best, params, 123);

% -- -- -- -- -- -- -- -- --Graphs &Tables-- -- -- -- -- -- -- -- --

    % Baseline performance BASE_SEED = 12345;  % for reproducibility
[meanCost, det_base] = simulate_cost(w_best, params, BASE_SEED);

MTot = meanCost;
% mean total cost MInv = mean(det_base.invCost);
% mean investment cost MSF = mean(det_base.sfCost);
% mean shortfall cost P95 = det_base.p95;
% 95th percentile of total cost

        fprintf('\n=== BASELINE (seed=%d) ===\n', BASE_SEED);
fprintf('Expected total cost      : %.4f\n', MTot);
fprintf('  Investment cost    : %.4f\n', MInv);
fprintf('  Shortfall cost     : %.4f\n', MSF);

% Optimised weights(optional print) disp('Optimised weights (w_best):');
disp(w_best(:)');

% Investment profile by lead time (average per period & path)
avg_s_by_k = squeeze(mean(mean(det_base.s,1),2));  % (Kmax+1)x1
figure;
bar(0:params.Kmax, avg_s_by_k, 'LineWidth', 1);
xlabel('Lead time k'); ylabel('Avg s_{t,k} per period');
title('Baseline investment profile by lead time'); grid on;


% Bar chart: mean raw demand per period with std errors (no AR carryover)
rng(BASE_SEED);
sigmas  = params.sigma * params.increase.^(0:params.T-1);
demands = exp(params.mu + randn(params.iter, params.T) .* sigmas);

mu_raw = mean(demands, 1);
sd_raw = std(demands, 0, 1);
Taxis  = 1:params.T;

figure;
bar(Taxis, mu_raw, 'LineWidth', 1);
hold on;
errLow  = zeros(size(sd_raw));   
errHigh = sd_raw;                

errorbar(Taxis, mu_raw, errLow, errHigh, 'k', 'LineStyle','none', 'LineWidth', 1.2);
hold off;

xlim([0.5, params.T + 0.5]);
xlabel('Period t');
ylabel('Demand (E[d_t])');
title('Mean raw demand by period with standard deviation');
grid on;


% Sensitivity

REOPT = true;                 % true for reoptimization
BASE_SEED = 12345;             % common random numbers (same seed everywhere)

% Quiet options
opts_reopt = optimoptions('patternsearch','Display','off','UseParallel',false);

% A) 2-way heatmap: sigma × sigmaeps
sigma_factors    = [0.7 1.0 1.3];
sigmaeps_factors = [0.7 1.0 1.3];
labels_d = {'Low', 'Med', 'High'};
labels_e = {'Low', 'Med', 'High'};

nD = numel(sigma_factors);
nE = numel(sigmaeps_factors);

meanCostMat = zeros(nD, nE);

profileMat = zeros(nD, nE, params.Kmax + 1);
% (demand x Forecast x k)investPeriodMat = zeros(nD, nE, params.T);
% (demand x Forecast x t)invCostMat = zeros(nD, nE);
% mean investment cost per scenario sfCostMat = zeros(nD, nE);
% mean shortfall cost per scenario

        Wdim = numel(w_best);
% number of weights actually optimized wMat = zeros(nD, nE, Wdim);

for i = 1:nD
    for j = 1:nE
        P = params;
% start from baseline each time P.iter = 50000;
% accuracy vs time P.sigma = params.sigma * sigma_factors(i);
P.sigmaeps = params.sigmaeps * sigmaeps_factors(j);

[ mc, det_cell, w_used ] =
    evalScenarioSimple(P, w_best, REOPT, opts_reopt, lb, ub, BASE_SEED);
meanCostMat(i, j) = mc;

wMat(i, j, :) = w_used;

        % average investment by lead time k (same metric you used for the baseline)
        avg_s_by_k = squeeze(mean(mean(det_cell.s,1),2));
        % (Kmax + 1)×1 profileMat(i, j, :) = avg_s_by_k;

        % total invested at decision time t : sum over all lead times k,
            then avg over paths avg_s_by_t =
                squeeze(mean(sum(det_cell.s, 3), 1));
        % T×1 investPeriodMat(i, j, :) = avg_s_by_t;

        % These are already totals over T in simulate_cost;
        take expectation over paths invCostMat(i, j) = mean(det_cell.invCost);
        sfCostMat(i, j) = mean(det_cell.sfCost);

        end

                end

            % Expected total cost figure;
        h1 = heatmap(labels_e, labels_d, meanCostMat);
        h1.Title = 'Expected Total Cost';
        h1.XLabel = 'Forecast Uncertainty (\xi_{k}^2)';
        h1.YLabel = 'Demand Uncertainty (\sigma_t^2)';

        % 3×3 matrix of investment profiles Kax = 0 : params.Kmax;
        yMax = max(profileMat, [], 'all');  % common y-limit for comparability

figure;
        tl =
            tiledlayout(nD, nE, 'TileSpacing', 'compact', 'Padding', 'compact');
        sgtitle(
            'Avg investment profile by lead time across uncertainty scenarios');

for i = 1:nD
    for j = 1:nE
        nexttile;
bar(Kax, squeeze(profileMat(i, j, :)), 'LineWidth', 0.5);
grid on;
ylim([ 0, 1.1 * yMax ]);
xlim([ Kax(1) - 0.5, Kax(end) + 0.5 ]);
title(sprintf('Demand %s | Forecast %s', labels_d{i}, labels_e{j}));
if i
  == nD, xlabel('Lead time k');
end if j == 1, ylabel('Avg s_{t,k}');
end end end

    % 3×3 matrix of average total investment by period Taxis = 1 : params.T;
yMax2 = max(investPeriodMat, [], 'all');  % common y-limit for visual comparability

figure;
tl2 = tiledlayout(nD, nE, 'TileSpacing', 'compact', 'Padding', 'compact');
sgtitle('Moment of investment across uncertainty scenarios');

for i = 1:nD
    for j = 1:nE
        nexttile;
bar(Taxis, squeeze(investPeriodMat(i, j, :)), 'LineWidth', 0.5);
grid on;
xlim([ 0.5, params.T + 0.5 ]);
ylim([ 0, 1.1 * yMax2 ]);
title(sprintf('Demand %s | Forecast %s', labels_d{i}, labels_e{j}));
if i
  == nD, xlabel('Period t');
end if j == 1, ylabel('Avg \Sigma_k s_{t,k}');
end end end

    % Absolute costs with % in parentheses den = invCostMat + sfCostMat;
shareINV = 100 * (invCostMat./ den);
shareSF = 100 * (sfCostMat./ den);

% White->Navy colormap N = 256;
navy = [ 8, 116, 196 ] / 255;
wbnavy = [linspace(1,navy(1),N)', linspace(1,navy(2),N)', linspace(1,navy(3),N)'];

figure('Name','Costs: absolute with % share (white -> navy)');
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

% Left: Investment cost
ax1 = nexttile;
imagesc(invCostMat); set(ax1,'YDir','normal'); colorbar;
colormap(ax1, wbnavy);
title('Investment Cost');
set(ax1,'XTick',1:numel(labels_e),'XTickLabel',labels_e, ...
         'YTick',1:numel(labels_d),'YTickLabel',labels_d);
xlabel('Forecast Uncertainty (\xi_{k}^{2})');
ylabel('Demand Uncertainty (\sigma_{t}^{2})');

lims1 = caxis(ax1);
for i = 1:size(invCostMat,1)
    for j = 1:size(invCostMat,2)
        v = invCostMat(i,j);
        frac = (v - lims1(1)) / diff(lims1);  % where this cell sits in its own scale
        tcol = 'k'; if frac > 0.6, tcol = 'w'; end
        text(j,i, sprintf('%.2f (%.1f%%)', v, shareINV(i,j)), ...
            'HorizontalAlignment','center','FontWeight','bold','Color',tcol);
    end
end

% Right: Shortfall cost
ax2 = nexttile;
imagesc(sfCostMat); set(ax2,'YDir','normal'); colorbar;
colormap(ax2, wbnavy);
title('Shortfall Cost');
set(ax2,'XTick',1:numel(labels_e),'XTickLabel',labels_e, ...
         'YTick',1:numel(labels_d),'YTickLabel',labels_d);
xlabel('Forecast Uncertainty (\xi_{k}^{2})');
ylabel('Demand Uncertainty (\sigma_{t}^{2})');

lims2 = caxis(ax2);
for i = 1:size(sfCostMat,1)
    for j = 1:size(sfCostMat,2)
        v = sfCostMat(i,j);
        frac = (v - lims2(1)) / diff(lims2);
        tcol = 'k'; if frac > 0.6, tcol = 'w'; end
        text(j,i, sprintf('%.2f (%.1f%%)', v, shareSF(i,j)), ...
            'HorizontalAlignment','center','FontWeight','bold','Color',tcol);
    end
end



lightBlue = [0.6 0.85 1];  % pastel blue

figure;
tl = tiledlayout(nD, nE, 'TileSpacing','compact','Padding','compact');
sgtitle('Cumulative Demand vs Capacity (bubble size = standard deviation of D^{tot})');

for i = 1:nD
    for j = 1:nE
        % Recompute scenario results (det_cell) for plotting
        P = params;
        P.iter    = 50000;
        P.sigma   = params.sigma    * sigma_factors(i);
        P.sigmaeps= params.sigmaeps * sigmaeps_factors(j);

        [~, det_cell] = evalScenarioSimple(P, w_best, REOPT, opts_reopt, lb, ub, BASE_SEED);

        % cumulative demand & capacity
        cumDemand_all   = cumsum(det_cell.Dtot, 2);           % iter × T
        cumCapacity_all = cumsum(sum(det_cell.s,3), 2);       % iter × T

        cumDemand_mean  = mean(cumDemand_all, 1);             % 1 × T
        cumCapacity_mean= mean(cumCapacity_all, 1);           % 1 × T
        cumDemand_std   = std(cumDemand_all, 0, 1);           % 1 × T

        % bubble size
        bubbleSize = 1800 * (cumDemand_std ./ max(cumDemand_std));

        % subplot
        nexttile;
        scatter(cumCapacity_mean, cumDemand_mean, bubbleSize, 'o', ...
            'MarkerFaceColor', lightBlue, 'MarkerEdgeColor','k'); 
        hold on;

        % numbers inside bubbles
        for t = 1:params.T
            text(cumCapacity_mean(t), cumDemand_mean(t), sprintf('%d', t), ...
                'VerticalAlignment','middle','HorizontalAlignment','center', ...
                'Color','k','FontWeight','bold');
        end

        % reference line (y=x)
        lineMin = min([cumCapacity_mean cumDemand_mean]);
        lineMax = max([cumCapacity_mean+50 cumDemand_mean+50]);
        plot([lineMin lineMax], [lineMin lineMax], '--k', 'LineWidth', 1.0);

        grid on;
        title(sprintf('Demand %s | Forecast %s', labels_d{i}, labels_e{j}));
        if i == nD, xlabel('S^{tot}'); end
        if j == 1,  ylabel('D^{tot}'); end
        hold off;
    end
end




% Weights Table (simple)
Wdim = params.Kmax + 1;   % enforce exactly Kmax+1 weights
fmt  = '%.5f';

for i = 1:nD
    for j = 1:nE
        % make a 1×Wdim row vector (and trim if anything extra sneaks in)
        w = reshape(wMat(i,j,1:Wdim), 1, []);

        % build "a, b, c" with your format
        weights_str = sprintf([fmt ', '], w);
        if ~isempty(weights_str)
            weights_str(end-1:end) = [];   % drop trailing ", "
        end

        fprintf('Demand uncertainty: %s, Forecast uncertainty: %s {%s}\n', ...
            lower(labels_d{i}), lower(labels_e{j}), weights_str);
    end
end






function [mc, det, w_used] = evalScenarioSimple(P, w_start, REOPT, opts_reopt, lb, ub, seed)
% Evaluate one scenario with optional re-optimisation.
    if REOPT
        % Lighter Monte Carlo during the search
        P_opt      = P;
        P_opt.iter = min(P.iter, 1000);   % e.g., 10k for speed (tweak if needed)

        obj = @(w) simulate_cost(w, P_opt, seed);   % deterministic via seed

        % Warm-start from provided start
        [w_used, ~] = patternsearch(obj, w_start, [],[],[],[], lb, ub, [], opts_reopt);

        % High-accuracy evaluation with the original P (e.g., 50k)
        [mc, det] = simulate_cost(w_used, P, seed);
    else
        w_used = w_start;
        [mc, det] = simulate_cost(w_used, P, seed);
    end
end


% c_k figure
figure;
plot(0:params.Kmax, params.c_k, '-o');
xlabel('k');
ylabel('Cost');
title('Unit costs of capacity installed k periods later c_k');
grid on;

% Force y-axis to start at 0
ylim([0, max(params.c_k)*1.05]);
