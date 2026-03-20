function [meanCost, details] = compute_decision_cost(w, params)
%COMPUTE_DECISION_COST  Wrapper around simulate_cost for arbitrary decisions.
%
%   [MEANCOST, DETAILS] = COMPUTE_DECISION_COST(W, PARAMS) returns the
%   mean total cost produced by the linear decision rule defined by the
%   weight vector W under the model parameters specified in the structure
%   PARAMS.  It also returns a DETAILS structure identical to that
%   produced by SIMULATE_COST for diagnostics and service‐level metrics.
%
%   The decision vector W must have length at least PARAMS.Kmax+2.  Its
%   elements are interpreted as follows:
%
%     W(1)   – a base level for all investments.  This constant term
%              represents how much capacity to invest in every period
%              regardless of the demand forecasts.
%
%     W(2)   – the weight applied to the immediate demand (the first
%              element of E(:,:,1) in SIMULATE_COST) when calculating
%              s_{t,k}.  This weight scales the current period’s demand
%              estimate in the investment decision.
%
%     W(3)   – the weight applied to the one‑step ahead forecast error
%              (eps_{t,1}).  If PARAMS.Kmax≥1 then s_{t,1} depends on
%              the two elements [demand, demand forecast for t+1], and
%              W(3) scales the latter.
%
%     W(4)   – the weight applied to the two‑step ahead forecast error,
%              and so on.  Generally, for 0 ≤ k ≤ PARAMS.Kmax,
%              W(k+2) is the coefficient on the k‑step ahead forecast
%              error in the affine decision rule.  When k=0 only W(2)
%              contributes (the demand term), for k=1 the terms W(2)
%              and W(3) contribute, etc.
%
%   The function uses SIMULATE_COST to generate a Monte‑Carlo simulation
%   of demand, forecasts and capacities, then computes the investment
%   cost and expected shortfall cost.  The output MEANCOST is the
%   expected total cost over the simulation horizon.
%
%   Example:
%       params = struct('T',3,'c_s',1,
%                       'r',0.05,'c_k',1./(1 + 0.05).^(0:3),
%                       'iter',10000,'Kmax',3,'delta',0,
%                       'mu',1,'sigma',1,'increase',1.1,
%                       'sigmaeps',1,'incr',1.1,'estmu',0);
%       w = [0.05, 0.9, 0.02, 0.01, 0.005];
%       [mc, det] = compute_decision_cost(w, params);
%       fprintf('Mean cost: %.2f\n', mc);
%
%   See also SIMULATE_COST.

    % Input validation
    arguments
        w (:,1) double
        params struct
    end

    % Ensure the weight vector is long enough; pad with zeros if needed.
    Kmax = params.Kmax;
    if numel(w) < Kmax + 2
        warning('Weight vector too short; padding with zeros to length %d.', Kmax+2);
        w = [w(:); zeros(Kmax + 2 - numel(w), 1)];
    end

    % Delegate the computation to the existing simulate_cost function.
    if nargout < 2
        meanCost = simulate_cost(w(:).', params);
        details  = [];
    else
        [meanCost, details] = simulate_cost(w(:).', params);
    end

end
