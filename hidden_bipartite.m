function [X, Y] = hidden_bipartite(A, p, K, eta, beta, maxiter, tol)

% This algorithm detects the hierarchical dense subgraphs with each
% hierarchy increases the density, e.g., density of 2nd hierarchy is
% smaller than that of 3rd hierarchy. And hierarchies are nested together.
% That is, the graph in the lower hierarchy contains the graph in the
% hierarchy. For example, the graph in the 3rd hierarchy is a dense
% subgraph of the graph in the 2nd hierarchy.
%
% Inputs:
%   - A: the m-by-n adjacency matrix of the input bipartite graph
%   - p: the penality constant of each missing edge
%   - K: the total number of hierarchies that the user wants to extract
%   - eta: the density increase ratio, it could be a scalar or a vector 
%   - beta: the parameter that controls the importance of density variety
%           constraints
%   - maxiter: maximum number of iterations until stop. Default is 40.
%   - tol: tolerance for stopping criterion. Default is 1e-4.
%
% Output: 
%   - X: the n-by-K indicator matrix where X(:,i) represents which node in 
%        one set is in the i-th hierarchy.
%   - Y: the n-by-K indicator matrix where Y(:,i) represents which node in 
%        the other set is in the i-th hierarchy.
% Reference:
% Zhang, Si, et al. "HiDDen: Hierarchical Dense Subgraph Detection with Application to Financial Fraud Detection." 
% Proceedings of the 2017 SIAM International Conference on Data Mining. 
% Society for Industrial and Applied Mathematics, 2017.


[m, n] = size(A);
if nargin <= 1, p = 0.001; end
if nargin <= 2, K = 10; end
if nargin <= 3, Eta = [1.5*ones(1, round(K/2)), 1.1*ones(1, K-round(K/2))]; end
if nargin <= 4, beta = 1.8; end
if nargin <= 5, maxiter = 100; end
if nargin <= 6, tol = 1e-5; end
if nargin >= 4 && length(eta) == 1, Eta = eta*ones(1,K); end

%% define function handles
% objective function only w.r.t the 1-st hierarchy indicator vector
func_x1 = @(x, y, p) -(1 + p)*x'*(A*y) + p*sum(x)*sum(y); 
% objective function w.r.t the k-th hierarchy indicator vector (k >= 2)
func_xk = @(x, y, p, C) -(1 + p + beta)*x'*(A*y) + (p + beta*C)*sum(x)*sum(y); 
% derivative function w.r.t the 1-st hierarchy indicator vector (x1 & y1)
grad_x1 = @(x, y, p) -(1 + p)*A*y + p*sum(y)*ones(m,1);
grad_y1 = @(x, y, p) -(1 + p)*A'*x + p*sum(x)*ones(n,1);
% derivative function w.r.t the k-th hierarhcy indicator vector (xk & yk)
grad_xk = @(x, y, p, C) -(1 + p + beta)*A*y + (p + beta*C)*sum(y)*ones(m,1);
grad_yk = @(x, y, p, C) -(1 + p + beta)*A'*x + (p + beta*C)*sum(x)*ones(n,1);


%% Initialization
X = 0.01*ones(m, K); Y = 0.01*ones(n, K); 
X(:, 1) = 0.5*ones(m, 1); Y(:, 1) = 0.5*ones(n, 1);
% calculate the parameter C
C0 = Eta(1)*sum(sum(A))/(m*n); 
% compute the initial gradient for each indicator vector
gradX1 = grad_x1(X(:, 1), Y(:, 1), p); gradXk = grad_xk(X(:,2), Y(:, 2), p, C0);
gradY1 = grad_y1(X(:, 1), Y(:, 1), p); gradYk = grad_yk(X(:,2), Y(:,2), p, C0);
% lower and upper bound of indicator vectors
lbx = X(:, 2); lby = Y(:, 2);
ubx = ones(m, 1); uby = ones(n, 1);
initnorm = sqrt(norm(gradX1, 'fro')^2 + (K-1)*norm(gradXk, 'fro')^2 + norm(gradY1, 'fro')^2 + (K-1)*norm(gradYk, 'fro')^2);
% compute the initial projected gradient norm by 'projgrad_x' function below
initprojX = [projgrad_x(X(:, 1), gradX1, lbx, ubx), repmat(projgrad_x(X(:, 2), gradXk, lbx, X(:, 1)), [1, K-1])];
initprojY = [projgrad_x(Y(:, 1), gradY1, lby, uby), repmat(projgrad_x(Y(:, 2), gradYk, lby, Y(:, 1)), [1, K-1])];

%% Update by alternative gradient descent
for iter = 1:maxiter   
    fprintf('iteration %d.\n', iter); 
    
    X_old = X; Y_old = Y;
    if iter == 1, 
        projgradX = initprojX; 
        projgradY = initprojY; 
    end
    
    % update for X(:,1), i.e., the 1-st hierarchy
    x_old = X_old(:, 1); y_old = Y_old(:, 1);
    func = func_x1(x_old, y_old, p); gradX1 = grad_x1(x_old, y_old, p);
    lbx = X(:, 2);
    X(:, 1) = altsub(func_x1, grad_x1, p, x_old, y_old, lbx, ubx, gradX1, func, '1', 'x', 0);
    projgradX(:, 1) = projgrad_x(X(:, 1), grad_x1(X(:, 1), Y(:, 1), p), lbx, ubx);
    % update for X(:,1), i.e., the 1-st hierarchy
    x_old = X(:, 1);
    func = func_x1(x_old, y_old, p); gradY1 = grad_y1(x_old, y_old, p);
    lby = Y(:, 2);
    Y(:, 1) = altsub(func_x1, grad_y1, p, x_old, y_old, lby, uby, gradY1, func, '1', 'y', 0);
    projgradY(:, 1) = projgrad_x(Y(:, 1), grad_y1(X(:, 1), Y(:, 1), p), lby, uby);
    
    % update for X(:, k) where k>=2, i.e., the k-th hierarhcy
    for k = 2: K
        x0 = X(:, k-1); x_old = X(:, k); 
        y0 = Y(:, k-1); y_old = Y(:, k);
        if k == K,       % define for the last hierarchy as the zeros
            lbx = zeros(m, 1);
            lby = zeros(n, 1); 
        else             % define as the indicator vector of the next hierarhcy
            lbx = X(:, k+1);
            lby = Y(:, k+1);
        end
        C0 = Eta(k)*x0'*(A*y0)/(sum(x0)*sum(y0));
        % update x
        gradXk = grad_xk(x_old, y_old, p, C0); func = func_xk(x_old, y_old, p, C0);
        X(:, k) = altsub(func_xk, grad_xk, p, x_old, y_old, lbx, x0, gradXk, func, 'k', 'x', C0);
        % update y
        gradYk = grad_yk(x_old, y_old, p, C0); func = func_xk(x_old, y_old, p, C0);
        Y(:, k) = altsub(func_xk, grad_yk, p, x_old, y_old, lby, y0, gradYk, func, 'k', 'y', C0);
        projgradY(:, k) = projgrad_x(Y(:, k), grad_yk(X(:, k), Y(:, k), p, C0), lby, y0);
    end
    
    % stopping criterion
    relativediff = ( norm([projgradX; projgradY]) < tol * initnorm );
    if relativediff == 1
        break;
    end
    
end

if iter == maxiter
    fprintf('Max iter in nlssubprob.\n');
end

end

function new = altsub(func, grad_func, p, x_old, y_old, lb, ub, grad_value, func_old, option1, option2, C)

% This function uses projected gradient descent algorithm to solve each
% optimization subproblem w.r.t the k-th hierarchy (1 <= k <= K).
%
% Input:
%   - func: function handler of the objective function
%   - grad_func: function handler of the derivative function
%   - p: penalty constant of each missing edge
%   - x_old: the indicator vector from the last outer loop
%   - lb, ub: lower and upper bound of the indicator vector
%   - grad_value: the gradient value from the last outer loop
%   - func_old: the objective function value from the last outer loop
%   - option1: differentiates whether it solves for the 1-st hierarchy or
%             k-th hierarchy (k >= 2).
%   - option2: differentiates whether it solves for 'x' or 'y'.
%   - C: the parameter same as C0 in the main function
%
% Output:
%   - new: the new indicator vector of X(:,k) (1 <= k <= K)

beta = 0.1; % ratio of the change of step size
sigma = 0.01; % parameter for line search
alpha = 1; % initial step size
cond_alpha = 0; s = 0;
upperboundalpha = 10;
maxiter = 1; tol = 1e-6;

% projected gradient descent method
for i = 1: maxiter
    
    if i == 1, 
        grad = grad_value; 
        if option2 == 'x', old = x_old;
        elseif option2 == 'y', old = y_old;
        end
    elseif i > 1 && option2 == 'x',
        x_old = new; old = x_old;
        if option1 == '1', 
            grad = grad_func(x_old, y_old, p);
            func_old = func(x_old, y_old, p);
        elseif option1 == 'k',
            grad = grad_func(x_old, y_old, p, C);
            func_old = func(x_old, y_old, p, C);
        end
    elseif i > 1 && option2 == 'y',
        y_old = new; old = y_old;
        if option1 == '1', 
            grad = grad_func(x_old, y_old, p);
            func_old = func(x_old, y_old, p);
        elseif option1 == 'k',
            grad = grad_func(x_old, y_old, p, C);
            func_old = func(x_old, y_old, p, C);
        end
    end
        
    % searching for step-size that satisfies the modified Armijo condition
    while s <= upperboundalpha && ~cond_alpha
        new = min(max(old - alpha*grad, lb), ub);
        % Safety procedure: make sure that we have unew = u - eta*grad_u >= 0: 
        % Usually not entered
        if new == 0
            indgradpos = find(grad > 0); 
            alpha = mean(old(indgradpos)./grad(indgradpos) ); 
            new = min(max(old - alpha*grad, lb), ub); 
        end

        % Armijo
        if option1 == '1'
            if option2 == 'x', func_new = func(new, y_old, p);
            elseif option2 == 'y', func_new = func(x_old, new, p); end
        elseif option1 == 'k'
            if option2 == 'x', func_new = func(new, y_old, p, C);
            elseif option2 == 'y', func_new = func(x_old, new, p, C); end
        end
        cond_alpha = ( (func_new - func_old - sigma*grad'*(new - old) <= 0) );
        s = s + 1;
        if cond_alpha == 0
            alpha = alpha*beta;
        else
            alpha = alpha/sqrt(beta);
        end
    end
    
    % stopping criterion
    if norm(new - old) < tol, break; end

end

end



function grad = projgrad_x(x, grad_x, lb, ub)
% project the gradient of xi
grad = grad_x;
grad(x == lb) = min(0,grad(x == lb));
grad(x == ub) = max(0,grad(x == ub));
end