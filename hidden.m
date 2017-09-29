function X = hidden(A, p, K, eta, beta, maxiter, tol)

% This algorithm detects the hierarchical dense subgraphs with each
% hierarchy increases the density, e.g., density of 2nd hierarchy is
% smaller than that of 3rd hierarchy. And hierarchies are nested together.
% That is, the graph in the lower hierarchy contains the graph in the
% hierarchy. For example, the graph in the 3rd hierarchy is a dense
% subgraph of the graph in the 2nd hierarchy.
%
% Inputs:
%   - A: the adjacency matrix of the input network
%   - p: the penality constant of each missing edge
%   - K: the total number of hierarchies that the user wants to extract
%   - eta: the density increase ratio, it could be a scalar or a vector 
%   - beta: the parameter that controls the importance of density variety
%           constraints
%   - maxiter: maximum number of iterations until stop. Default is 40.
%   - tol: tolerance for stopping criterion. Default is 1e-4.
%
% Output: 
%   - X: the n-by-K indicator matrix where X(:,i) represents which node is 
%        in the i-th hierarchy.
% Reference:
% Zhang, Si, et al. "HiDDen: Hierarchical Dense Subgraph Detection with Application to Financial Fraud Detection." 
% Proceedings of the 2017 SIAM International Conference on Data Mining. 
% Society for Industrial and Applied Mathematics, 2017.



n = size(A, 1);
if nargin <= 1, p = 0.008; end
if nargin <= 2, K = 10; end
if nargin <= 3, Eta = [1.5*ones(1, round(K/2)), 1.25*ones(1, K-round(K/2))]; end
if nargin <= 4, beta = 2.5; end
if nargin <= 5, maxiter = 100; end
if nargin <= 6, tol = 1e-5; end
if nargin >= 4 && length(eta) == 1, Eta = eta*ones(1,K); end
 
%% define function handles
% objective function only w.r.t the 1-st hierarchy indicator vector
func_x1 = @(x, p) -(1 + p)*x'*(A*x) + p*sum(x)^2 - p*norm(x, 2)^2; 
% objective function w.r.t the k-th hierarchy indicator vector (k>=2)
func_xk = @(x, p, C) -(1 + p + beta)*x'*(A*x) + (p + beta*C)*sum(x)^2 - (p + beta*C)*norm(x, 2)^2;
% derivative function w.r.t the 1-st hierarchy indicator vector 
grad_x1 = @(x, p) -2*(1 + p)*A*x + 2*p*sum(x)*ones(n,1) - 2*p*x;
% derivative function w.r.t the k-th hierarhcy indicator vector
grad_xk = @(x, p, C) -2*(1 + p + beta)*A*x + 2*(p + beta*C)*(sum(x)*ones(n,1) - x);

%% Initialization
X = 0.01*ones(n, K); X(:, 1) = 0.5*ones(n, 1);
lb = X(:, 2); ub = ones(n, 1);      % lower and upper bound for X(:,1) 
C0 = Eta(1)*sum(sum(A))/(2*n^2 - 2*n); % calculate the parameter C of Eq. 3.4
% compute the gradient for each indicator vector
gradX1 = grad_x1(X(:, 1), p); gradXk = grad_xk(X(:,2), p, C0);
initnorm = sqrt(norm(gradX1, 'fro')^2 + (K-1)*norm(gradXk, 'fro')^2);
% compute the initial projected gradient norm by 'projgrad_x' function below as in Eq. 3.6
initproj = [projgrad_x(X(:, 1), gradX1, lb, ub), repmat(projgrad_x(X(:, 2), gradXk, lb, X(:, 1)), [1, K-1])];


%% Update by alternative gradient descent
for iter = 1:maxiter     
    fprintf('iteration %d.\n', iter); 
    if iter == 1, projgrad = initproj; end
    X_old = X; 
    
    % update for X(:,1), i.e., the 1-st hierarchy
    x_old = X_old(:, 1);   
    func = func_x1(x_old, p); gradX1 = grad_x1(x_old, p); 
    % lower bound as the current 2nd hierarchy indicator vector, i.e.,
    % nested node constraint
    lb = X(:, 2); 
    % solve the subproblem of X(:,1) where C0 = 0 because it has no use.
    X(:, 1) = altsub(func_x1, grad_x1, p, x_old, lb, ub, gradX1, func, '1', 0);
    projgrad(:, 1) = projgrad_x(X(:, 1), grad_x1(X(:, 1), p), lb, ub);
    
    % update for X(:, k) where k>=2, i.e., the k-th hierarhcy
    for k = 2: K
        
        x0 = X(:, k-1); x_old = X(:, k); 
        if k == K, 
            lb = zeros(n, 1);  % define for the last hierarchy as the zeros
        else
            lb = X(:, k+1); % define as the indicator vector of the next hierarhcy
        end
        % compute the parameter C
        eta = Eta(k);
        C0 = eta * x0' * (A * x0)/(sum(x0)^2 - norm(x0, 2)^2);
        gradXk = grad_xk(x_old, p, C0); func = func_xk(x_old, p, C0);
        % solve the subproblem of X(:, k)
        X(:, k) = altsub(func_xk, grad_xk, p, x_old, lb, x0, gradXk, func, 'k', C0);
        projgrad(:, k) = projgrad_x(X(:, k), grad_xk(X(:, k), p, C0), lb, x0);
        
    end
    
    % stopping criterion
    relativediff = ( norm(projgrad) < tol * initnorm );
    if relativediff == 1
        break;
    end
    
end
    
if iter == maxiter
    fprintf('Max iter in nlssubprob.\n');
end

end

function new = altsub(func, grad_func, p, x_old, lb, ub, grad_value, func_old, option, C)

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
%   - option: differentiates whether it solves for the 1-st hierarchy or
%             k-th hierarchy (k >= 2).
%   - C: the parameter same as C0 in the main function
%
% Output:
%   - new: the new indicator vector of X(:,k) (1 <= k <= K)


alpha = 1; % initial step size
beta = 0.1; % ratio of the change of step size
sigma = 0.01; % parameter for line search
maxiter = 100; tol = 1e-6;

cond_alpha = 0; s = 0;
upperboundalpha = 20;

% projected gradient descent method
for i = 1: maxiter
    
    if i == 1, grad = grad_value; 
    else
        x_old = new; 
        if option == '1', 
            grad = grad_func(x_old, p);
            func_old = func(x_old, p);
        elseif option == 'k',
            grad = grad_func(x_old, p, C);
            func_old = func(x_old, p, C);
        end
    end

    % searching for step-size that satisfies the modified Armijo condition
    while s <= upperboundalpha && ~cond_alpha
        new = min(max(x_old-alpha*grad, lb), ub);
        % Safety procedure: make sure that we have unew = u - eta*grad_u >= 0: 
        % Usually not entered
        if new == 0
            indgradpos = find(grad > 0); 
            alpha = mean(x_old(indgradpos)./grad(indgradpos) ); 
            new = min(max(x_old - alpha*grad, lb), ub); 
        end

        % Armijo
        if option == '1', func_new = func(new, p);
        elseif option == 'k', func_new = func(new, p, C); end
        cond_alpha = ( (func_new - func_old - sigma*grad'*(new - x_old) <= 0) );
        s = s + 1;
        if cond_alpha == 0
            alpha = alpha*beta;
        else
            alpha = alpha/sqrt(beta);
        end
    end
    
    % stopping criterion of the subproblem
    if norm(new - x_old) < tol, break; end
 
end

end

function grad = projgrad_x(x, grad_x, lb, ub)
% project the gradient of xi
grad = grad_x;
grad(x == lb) = min(0,grad(x == lb));
grad(x == ub) = max(0,grad(x == ub));
end