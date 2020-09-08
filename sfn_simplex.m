% --------------------------------------------------------------------
% Structrual enriched functional network 
% based on simplex framework with graphnet constraint 
%
%  min  ||X*beta - y||^2 + lg*beta'*G*beta
%  s.t. beta>=0, 1'beta=1
%
% Input:
%   - y: functional MRI time-series data for target region (n x 1)
%   - X: functional MRI time-series data for remaining regions ( n x (p-1))
%   - G: graph laplacian of structrual network (p x p)
%   - lg: tuning parameter for graphnet constraint (lg > 0)
%   - (optional) beta0: inital coefficient vector
%
% Output:
%   - beta: estimated coefficient vector
%
% Author: Mansu Kim, mansu.kim@pennmedicine.upenn.edu
% Date created: Sep-07-2020
% @University of Pennsylvania Perelman School of Medicine
% --------------------------------------------------------------------


function [beta, obj]= sfn_simplex(X, y, G, lg)

[~, n] = size(X);

% intial setting
Xy = X'*y;
beta = 1/n*ones(n,1);

num_iter = 500;
max_iter = 20;

prev_beta = beta;
t = 1;
t1 = 0;
r = 0.5;

for iter = 1:num_iter
    p = (t1-1)/t;
    s = beta + p*(beta-prev_beta);
    prev_beta = beta;
    g = X'*(X*s) + lg*G*s - Xy;
    ob1 = norm(X*beta - y);
    for it = 1:max_iter
        proj_beta = s - r*g;
        proj_beta = simplex(proj_beta);
        ob = norm(X*proj_beta - y);
        if ob1 < ob
            r = 0.5*r;
        else
            break;
        end
    end
    
    if max(proj_beta - prev_beta)./max(prev_beta) < 0.01
        break;
    end
    
    % update beta
    beta = proj_beta;   
    t1 = t;
    t = (1+sqrt(1+4*t^2))/2;
    obj(iter) = ob;    
end

function [x, ft] = simplex(v)

ft=1;
n = length(v);

v0 = v-mean(v) + 1/n;
vmin = min(v0);
if vmin < 0
    f = 1;
    lambda_m = 0;
    while abs(f) > 10^-10
        v1 = lambda_m - v0;
        posidx = v1>0;
        npos = sum(posidx);
        g = npos/n - 1;
        f = sum(v1(posidx))/n-lambda_m;
        lambda_m = lambda_m - f/g;
        ft=ft+1;
        if ft > 100
            x = max(-v1,0);
            break;
        end
    end
    x = max(-v1,0);

else
    x = v0;
end
