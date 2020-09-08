clear all; clc;close all;

% load data sets
load example.mat;

% inital setting
lambda = 0.01;

% calculate network edges
for i = 1:360
    % SN without i-th regions
    sn = lg;
    sn(i,:) = []; sn(:,i) = [];
    
    % Graph laplacian of SN
    L = diag(sum(sn-diag(diag(sn)),2))-(sn-diag(diag(sn)));
    
    % fMRI time-seirs of i-th region 
    Y = X(:,i);
    X_remain = X(:,[1:i-1,i+1:end]);
    
    % Estimate edges between i-th region and ~i-th regions    
    [beta, ~] = sfn_simplex(X_remain, Y, L, lambda);
    
    % Constructing sfn
    S(:,i) = [beta(1:i-1);0;beta(i:end)];
end
