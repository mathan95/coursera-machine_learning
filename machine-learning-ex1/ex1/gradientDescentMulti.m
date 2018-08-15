function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = size(X,1); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

      
    J_history(iter) = computeCostMulti(X, y, theta);
    Q1= theta(1,1)- ((alpha/m)*(sum((X*theta)-y)));
    Q2=theta(2,1)- ((alpha/m)*(sum(((X*theta)-y).*X(:,2))));
    Q3=theta(3,1)- ((alpha/m)*(sum(((X*theta)-y).*X(:,3))));
    theta(1,1)=Q1;
    theta(2,1)=Q2;
    theta(3,1)=Q3;
end












end

