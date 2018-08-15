function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
mat=(log(sigmoid(X*theta)).*(-y))+(log(1-sigmoid(X*theta)).*(-(1-y)));
theta1=theta;
theta1(1)=0;
vec1=((lambda/(2*m))*sum(theta1.*theta1));
J=(sum(mat)/(m))+vec1;
mat2=X.*(sigmoid(X*theta)-y);

vec=(lambda/m).*theta1;
grad=((sum(mat2)/m)+vec');





% =============================================================

end