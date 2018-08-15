function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
mat=(log(sigmoid(X*theta)).*(-y))+(log(1-sigmoid(X*theta)).*(-(1-y)));
theta1=theta;
theta1(1)=0;
vec1=((lambda/(2*m))*sum(theta1.*theta1));
J=(sum(mat)/(m))+vec1;
mat2=X.*(sigmoid(X*theta)-y);

vec=(lambda/m).*theta1;
grad=((sum(mat2)/m)+vec');
grad=grad(:);
end
