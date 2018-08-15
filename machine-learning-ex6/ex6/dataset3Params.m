function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
Clist = [0.01 0.03 0.1 0.3 1 3 10 30];
sigmalist = [0.01 0.03 0.1 0.3 1 3 10 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
errlist=[0];
ans=[0 0];
for i=1:8
    for j=1:8
        C1=Clist(1,i);
        sigma1=sigmalist(1,j);
        model= svmTrain(X, y, C1, @(x1, x2) gaussianKernel(x1, x2, sigma1));
        predictions = svmPredict(model, Xval);
        err=mean(double(predictions ~= yval));
        errlist=[errlist err];
        ans=[ans;[C1 sigma1]];
    end
end
[mini place]=min(errlist(1,2:65));
ans1=ans(place+1,:);
C=ans1(1);
sigma=ans1(2);

% =========================================================================

end
