function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

sumed=sum(X_norm);
mean=sumed/size(X,1);
mean_vect=[];
mu=mean;
for i=1:size(X,1)
    mean_vect=[mean_vect;mean];

end
for i=1:size(X,2)
    stand(i)=std(X(:,i));
end

sigma=stand;
X_norm=(X-mean_vect)./stand;