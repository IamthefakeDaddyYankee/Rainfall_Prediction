function p = predict(theta, X)


m = size(X, 1); % Number of training examples

p = round(sigmoid(X * theta));

end