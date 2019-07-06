%Rainfall prediction using Logistic Regression.
%Rainfall is predicted using Precipitaion percentage, humidity percentage
%and wind speed in km/hr.
%Data has been taken from google.


%% Initialization
clear; close all; clc


data = load('Data.txt');
X = data(:, [1, 2,3]); y = data(:, 5);




%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n,e,r] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);





%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 4000);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);





prob = sigmoid([1 35 50 14 ] * theta);
fprintf(['For place with precipitation of 35, humidity of 50 and wind speed of 14km/hr, we predict an  ' ...
         'probability of raining= %f\n\n'], prob*100);
	 if(prob*100>99.5)
         
         fprintf('Heavy rainfall is expected\n')
     end

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
