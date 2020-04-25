function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% in this case
% Theta1: 25 x 401
% Theta2: 10 x 26
% X: M x 400

X = [ones(m, 1) X]';     % X: 401 x M, where each column is an x input with bias

A = sigmoid(Theta1 * X); % A: 25 x M
A = [ones(1, m); A];     % A: 26 X M, where each column is an a input with bias

Y = sigmoid(Theta2 * A); % Y: 10 x M, where each column is a prediction
[_, p] = max(Y);         % p row vector with the index of the maximum of each column
p = p';

% =========================================================================


end
