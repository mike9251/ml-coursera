function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
pos = find(y==1);
neg = find(y==0);

figure; hold on;
plot(X(pos, 1), X(pos, 2), 'k+', 'MarkerSize', 10);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
grid on;
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%









% =========================================================================



hold off;

end
