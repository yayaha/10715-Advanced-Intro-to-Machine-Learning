n = 20;
lambda = 0.2;
beta = 100;

rmse_1 = zeros(30, 1);
rmse_2 = zeros(30, 1);
for alpha = 1:30
  m = 50;
  diff_1 = zeros(m, 1);
  diff_2 = zeros(m, 1);
  for i = 1:m
    x = exprnd(1 / lambda, n, 1);
    L1 = n / sum(x);
    L2 = (n + alpha - 1) / (beta + sum(x));
    diff_1(i) = L1 - lambda;
    diff_2(i) = L2 - lambda;
  end
  rmse_1(alpha) = diff_1' * diff_1 / m;
  rmse_2(alpha) = diff_2' * diff_2 / m;
  
end

plotyy(1:30, rmse_1, 1:30, rmse_2);
legend('MLE', 'MAP');