lambda = 0.2;
alpha = 30;
beta = 100;

rmse_1 = zeros(99, 1);
rmse_2 = zeros(99, 1);
ns = 20:10:1000;
for j = 1:size(ns,2)
  n = ns(j);
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
  rmse_1(j) = diff_1' * diff_1 / m;
  rmse_2(j) = diff_2' * diff_2 / m;
  
end

[ax, h1, h2] = plotyy(ns, rmse_1, ns, rmse_2);
legend('MLE', 'MAP');
xlabel('n');
set(ax(1), 'YLim', [0, 0.003]);
set(ax(2), 'YLim', [0, 0.003]);