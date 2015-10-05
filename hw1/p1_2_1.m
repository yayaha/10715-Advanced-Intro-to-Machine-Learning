xSum = sum(xs);
xLogSum = sum(log(xs));

alpha = 1;
beta = 1;
alphaNew = 1;
betaNew = 1;


gamma = 0.1;
n = size(xs, 1);

flag = false;

thld = 1e-6;

while (~flag)
  alphaNew = alphaNew + gamma * (n * (log(beta) - gsl_sf_psi(alpha)) + xLogSum);
  betaNew = betaNew + gamma * (n * alpha / beta - xSum);
  
  if (abs(alphaNew - alpha) < thld && abs(betaNew - beta) < thld)
    flag = true;
  end
end