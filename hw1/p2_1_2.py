import numpy as np
import scipy.special
import pdb
import pylab as pl

f = open('data/estimate.mat')

nums = []

for line in f:
	if line.startswith('#') or line.strip() == '':
		continue
	nums.append(float(line.strip()))

f.close()


f = open('p2_1_2_result', 'w')
xs = np.array(nums)
xSum = sum(xs)
xLogSum = sum(np.log(xs))

# Set initial value close to true value, which we know from question 1
x = np.array([4, 0.5])
# x = np.random.rand(2) * 10
maxIter = 10000
tol = 1e-7
n = len(nums)

alphas = [4]
betas = [0.5]

for i in xrange(maxIter):
	alpha = x[0]
	beta = x[1]

	f.write(str(alpha) + ' ' + str(beta) + '\n')

	# Calculate function value
	F1 = n * (np.log(beta) - scipy.special.psi(alpha)) + xLogSum
	F2 = n * alpha / beta - xSum
	F = np.array([F1, F2])

	# Calculate Jacobian Matrix
	F1_alpha = - n * scipy.special.polygamma(1, alpha)
	F1_beta = n / beta
	F2_alpha = n / beta
	F2_beta = - n * alpha / (beta ** 2)
	J = np.array([[F1_alpha, F1_beta], [F2_alpha, F2_beta]])

	# Solve linear system
	xNew = np.linalg.solve(J, -F) + x
	alphas.append(xNew[0])
	betas.append(xNew[1])
	# pdb.set_trace()
	if np.allclose(xNew, x, 0, tol):
		break
	x = xNew

f.write(str(alpha) + ' ' + str(beta) + '\n')

f.close()


pl.plot(xrange(len(alphas)), alphas, label='alpha')
pl.plot(xrange(len(alphas)), betas, label='beta')
pl.legend()
pl.title('Newton\'s method')
pl.show()