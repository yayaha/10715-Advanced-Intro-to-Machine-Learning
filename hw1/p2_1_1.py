import numpy as np
import scipy.special
import pylab as pl


f = open('data/estimate.mat')

nums = []

for line in f:
	if line.startswith('#') or line.strip() == '':
		continue
	nums.append(float(line.strip()))

f.close()

f = open('p2_1_1_result', 'w')
x = np.array(nums)
xSum = sum(x)
xLogSum = sum(np.log(x))

alpha = np.random.rand() * 100
beta = np.random.rand() * 100
maxIter = 1000000
stepsize = 1e-3
thld = 1e-7
n = len(nums)

flag = False

alphas = [alpha]
betas =[beta]

for i in xrange(maxIter):
	alphaNew = alpha + stepsize * (n * (np.log(beta) - scipy.special.psi(alpha)) + xLogSum)
	betaNew = beta + stepsize * (n * alpha / beta - xSum)
	if (np.abs(alpha - alphaNew) < thld) and (np.abs(beta - betaNew) < thld):
		flag = True	
	alpha = alphaNew
	beta = betaNew
	f.write(str(alpha) + ' ' + str(beta) + '\n')
	alphas.append(alpha)
	betas.append(beta)
	if flag:
		break



f.close()

pl.plot(xrange(len(alphas)), alphas, label='alpha')
pl.plot(xrange(len(alphas)), betas, label='beta')
pl.legend()
pl.title('Gradient descent')
pl.show()
