import numpy as np

import matplotlib.pyplot as plt

#%matplotlib inline

mu, sigma = 0, 0.1 # mean and standard deviation

sample = np.random.normal(mu, sigma, 500)

print('mean and std of drawn sample is {}, {}'.format(np.mean(sample), np.std(sample)))