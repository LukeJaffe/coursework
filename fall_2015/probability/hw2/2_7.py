# A noisy resistor produces a voltage v_n(t). At t = t_1, 
# the noise level X = v_n(t_1) is known to be a Gaussian RV
# with pdf:
# f_X(x) = 1 / sqrt(2*pi*sigma^2) * exp[(-1/2)(x/sigma)^2]
# Compute and plot the probability that |X| > k*sigma for k = 1,2,...

import math

def f_X(x):
    
