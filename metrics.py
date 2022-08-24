# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:05:39 2019

@author: acer
"""

from scipy.stats import pearsonr
from scipy.stats import spearmanr

def pearson_r(x,y):
    corr, p_value = pearsonr(x, y)
    print('Spearmans correlation coefficient: %.3f' % corr)
# interpret the significance
    alpha = 0.05
    if p_value > alpha:
	    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p_value)
    else:
	    print('Samples are correlated (reject H0) p=%.3f' % p_value)
    return corr,p_value

def spearman_r(x,y):
    coef, p = spearmanr(x, y)
    print('Spearmans correlation coefficient: %.3f' % coef)
# interpret the significance
    alpha = 0.05
    if p > alpha:
	    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
    else:
	    print('Samples are correlated (reject H0) p=%.3f' % p)
    return coef,p