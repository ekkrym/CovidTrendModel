 
import more_itertools
import datetime
from datetime import timedelta
import os
import json
import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
 
# ============================================================================
# functions
# ============================================================================
def ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1)/float(len(xs))
    return xs, ys

def get_slope_bin_edges(trend,nbins=10):
    if type(trend) is list:
        # aggregate slope data
        slope = np.diff(trend[0])/(1+trend[0][:-1])
        for ii in range(len(trend)-1):
            slope = np.concatenate((slope,np.diff(trend[ii+1])/(1+trend[ii+1][:-1])),axis=0)
    else:
        # compute slopes from data
        slope = np.diff(trend)/(1+(trend[1:]+trend[:-1])/2)
 
    # get histogram of slopes
    hist, bin_edges = np.histogram(slope)
    
    # empirical CDF
    xc,yc = ecdf(slope)
    
    # new edges
    b_edge = np.zeros(nbins+1); b_edge[0] = bin_edges[0]; b_edge[-1] = bin_edges[-1]    
    b_edge[1:-1] = np.interp(np.linspace(0,1,nbins+1)[1:-1],yc,xc)
    
    return b_edge
    
def error_slope(err,trend,bins='auto'):
    """
    This function returns the average error as a function of the slope
    of the trend signal.
    
    Use:
        
        avg_err, bin_edges = error_slope(err,trend,bins='auto')
        
    Input:
        err   : array with error signals or list of arrays
        trend : corresponding trend signal associated with the errors
        bins  : ['auto'] if scalar, the it is the number of bins for the
                slope (see np.histogram)
                
    Output:
        avg_err : average error associated to each bin
        bin_edges : edges that specify binning of slope
        
    """
    if type(err) is list:
        # aggregate slope data
        slope = np.diff(trend[0])/(1+trend[0][:-1])
        e_arr = err[0][:-1]
        for ii in range(len(err)-1):
            slope = np.concatenate((slope,np.diff(trend[ii+1])/(1+trend[ii+1][:-1])),axis=0)
            e_arr = np.concatenate((e_arr,err[ii+1][:-1]),axis=0)
        print(np.max(e_arr)) 
    else:
        # compute slopes from data
        slope = np.diff(trend)/(1+(trend[1:]+trend[:-1])/2)
        e_arr = err[:-1]
 
    # get histogram of slopes
    hist, bin_edges = np.histogram(slope,bins=bins)
    
    # bin error
    bindx = np.digitize(slope,bin_edges)
    
    # average relative error
    
    #changed to return the median and quantiles
    avg_err = np.zeros(bin_edges.size-1); avg_err[:] = np.nan
    avg_err_u = np.zeros(bin_edges.size-1); avg_err_u[:] = np.nan
    avg_err_l = np.zeros(bin_edges.size-1); avg_err_l[:] = np.nan
    
    for ii in np.arange(bin_edges.size-1):
        idx = bindx==(ii+1)
        if np.any(idx):
            avg_err[ii] = np.median(e_arr[idx])
            avg_err_u[ii] = np.quantile(e_arr[idx],0.7)
            avg_err_l[ii] = np.quantile(e_arr[idx],0.3)
    # return result 
    return avg_err, bin_edges,  avg_err_u,  avg_err_l

