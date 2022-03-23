import numpy as np
import pandas as pd
import numpy.matlib as mp
import statsmodels.api as sm  
import statsmodels
from scipy import signal
import scipy as sp 
import matplotlib.pyplot as plt
import csaps
from statsmodels.tsa.seasonal import STL as STL
import scipy.stats as stats
import statsmodels.api as sm 
  
def simple_mirroring(x, Ws=8, daily=False,corr=4, H=7): 
    """
    simple smoothing: equalizing two observation forward and backwards  
    with linear forecast in the initial or log scale in the end of time series 
    for the last "corr" values 
    
    x:    cumulative numbers if not "daily" otherwise daily numbers
    Ws:   number of iterations  
    corr: length of history in the end to be corrected with linear trend
    H:    parameter for imputation function, the non-zero observation would be spread 
          uniformly backwards for maximum 2*H zero days (see function "imputations" below) 
    
    Output
    smoothed cumulative numbers if not "daily" otherwise daily numbers
    """
    x = np.array(x)  
    
    if len(x)<2:
        return x
    if daily:
        z = x.copy()
    else:
        z = np.diff(x.copy(), axis=0)  
    z = np.array([0]*2*corr+list(z))  
    
    if (len(z)>H+1) and (corr>0): 
        z = imputations(z, H=H)    
 
    range_J = range(len(z)-1)
    for i in range(Ws):
        range_J = range_J[::-1] 
        for j in range_J: 
            running_z = 0.5*(z[j] + z[j+1])  
            z[j], z[j + 1] = running_z, running_z 
     
    if corr>0 and len(z)>corr: 
        if z[-corr ]-z[-corr-1]>0:
            z[-corr:] = z[-corr] + (z[-corr ]-z[-corr-1])*range(corr)
        else:
            z[-corr:] = np.exp(np.log(z[-corr]+1) + (np.log(1+z[-corr])-np.log(1+z[-corr-1]))*range(corr)) 
            z[-corr:] = np.where(z[-corr:]-1>0,z[-corr:]-1,0)
    if not daily:
        cumul = np.cumsum(list([x[0]]) + list(z))[-len(x):] 
        return x[-1]*cumul/cumul[-1] 
    else:  
        if np.sum(z[-len(x):])>0:
            return np.sum(x)*z[-len(x):]/np.sum(z[-len(x):])
        else:
            return z[-len(x):]

        
        
def two_step_STL(x, start_p=0, robust=False, period=7, trend=15):
    """
    apply STL twice, first for outlier removal if robust and next non-robust version for additional smoothing
    x:       raw daily numbers
    start_p: starting point for applying STL
    robust:  boolean, whether to apply preliminary robust STL smoothing step      
    period:  period pararmeter in STL from statsmodels.tsa.seasonal 
    trend:   trend smoothing window parameter in STL from statsmodels.tsa.seasonal 
    
    Output:
    smoothed daily numbers in x starting from start_p 
    """
    H = 7 
    z = x.copy() 
    if robust: 
        stl_daily = STL(z[start_p:], robust=robust, seasonal=period, period=period, trend=trend).fit() 
        trend_no_outl =  stl_daily.trend  
        trend_no_outl = np.where(trend_no_outl>0, trend_no_outl, 0)  
        z[start_p:] = trend_no_outl 
        
    stl_no_outliers =  STL(z[start_p:], seasonal=period, period=period, trend=trend).fit()
    z[start_p:] =  np.where(stl_no_outliers.trend>0, stl_no_outliers.trend, 0)  
    return z 

def imputations(z0,H=7):
    """
    spread the non-zero observation in time series back in time in the case of preceeding zeros
    
    z0: daily numbers
    H:  parameter for imputation function, the non-zero observation would be spread 
          uniformly backwards for maximum 2*H zero days
    Output:
    corrected daily numbers
    
    """
    a = np.where(z0>0)[0]  
    z = z0.copy() 
    for i, j in enumerate(a):
        if (i >= 1): 
            temp = a[i] - a[i-1]
            if (temp >= 2):
                if temp<=2*H:
                    z[a[i-1]+1:j+1] = z0[j]/temp 
    return np.array(z)

def redistribute_excess(z0, z, smoothed, smoothed_last,index,H=7, most_recent=False):
    """
    scaling the excess of observations when smoothing:  
    if it is the last interval (if "most_recent") smoothing in the last interval is scaled 
    that its sum meets the raw numbers sum
    
    if the excess=(sum of the smoothed observations  minus raw observations starting from "index") is
    positive, then the excess is redistributed backwards to update z[index:] till
    otherwise - the smoothing from "index" is scaled to meet the raw numbers sum  
    
    z0:         raw daily data
    z:          raw data scaled together with smoothed if excess is positive
    smoothed:   daily data smoothed from index "index"    
    smoothed_last: smoothed subinterval scaled together with "smoothed" if  most_recent
    index:      index of time starting from which the piecewise smoothing has been performed
    H:          threshold for checking the that there are enough non-zero observations
    
    output 
    smoothed: smoothed and rescaled daily data from index "index"
    smoothed_last: corrected last interval
    z       : raw data corrected for rescaling
    
    """
    #excess of raw data
    outl_last_int = np.sum(z0[index:]) - np.sum(smoothed[index:]) 
    
    #for the most recent data rescale the smoothed data for any non-zero excess
    #that the sum of smoothed meets the last observations
    if most_recent: 
        if np.sum(smoothed[index:])>0:
            scaler = np.sum(z0[index:])/np.sum(smoothed[index:]) 
            smoothed[index:] =  smoothed[index:]*scaler 
            smoothed_last = smoothed_last*scaler
        return smoothed, smoothed_last, z 
    
    #for positive excess spread it to the raw data before index "index"
    if (outl_last_int>0)&(np.sum(z[index:])>0) & (np.count_nonzero(z[:index])>H): 
        scaler = (outl_last_int+np.sum(z0[:index]))/np.sum(z[:index])
        z[:index] = z[:index]*scaler  
    #for negative excess the smoothed data is scaled in the future after index "index"
    if (outl_last_int<0)&(np.sum(z[:index])>0):   
        scaler = np.sum(z0[index:])/(np.sum(smoothed[index:]))
        smoothed[index:] = smoothed[index:]*scaler
                                                                                            
    return smoothed, smoothed_last, z

 

def piecewise_STL(x, Ws=3, H=7, log_scale=False, len_piece=21, robust=True, period=7, trend=15):
    """
    applying the STL in the overlaping intervals of time series with additional rescaling 
    for the details see the paper 
    
    x:  cumulative numbers
    Ws: redundant parameter for unification of smoothing methods 
    H:  threshold for imputation and rescaling
    log_scale: whether to use log scale for daily numbers
    len_piece: the length of sub-interval for smoothing, 
               STL is applied to a subinterval of lengths 2*len_piece,
               in the overlapping len_piece subinterval the STL trends
               are smoothly combined with sigmoidal weights
    robust:  boolean, whether to apply preliminary robust STL smoothing step      
    period:  period pararmeter in STL from statsmodels.tsa.seasonal 
    trend:   trend smoothing window parameter in STL from statsmodels.tsa.seasonal 
    
    Output:
    smoothed cumulative cases
    """ 
    #daily numbers
    z0 = np.diff(x, axis=0) 
    #correct zero ovservations
    z0 = imputations(z0, H=H)   
    if log_scale:
        z0 = np.log(z0+1) 
        
    #raw daily numbers to partially rescale in the procedure below
    z = z0.copy()  
    
    # subintervals
    int_lims = np.unique([0]+list(np.sort(np.arange(len(z),-1,-len_piece))))
    # smooth weights to combine local smoothings
    weights =  1./(1.+np.exp(10.549*(np.arange(len_piece)/len_piece-0.518)))  
    
    if len(int_lims)>3:
        #the result of smoothing in
        smoothed = z.copy()
        #last sub-interval
        last_interval = z[int_lims[-3]:].copy() 
        #smooth the data in the last sub-interval
        smoothed_last = two_step_STL(last_interval, 0, robust=robust,period=period, trend=trend)   
        #save first half subinterval to the smoothed
        smoothed[int_lims[-2]:] = smoothed_last[len_piece:].copy()
        #rescale that the sum of raw numbers is equal to sum of smoothed numbers in the last interval
        smoothed, smoothed_last, z = redistribute_excess(z0, z, smoothed, smoothed_last, int_lims[-2],most_recent=True) 
           
        #repreat backwards in subintervals    
        for i in range(len(int_lims)-4,-1,-1):   
            # take the next (back in time) subinterval to smooth 
            next_sub_interval = z[int_lims[i]:int_lims[i+2]].copy()
            # smooth with STL
            smoothed_next = two_step_STL(next_sub_interval, 0, robust=False, period=period, trend=trend)     
            # "sew" with smooth trend of previously smoothed subinterval
            smoothed[int_lims[i+1]:int_lims[i+2]] = smoothed_last[:len_piece]*(1-weights) + smoothed_next[-len_piece:]*(weights) 
            smoothed_last = smoothed_next.copy() 
            # redistribute the excess
            smoothed, smoothed_last, z = redistribute_excess(z0, z, smoothed, smoothed_last, int_lims[i+1])
        smoothed[:int_lims[1]] = smoothed_last[:int_lims[1]]   
    else:
        # if there are too few observations, use non-robust STL
        smoothed = two_step_STL(z)   
  
    if log_scale:
        smoothed = np.exp(smoothed)-1
    cumsum_sm = np.cumsum(list([x[0]]) + list(np.where(smoothed>0,smoothed,0))) 
    #final scaling to meet the last cumulative observation
    if cumsum_sm[-1]>0:
        cumsum_sm = cumsum_sm*x[-1]/cumsum_sm[-1]                
    return cumsum_sm
 

 