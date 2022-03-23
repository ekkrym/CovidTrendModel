import os
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import UnivariateSpline  
from precomputing import precompute_forecasts

 

def confidence_intervals(cumul_observations, method, kwargs, smoothing_fun, incremental_history=[],
                         smoothed_history = [], errortype="sqrt", H=7, inc_=1):
    """
    Get confidence intervals and (optionally) plot the error bars
    cumul_observations:  all cumulative cases history
    methods: list of methods references
    kwargs_list: list of dictionaries with parameters for each method
 
    errortype: from {"MSE", "MAE", "sqrt weighted MAE", "abs weighted MAE", "abs weighted MSE", "sqrt weighted MAE"},
               where "sqrw" with weights sqrt(y)**(-1), "w" with weights (y)**(-1) 
    output: quantiles, list with each element corresponding to method.
            element is an array of all 1,...,7 out of sample forecasts normalized errors

    """  
    
    # get prefictions for the next H=7 days
    cumul_forecasts = precompute_forecasts(incremental_history[-1], method, kwargs, smoothing_fun,
                                           incremental_history=incremental_history, smoothed_history=smoothed_history,
                                           Ws=2, H=H) 
    
    npq = 5 #number of points to fit the tail coeff for extrapolation
    W = 7 # window size to compute weekly numbers
    k =  ((inc_+1)*(19)+inc_) 
 
    #for dashboard, daily confidence 
    dforecasts = np.diff(cumul_forecasts, axis=1)[:-1,:]  #-1 since the last one forecasts in the unknown future
    fc_len = np.shape(dforecasts)[0] 
    weekly = pd.Series(np.diff(cumul_observations)).rolling(W).mean() #this is centered in -4 point, forecast for a week  -H to -1 
    groundtruth = weekly[-k:]

    for i in range(H):    
        forecast = dforecasts[-k-3-i:-3-i, i]  
        AE = scale_q(forecast, weekly[-k:], errortype)
        q, qvals = compute_q(AE, inc_, npq)
        if i==0: 
            qdf = pd.DataFrame(columns = ["horizon"]+q)
        qdf = qdf.append(pd.DataFrame(np.array([i]+qvals)[np.newaxis,:], columns= qdf.columns), ignore_index=True) 
    
    # for evaluation and forecast of weekly numbers
    for i in range(1,int(H/7)+1):
        H_forecast = np.sum(np.diff(cumul_forecasts, axis=1)[:-1,:][:-i*7,(i-1)*7:i*7], axis=1) #i-weekly forecast  
        H_week =  pd.Series(np.diff(cumul_observations)).rolling(7).mean()*7 
        AE = scale_q(H_forecast[-k:], H_week[-k:], errortype)
        q, qvals = compute_q(AE, inc_, npq)  
        qdf = qdf.append(pd.DataFrame(np.array(["w"+str(i)]+qvals)[np.newaxis,:], columns=qdf.columns), ignore_index=True)      
        
    return qdf

def lr(x,y, I=True):
    """
       simple 1D linear regression with intercept
    x: list or array of 1d values
    y: response 1d list or array 
    I: boolean whether to use intercept or not
    
    output 
    alpha: intercept
    beta : linear coefficient
    """
    x = np.array(x)
    y = np.array(y)
    meanx = np.mean(x)
    meany = np.mean(y)
    if I:
        beta = np.sum((x-meanx)*(y-meany))/np.sum((x-meanx)**2)
        alpha = meany - beta*meanx
    else:
        alpha = 0
        beta = np.sum(y)/np.sum(x)
    return alpha, beta    

def scale_q(forecast, gtruth, errortype):
    
    """
       returns the normalized deviation of forecast from gtruth
       for errortype normalization ("abs" or "sqrt")
       
    forecast: 1d forecast numbers
    gtruth:   ground truth
    errortype: normalization type: abs or sqrt
    
    Output:
    normalized deciation of forecast from ground truth
    """
    err = (gtruth-forecast)   
    if "abs" in errortype: 
        weights = (forecast + 1) ** (-1)
    if "sqrt" in errortype: 
        weights = (forecast + 1) ** (-1 / 2)
    
    err = err * weights  
    return err

def compute_q(err, int_, npq):
    """
    computing 19 quantiles with the step of level 0.05  and with extrapolation  to 0.025 and 0.01 
    assuming the exponential tail
    
    input: 
    err: deviation of the forecast from the ground truth
    int_: number of intermediate points to compute the empirical quantiles
    npq:  number of empirical quantiles for each of the tail (right and left) to compute 
          the parameters of exponential tail
    Output:
    quantiles: quantile levels
    qvales:    corresponding  empirical quantiles
    """
    k =  ((int_+1)*(19)+int_)
    quantiles = np.arange(0,k+1)/(k+1) 
    quantiles05 = quantiles[::(int_+1)][1:] 
    quantiles = quantiles[1:] 
    
    r05 = range(int_,k,int_+1)
    err = np.sort(err)
    
    #estimated 19 quantiles with levels 0.05,0.1,...,0.95
    err05 = [err[i] for i in r05]   
    x = - np.array(err05[-npq:][::-1])
    y = np.log(quantiles05[:npq]) 
    
    #estimate the parameters of the right tale 
    a,b = lr(x,y)
    q975 = -(np.log(0.025)-a)/b 
    q99 = -(np.log(0.01)-a)/b
    
    
    #estimate the parameters of the left tale 
    x = err05[:npq]  
    a,b = lr(x,y)
    q025 = (np.log(0.025)-a)/b 
    q01 = (np.log(0.01)-a)/b
    quantiles = [0.01,0.025] + list(quantiles05) + [0.975,0.99]
    quantiles = [str(round(i,3)) for i in quantiles]
    qvals = [q01, q025] + err05 +[q975, q99]
    return quantiles, qvals 
 
    
def cut_negatives(a):
    res = np.where(a>0,a,0)
    return res

def normalize_back_add_mean(df_row, median, weights):
    """
    inverse of normalization and shifting the values that the median in df_row matches with "median"
    additional cutting off the negative values 
    
    df_row: a row from df 
    median:   values in df_row will be shifted after scaling that df_row["0.5"] contains "median"
    weights:  for scaling
    
    Output:
    scaled and shifted row of DataFrame
    """ 
    res = median + df_row.astype(float).values*weights - df_row["0.5"].astype(float).values*weights
    res = cut_negatives(res)
    return res

def convert_quantiles_to_ci(ci, sm_concat, H, error_ci, coef_spline_smoothing=0): 
    """
    convert the computed quantiles for the normalized errors 
    into the quantiles of the errors by scaling back 
    with additional spline smoothing 
    """ 
    weights = np.ones((1,H))
    weights_W = 1
    q = list(set(ci.columns)-set(["horizon"]))  
    forecast = np.diff(sm_concat[-H-1:])
    if error_ci=="abs": 
        weights = forecast + 1
    if "sqrt" in error_ci: 
        weights = np.sqrt(forecast + 1) 
    
    # for the weekly forecast with horizon till int(H/7), evaluation
    for i in range(1,int(H/7)+1):
        if "sqrt" in error_ci: 
            weights_W = (np.sum(forecast[(i-1)*7:i*7]) + 1) ** (1 / 2)  
        if error_ci=="abs": 
            weights_W = (np.sum(forecast[(i-1)*7:i*7]) + 1) 
        ci.loc[ci["horizon"]=="w"+str(i), q] = normalize_back_add_mean(ci.loc[ci["horizon"]=="w"+str(i),q],np.sum(forecast[(i-1)*7:i*7]),weights_W)   
            
    # for the daily forecast, for dashboard
    for i in range(H):
        ci.loc[ci["horizon"]==i,q] = normalize_back_add_mean(ci.loc[ci["horizon"]==i,q],forecast[i],weights[i])  
  
    # additional smoothing
    qn = len(ci.columns[1:])  
    for i,q  in enumerate(ci.columns[1:]):
        conf_ = ci[q].values[:H]  
        spl = UnivariateSpline(range(H),conf_)
        spl.set_smoothing_factor(coef_spline_smoothing*np.std(conf_)**2) 
        conf_ = spl(range(H)) 
        ci[q].values[:H]  = conf_  
    return ci

def save_ci(ci, date_, country, H, error_ci):
    """
    prepare a DataFrame with CI for a country
    """ 
    ci_full = ci 
    date_ = [np.datetime64(x, 'D') for x in date_][-H:]
 
    for i in range(int(H/7)):
        date_ = date_ + [date_[i*H-1]] 
    
    ci_full["date"] = date_
    ci_full["country"] = [country] * (len(date_)) 
    ci_full["confidence_norm"] = [error_ci] * (len(date_)) 
    return ci_full

 