import pandas as pd
import numpy as np 


def IS(l,u,y,q): 
    """
    calculation of the Interval Score
    
    l:  left interval limit
    u:  right interval limit
    y:  argument of IS
    q:  quantile level corresponding to limits
    
    Output
    IS value in y for given l,u,q
    """
    if y<l:
        return 2*(l-y) + q*(u-l) 
    elif y>u:
        return 2*(y-u) + q*(u-l) 
    else:
        return q*(u-l) 
    
def WIS_calc(ci, target):
    """
    compute WIS in the point target in for 21 quantiles q_vals(below)
    
    ci:     array with empirical quantiles of the forecast
    target: ground truth at which to estimate WIS
    
    Output:
    WIS value
    """
    q_vals = [0.01,0.025] + list(np.arange(0.05, 0.5, 0.05)) 
    WIS_qpart = np.sum([IS(ci[i], ci[len(ci)-i-1], target, q_vals[i]) for i in range(len(q_vals))])
    WIS = (0.5*np.abs(ci[len(q_vals)]-target) + WIS_qpart)/(2*len(q_vals)+1)   
    return WIS  

def compute_WIS(df, model, target_type="day"):
    """
    create a DataFrame with WIS estimates for each time point in df
    df: DataFrame with empirical quantiles for the forecast and ground truth value
    model: model name
    target_type: "cumulative" or "day" depending on the data in df
    
    Output:
    DataFrame with WIS for each time point 
    """
    WIS_ = []
    q = [0.01,0.025] + list(np.arange(0.05,1,0.05)) +  [0.975,0.99]
    q = [str(round(x,3)) for x in q]   
    if target_type=="day":
        for index, row in df[q].iterrows():  
            WIS_.append(WIS_calc(row,df["target_uptodate"][index]))
    if target_type=="cumulative": 
        for index, row in df[q].iterrows():
            WIS_.append(WIS_calc(row, df["target_cumul"][index])) 
    df[model+"WIS"] = WIS_  
    return df

def coverage_point(ci, target):    
    """
    average coverage over 21 quantiles 
    ci:     array with empirical quantiles of the forecast
    target: ground truth 
    
    Output:
    coverage averaged over 21 quantiles
    """
    q_vals = [0.01,0.025] + list(np.arange(0.05, 0.5, 0.05)) 
    coverage = np.sum([1*((ci[i]<=target)&(ci[len(ci)-i-1]>=target)) for i in range(len(q_vals))]) 
    return coverage 

def compute_coverage(df, model, target_type="day"):
    
    """
    create a DataFrame with average coverage estimates for each time point in df
    df: DataFrame with empirical quantiles for the forecast and ground truth value
    model: model name
    target_type: "cumulative" or "day" depending on the data in df
    
    Output:
    DataFrame with average coverage for each time point 
    """ 
    coverage = []
    q = [0.01,0.025] + list(np.arange(0.05,1,0.05)) +  [0.975,0.99]
    q = [str(round(x,3)) for x in q] 
    if target_type=="day":
        for index, row in df[q].iterrows(): 
            coverage.append(coverage_point(row, df["target_uptodate"][index]))
    if target_type=="cumulative": 
        for index, row in df[q].iterrows():
            coverage.append(coverage_point(row, df["target_cumul"][index])) 

    df[model+"_coverage"] = coverage  
    return df
 