import numpy as np 

def test_poisson(cumulative_cases, date,
                 threshold = 0.2, H=7):
    """
    Simple test whether to consider zeros in the end of time series as missing data
    if the average of last H observation is less than threshold, then the last zeros 
    are considered to be missing
    
    Input 
    cumulative_cases list, corresponding date list, threshold and forecast horizon H
    
    Outputs 
    cumulative_cases: without last missing data 
    date: without last missing data
    flag_nonzero_recent: boolean whether missing data was found
    zeros_missing: list of last missing values
    zeros_missing_nan: list of last missing values           
    """
    zeros_missing = zeros_missing_nan = [] 
    
    if len(np.where(np.diff(cumulative_cases)>0)[0])==0:
        return  cumulative_cases, date, False, zeros_missing, zeros_missing_nan 
    
    max_nonzero = np.max(np.where(np.diff(cumulative_cases)>0)) 
    flag_nonzero_recent = True
    num_last_zeros = len(cumulative_cases)-max_nonzero-2 
    if num_last_zeros>0:
        if num_last_zeros>H:
            flag_nonzero_recent = False
        else:
            #poisson intensity estimation and test
            int_a = np.max([0, max_nonzero+1-H]) 
            intensity = (cumulative_cases[-2]-cumulative_cases[-H-1])/H
            probability = np.exp(-intensity)  
            if probability<threshold:
                cumulative_cases = cumulative_cases[:max_nonzero+2]
                date = date[:max_nonzero+2] 
                zeros_missing_nan = [np.nan]*num_last_zeros
                zeros_missing = [0]*num_last_zeros
                comment = "zero observations in the end, the predictions start from the last non-zero" 
    return  cumulative_cases, date, flag_nonzero_recent, zeros_missing, zeros_missing_nan 

 