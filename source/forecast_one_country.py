import numpy as np
import time
import pandas as pd
from matplotlib import pyplot as plt 

from benchmark import benchmark 
from precomputing import compute_method_depth, precompute_smoothing, precompute_outofsample, minimal_method_depth, repair_increasing
from visualization import plot_last_forecast 
from smoothing import simple_mirroring, piecewise_STL
from precompute_smoothing import update_presmoothing_one_country
from test_missing_or_zero import test_poisson  

from misc_methods import mean_const, linear  
from precomputing import precompute_forecasts
from ci import confidence_intervals, convert_quantiles_to_ci, save_ci 

def forecast_one_country(return_dict, country, cumulative_cases, date_, 
                         methods_, kwargs_list_, names_, 
                         smoothing_fun,
                         datasource,   
                         H=7, 
                         type_data = "cases",
                         missing_var=True,
                         return_val = False,
                         saveplot=False, newly=False): 
    """
    main function with methodology for singly country/region numbers forecastng: 
    preprocessing, trend estimation and forecasting
    
    
    Arguments:
    -- return_dict: dictionary for parallelized runs 
    -- country: country or region name
    -- cumulative_cases: np array with the cumulative historic observations
    -- date_: list with dates corresponding to cumulative_cases
    -- methods_, kwargs_list_, names_: parameters of the extrapolation methods   
    -- smoothing_fun: smoothing function
    -- datasource: name of the datasource, e.g. "JHU"   
    -- H: forecasting horizon 
    -- type_data: type of the data, e.g. "cases"
    -- missing_var: whether to check for missing values 
    -- return_val: whether to return values, for non-parallel use
    -- saveplot: boolean, whether to save figures with forecast
    -- newly: whether to save trend in smoothing_res
    
    if return_val, function returns the trend, confidence intervals and retrospective trends (used in evaluation)
    """
    
    methods_min_depths = [minimal_method_depth(methods_[i], kwargs_list_[i]) for i in range(len(methods_))]
    comment = "" 
    inc_ = 1 #number of intermediate points for quantile estimation
    history_for_conf_interval =  ((inc_+1)*(19)+inc_)+H+3 #length of history used to compute CI
 
    #------------------handling negative values--------------------------------------
    cumulative_cases_ = repair_increasing(cumulative_cases.copy())   
    
    #------------------handling missingness------------------------------------------
    if missing_var:
        cumulative_cases_, date_, flag_nonzero_recent, zeros_missing, zeros_missing_nan  = test_poisson(cumulative_cases_, date_)     
    else:
        _, _, flag_nonzero_recent, _, _ = test_poisson(cumulative_cases_, date_)  
        zeros_missing, zeros_missing_nan = [], [] 
    total_days_to_neglect = len(zeros_missing)
    
    
    #------------------treating special cases in delay in reporing--------------------
    if (country in ["Switzerland", "Belgium"]) or (datasource in ["OZH",'BAG_KT']): 
        step_back = 1
        if country in ["Belgium"]:
            step_back = 3
        if datasource in ["OZH",'BAG_KT']:
            step_back = 1
            if (datasource == "OZH") and (len(zeros_missing)>step_back):
                step_back=0
        total_days_to_neglect = step_back + len(zeros_missing) 
        if (step_back>0): 
            date_ = date_[:-step_back]
            zeros_missing = list(np.diff(cumulative_cases_)[-step_back:]) + zeros_missing
            zeros_missing_nan = list(np.diff(cumulative_cases_)[-step_back:]) + zeros_missing_nan
            cumulative_cases_ = cumulative_cases_[:-step_back]
    #--------------------------------------------------------------------------------- 
    H_ = H + total_days_to_neglect # forecast on longer horizon in case of missing data
    data_hist = precompute_outofsample(cumulative_cases_)
    start_smoothing = np.max([0,len(cumulative_cases_)-history_for_conf_interval-1])
   
    #--------compute the trend--------------------------------------------------------
    smoothed_hist = update_presmoothing_one_country(
                        cumulative_cases_, date_, country, 
                        smoothing_fun, 
                        datasource = datasource, newly=newly)   
    nnz_recent = np.count_nonzero(np.diff(smoothed_hist[-1][-H:]))
    
    #---------forecasting-------------------------------------------------------------
    
    if (len(cumulative_cases_) > np.min(methods_min_depths) + 2*H) & flag_nonzero_recent &(nnz_recent >= 1): 
        # if there is enough data, use the predefinded extrapolation-forecasting method
        method_opt, name_opt, kwargs_optimum = methods_[0], names_[0], kwargs_list_[0] 
        method_opt_name = method_opt.__name__ 
        forecast = method_opt(cumulative_cases_, smoothed_dat=smoothed_hist[-1], **kwargs_optimum, H = H_)[-(H_ + 1):] 
    else:
        # not enough data for country  use the constant trend or zero trend
        comment = "very few cases (<= 10), unreliable forecast or no change in the data"
        print(comment)
        if len(smoothed_hist[-1])>H: 
            name_opt, method_opt_name, method_opt = "mean const trend", mean_const.__name__, mean_const
            kwargs_optimum = {'smooth': True}
            forecast = mean_const(smoothed_hist[-1], smoothed_dat=smoothed_hist[-1], H = H_, **kwargs_optimum)[-(H_ + 1):]   
        elif len(smoothed_hist[-1])>0:
            forecast = [smoothed_hist[-1][-1]] * (H_+1)
        else:
            forecast = [0] * (H_+1) 
        name_opt, method_opt_name, kwargs_optimum, method_opt = "no", "no", {}, mean_const

    smoothed, date_, sm_concat, date_delta = save_smoothed_df(country,date_,cumulative_cases,smoothed_hist,forecast,H,H_)   
    
    #-----------compute confidence intervals limits----------------------------------------------
    ci_full = compute_confidence(cumulative_cases_,cumulative_cases,date_,method_opt, kwargs_optimum, 
                       smoothing_fun, data_hist,smoothed_hist, country, names_, H,
                       saveplot, sm_concat, forecast, name_opt)
    
    #----------------------save data ------------------------------------------------------------     
    opt_method_info = [country, method_opt_name, kwargs_optimum]
    forecast_opt = [country, name_opt, method_opt_name] + list(np.diff(forecast)[-H:]) + date_delta[-H:] + [comment]
    return_dict[country] = [opt_method_info, forecast_opt, smoothed, ci_full]  
    if return_val:
        return return_dict, smoothed_hist, ci_full  
    
    
def compute_confidence(cumulative_cases_,cumulative_cases,date_,
                       method_opt, kwargs_optimum, 
                       smoothing_fun, 
                       data_hist, smoothed_hist, country, names_, H,
                       saveplot,sm_concat,forecast, name_opt, inc_=1):
    """
    compute confidence intervals  
    """
    ci_full = [] 
    history_for_conf_interval =  (inc_+1)*(20)+H+2
    if (len(cumulative_cases_)>20+H+3) and (len(cumulative_cases_)<history_for_conf_interval):
        inc_ = 0
        history_for_conf_interval = 20+H+2
    
    if len(cumulative_cases_)<history_for_conf_interval+1:
        return ci_full

    for i, error_ci in enumerate(["sqrt", "abs"]):
        ci_re = confidence_intervals(cumulative_cases_, method_opt, kwargs_optimum, smoothing_fun, 
                                     data_hist[-history_for_conf_interval:], smoothed_hist[-history_for_conf_interval:], 
                                     error_ci, H=H, inc_= inc_);   
        ci = convert_quantiles_to_ci(ci_re, sm_concat, H, error_ci, 10) #the last position is the smoothing coef  
        if saveplot:
            plot_last_forecast(cumulative_cases, sm_concat, 
                               ci, date_=date_, country=country, H=H, add_info = error_ci) 
        if i==0:
            ci_full = save_ci(ci, date_, country, H, error_ci)
        else:
            ci_full = pd.concat([ci_full, save_ci(ci, date_, country, H, error_ci)])
    return ci_full
    
    
    

def save_smoothed_df(country,date_,cumulative_cases,smoothed_hist, forecast,H,H_):
    """
    form a DataFrame with the output of forecast to prepare the results for dashboard
    """
    smoothed = pd.DataFrame()
    daily_forecast = np.diff(forecast[-(H_+1):]) 
    cumulative_forecast = np.cumsum(daily_forecast[-H:]) + cumulative_cases[-1]
    sm_cumulative_forecast = np.cumsum(daily_forecast) + smoothed_hist[-1][-1]  
    cases_n_forecast = np.array(list(cumulative_cases) +   list(cumulative_forecast))   
    
    sm_concat = np.array(list(smoothed_hist[-1])+list(sm_cumulative_forecast)) 
    sm_concat = simple_mirroring(sm_concat)  
    sm_cases_n_forecast = sm_concat  
    date_ = [np.datetime64(x, 'D') for x in date_]
    date_delta = [date_[-1] + np.timedelta64(i, 'D') for i in range(1, H_ + 1)]
    date_ = date_ + date_delta  
    
    smoothed["date"] = date_
    smoothed["daily"] = list(np.insert(np.diff(cumulative_cases),0,np.nan)) + list(daily_forecast[-H:])
    smoothed["cumul"] = cases_n_forecast

    smoothed["cumul_smoothed"] = sm_cases_n_forecast  
    smoothed["daily_smoothed"] = np.insert(np.diff(sm_concat),0,0) 
     
    observed_cumul_smoothed = np.append(smoothed_hist[-1], len(sm_cumulative_forecast) * [np.nan]) 
    smoothed["observed_smoothed"] = np.insert(np.diff(observed_cumul_smoothed),0,0)
    smoothed["observed"] = ["Observed"] * (len(cumulative_cases)) + ["Predicted"] * H
    smoothed["country"] = [country] * (len(date_))
    smoothed = smoothed.sort_values(by="date", ascending=False).reset_index(drop=True)

    return smoothed, date_, sm_concat, date_delta       


