"""
a function to draw the bar plots in SI for evaluation based on the cases data from JHU

""" 
import more_itertools
import datetime
from datetime import timedelta
import os
import json
import pandas as pd
import numpy as np
import glob

import matplotlib.pyplot as plt
 
from precomputing import read_countries, repair_increasing
from compute_WIS_coverage import compute_WIS, compute_coverage
from paper_visualization import plot_forecast_baseline, plot_CI 

def substring_index(l, substr):  
    """
    select index of substring substr in list l 
    """
    index = [idx for idx, s in enumerate(l) if substr in s]
    return index 

def substring_two_indeces(l,substr):
    """
    find indices of the first and second element in substing in list l 
    """
    substr1, substr2 = substr[0], substr[1]
    index1 = substring_index(l,substr1)  
    index2 = substring_index(l,substr2) 
    if len(index1)==0:
        index1 = 0
    else:
        index1 = index1[0]
    if len(index2)==0:
        index2 = len(l)-1
    else:
        index2 = index2[0]
    
    return index1, index2 

def get_df_limits(dir_, date_limits, country_list, add_days=0):
    """
    merge forecasts (CI) from the path in "dir_" within the limits in "date_limits"
    for countries in "country_list"
    Output:
    DataFrame with forecasts (CI)
    
    """
    paths = sorted(glob.glob(dir_)) 
    date_limit_2 = datetime.datetime.strptime(date_limits[1], "%Y-%m-%d") 
    date_limits_1 = datetime.datetime.strptime(date_limits[0], "%Y-%m-%d")+ timedelta(days=-add_days)
    start_f,end_f = substring_two_indeces(paths, [str(date_limits_1)[:10], str(date_limit_2)[:10]])
 
    paths = paths[start_f:end_f+1]  
    df = pd.read_csv(paths[0])
    if len(country_list)>0:
        df = df[df["country"].isin(country_list)]
    for path in paths[1:]:
        df_ = pd.read_csv(path)
        if len(country_list)>0:
            df_ = df_[df_["country"].isin(country_list)]
        df = df.append(df_,ignore_index=True)
        
    return df 


def evaluation(country_list, date_limits=['2020-04-01','2021-03-07'], path_data=[], datasource="JHU",typedata="cases",addon="",
               path_results = "../data/paper/", H=7, raw_ground_truth=False):
    """
    run the evaluations based on the forecasts saved in path_results folder 
    computes the RMAE, RmedianAE, RC and RWIS for the forecasting methodology in the dashboard compared to the constant forecast baseline
    for the list of countries in country_list
    """
 
    path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/b849410bace2cc777227f0f1ac747a74fd8dc4be/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"  #Link to the data of 2022-01-10
    datasource, parse_column = "JHU", "Country/Region"       
    df = pd.read_csv(path_data)
    data_type = "horison_"+str(int(H/7))+"_week/"  + datasource+ "_"+typedata +"/"   
    path_TrendForecast = path_results + data_type + "TrendForecast" + addon 
        
    #-------------------------------forecast of our model---------------------------------------------------
   
    df_forecasts = get_df_limits(path_TrendForecast+"/Point/*", date_limits, country_list) 
    df_forecasts = df_forecasts[(df_forecasts["target_date"]>=date_limits[0])&(df_forecasts["target_date"]<=date_limits[1])]
    df_target = get_df_limits(path_TrendForecast+"/Target/*", date_limits, country_list, add_days=H)
    df_forecasts_CI = get_df_limits(path_TrendForecast+"/CI/*", date_limits,country_list) 
    df_errors = pd.DataFrame(columns=["country","forecast_MAE","forecast_MedianAE","baseline_MAE","baseline_MedianAE","coverage"])

    #-----------------------------fix confidence intervals--------------------------------------------------
    type_norm = ["sqrt"] 
    df_forecasts_CI = df_forecasts_CI[df_forecasts_CI["confidence_norm"].isin(type_norm)]
    df_forecasts_CI = df_forecasts_CI[(df_forecasts_CI["target_date"]>=date_limits[0])&(df_forecasts_CI["target_date"]<=date_limits[1])] 
    df_forecasts_CI["type"] = "day"
    #--------------------------------------------------------------------------------------------------------------
    RI = pd.DataFrame(columns=["country","$RI_{MAE}$","$RI_{MedianAE}$","$RI_old0$"])
        
    WIS =  pd.DataFrame(columns=["country","forecast_WIS","baseline_WIS"])  
    coverage = pd.DataFrame(columns=["country","forecast","baseline"])  
    for numit, country in enumerate(country_list):
        try:
            res = read_countries(df, [country], 0, datasource, typedata)
        except:
            print(country)
        cumulative_, date_ = res[0][0], res[1][0] 
        cumulative_ = repair_increasing(cumulative_)  
        target_now = pd.DataFrame()
        target_now["target_uptodate"] = pd.Series(np.diff(np.ravel(cumulative_))).rolling(H).mean()[H:]*H  
        target_now["target_date"] = date_[H+1:]
        target_now["target_date"] =  target_now["target_date"].astype(str)
        country_forecast = df_forecasts[df_forecasts["country"]==country].sort_values(by="target_date")
        country_target = df_target[df_target["country"]==country].sort_values(by="target_date").merge(target_now[["target_date","target_uptodate"]],on="target_date") 
        country_forecast = country_forecast.merge(country_target[["target_date","target"]],on="target_date")
        country_forecast = country_forecast.merge(target_now[["target_date","target_uptodate"]],on="target_date")
        country_forecast["fc_AE"] = np.abs(country_forecast["target_uptodate"]-country_forecast["forecast"])
        country_target["bl_AE"] = np.nan  
        
        if country_target.shape[0]>H: 
 
            country_baseline_forecast = country_target.copy()
            country_baseline_forecast["target_date"] = pd.to_datetime(country_baseline_forecast["target_date"]) + timedelta(days=H)
            country_baseline_forecast["target_date"] = country_baseline_forecast["target_date"].astype(str)
            country_baseline_forecast =country_baseline_forecast.rename(columns={"target":"baseline_forecast"}) 
            country_target = country_target.merge(country_baseline_forecast[["target_date","baseline_forecast"]], on=["target_date"], how="inner") 
            country_target["bl_AE"] =  np.abs(country_target["target_uptodate"]-country_target["baseline_forecast"])  
            
            
            AE = pd.merge(country_target[["bl_AE","target_date","target_uptodate"]],
                                   country_forecast[["fc_AE","target_date"]],
                                   on = "target_date", how="inner")
            AE = AE.dropna(axis=0).reset_index(drop=True) 
            CI_baseline = compute_baseline_quantiles(AE[["target_date","bl_AE"]], AE["target_uptodate"], H=H)  
            CI_baseline["target_date"] =[0]*H + list(CI_baseline["target_date"][:-H])
            CI_baseline = CI_baseline.iloc[H:]  
            CI_baseline = CI_baseline.merge(country_target[["target_uptodate","target_date"]],on=["target_date"])
 
            
            country_CI_ =  df_forecasts_CI[df_forecasts_CI["country"]==country]   
            country_CI_ = country_CI_[country_CI_["target_date"].isin(list(country_forecast["target_date"]))].reset_index().sort_values(by="target_date") 
            country_CI_ = country_CI_.merge(country_target[["target_date","target"]], on="target_date") 
            country_CI = country_CI_.merge(target_now[["target_date","target_uptodate"]], on="target_date")   
            
            
            wis_forecast = compute_WIS(country_CI, "forecast","day") 
            wis_baseline =  compute_WIS(CI_baseline, "baseline","day")
            coverage_forecast = compute_coverage(country_CI, "forecast","day") 
            coverage_baseline =  compute_coverage(CI_baseline, "baseline","day") 

            WIS = WIS.append(pd.DataFrame([[country]+ [np.mean(wis_forecast["forecastWIS"]),np.mean(wis_baseline["baselineWIS"])]],
                                          columns=WIS.columns), ignore_index=True) 
            
            coverage = coverage.append(pd.DataFrame([[country] + [np.mean(coverage_forecast["forecast_coverage"]),
                                                                 np.mean(coverage_baseline["baseline_coverage"])]],
                              columns=coverage.columns), ignore_index=True) 
            AE["cover"] = 1*(AE["bl_AE"]>AE["fc_AE"]) 
            
            errors = pd.DataFrame([[country, np.mean(AE["fc_AE"]),np.nanmedian(AE["fc_AE"]),
                                    np.nanmean(AE["bl_AE"].values), np.nanmedian(AE["bl_AE"].values), np.mean(AE["cover"])]],
                                    columns = df_errors.columns)
            df_errors = df_errors.append(errors,ignore_index=True)
    
            ri_mean = (errors["baseline_MAE"].values-errors["forecast_MAE"].values)/(1+errors["baseline_MAE"].values)
            ri_median = (errors["baseline_MedianAE"].values-errors["forecast_MedianAE"].values)/(1+errors["baseline_MedianAE"].values)
            ri_0 = errors["coverage"].values[0]
            RI = RI.append(pd.DataFrame([[country, ri_mean[0], ri_median[0], ri_0]], columns = RI.columns),ignore_index=True)    
    
    return RI, WIS, coverage
    
def compute_baseline_quantiles(AE, forecast, H=7):
    """
    compute empirical quantiles for the forecast by baseline
    motivated by https://github.com/reichlab/covidModels/blob/master/R-package/R/quantile_baseline.R

    """
    AE = AE.sort_values(by="target_date").reset_index(drop=True) 
    quantiles = [0.01,0.025] + list(np.arange(0.05,1,0.05)) + [0.975,0.99] 
    ind_median = int(len(quantiles)/2)+1
    col_names = [str(round(q,3)) for q in quantiles]+["target_date"] 
    q_AE = pd.DataFrame(columns=col_names) 
    for i in range(2,AE.shape[0]):
        vals = AE["bl_AE"].values[:i]
        vals = list(vals) + list(-vals)
        quant = np.quantile(vals, quantiles) 
        quant = quant - quant[ind_median] + forecast[i-1] 
        q_ = [list(quant)+ [AE["target_date"].values[i-1]]] 
        q_AE = q_AE.append(pd.DataFrame(q_, columns=col_names)) 
    return q_AE     
            
        
        
        
def evaluation_AE(country_list, date_limits=['2020-04-01','2021-03-07'], path_data=[], datasource="JHU",typedata="cases",addon="",
               path_results = "../paper/", H=7, raw_ground_truth=False):
    """
    run the evaluations to obtain absolute deviations of TrendModel and baseline needed for slope computation (see the notebook notebooks/paper_error_vs_slope_Fig_SI4.ipynb)   
    """
 
    path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/b849410bace2cc777227f0f1ac747a74fd8dc4be/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"  #Link to the data of 2022-01-10
    datasource, parse_column = "JHU", "Country/Region"       
    df = pd.read_csv(path_data)
    data_type = "horison_"+str(int(H/7))+"_week/"  + datasource+ "_"+typedata +"/"   
    path_TrendForecast = path_results + data_type + "TrendForecast" + addon 
        
    #-------------------------------forecast of our model--------------------------------------------------- 
    df_forecasts = get_df_limits(path_TrendForecast+"/Point/*", date_limits, country_list) 
    df_forecasts = df_forecasts[(df_forecasts["target_date"]>=date_limits[0])&(df_forecasts["target_date"]<=date_limits[1])]
    df_target = get_df_limits(path_TrendForecast+"/Target/*", date_limits, country_list, add_days=H)  
 
    stl,baseline,target = {},{},{}
    
    for numit, country in enumerate(country_list):
        
        res = read_countries(df, [country], 0, datasource, typedata)
        cumulative_, date_ = res[0][0], res[1][0] 
        cumulative_ = repair_increasing(cumulative_)  
        target_now = pd.DataFrame()
        target_now["target_uptodate"] = pd.Series(np.diff(np.ravel(cumulative_))).rolling(H).mean()[H:]*H  
        target_now["target_date"] = date_[H+1:]
        target_now["target_date"] =  target_now["target_date"].astype(str)
        country_forecast = df_forecasts[df_forecasts["country"]==country].sort_values(by="target_date")
        country_target = df_target[df_target["country"]==country].sort_values(by="target_date").merge(target_now[["target_date","target_uptodate"]],on="target_date") 
        country_forecast = country_forecast.merge(country_target[["target_date","target"]],on="target_date")
        country_forecast = country_forecast.merge(target_now[["target_date","target_uptodate"]],on="target_date")  
        
        country_baseline_forecast = country_target.copy()
        country_baseline_forecast["target_date"] = pd.to_datetime(country_baseline_forecast["target_date"]) + timedelta(days=H)
        country_baseline_forecast["target_date"] = country_baseline_forecast["target_date"].astype(str)
        country_baseline_forecast = country_baseline_forecast.rename(columns={"target":"baseline_forecast"})  
        country_target = country_target.merge(country_baseline_forecast[["target_date","baseline_forecast"]], on=["target_date"], how="inner")  

        country_df = country_target.merge(country_forecast[["target_date","forecast"]], on=["target_date"], how="inner") 
        country_df.index = country_df["target_date"] 
        stl[country] = country_df[["forecast"]]
        baseline[country] = country_df[["baseline_forecast"]]  
        target[country] = country_df[["target_uptodate"]]  
     
    return stl, baseline, target
       
