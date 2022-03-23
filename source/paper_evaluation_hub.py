"""
scripts to generate relative MAE, WIS and coverage w.r.t. EU Covid 19 Forecast Hub baseline

used to generate Tables and histograms in SI (see notebooks with comparison with eu hub methods)
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
from paper_evaluation import get_df_limits
from compute_WIS_coverage import compute_WIS, compute_coverage
from paper_visualization import plot_forecast_baseline, plot_CI 


def load_hub_models(models, typedata, date_limits, df_forecasts, hub_data_folder):
    """
    load the forecasts and CI for the "models" which are saved in "hub_data_folder)"  
    within "date_limits" intersected with target dates from "df_forecasts"
    output:
    
    dict_methods_dfs:      dictionary with forecasts for each model
    models:                list of models excluding models with no data
    date_intersection_set: list of dates in the intersection of models submittiion dates
    """
    dict_methods_dfs = {}
    date_intersection_set =  set(df_forecasts["target_date"])
    exclude = [] 
    if typedata=="cases" or typedata=="deaths":
        if hub_data_folder[-2:]=="us":
            df_locations = pd.read_csv("https://raw.githubusercontent.com/reichlab/covid19-forecast-hub/master/data-locations/locations.csv", nrows=59)
            df_locations["country"] = df_locations["location_name"]

        if hub_data_folder[-2:]=="eu":
            df_locations = pd.read_csv("https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv")
            df_locations["country"] = df_locations["name"]
            df_locations["location"] = df_locations["alpha-2"] 
    for model in models:
        dict_methods_dfs[model] =  get_df_limits(hub_data_folder + "/"+ model +"/*", date_limits,[]) 
        dict_methods_dfs[model]["location"] = dict_methods_dfs[model]["location"].astype(str)
        dict_methods_dfs[model] =  dict_methods_dfs[model].merge(df_locations[["country","location"]].astype(str), 
                                                                 on=["location"], how="inner") 
        
        if len(dict_methods_dfs[model])>0:  
            date_intersection_set = date_intersection_set.intersection(set(dict_methods_dfs[model]["target_date"])) 
        else:
            exclude.append(model)
    models = list(set(models)-set(exclude))
    date_intersection_set = list(date_intersection_set) 
    return dict_methods_dfs, models, date_intersection_set




def evaluation_hub(country_list, date_limits=['2020-04-01','2021-03-07'],  
                   typedata="deaths", models = [],  path_results = "../data/paper",
                   H=7, raw_ground_truth=False, addon = "",baseline = "baseline_", datasource="JHU"):
    """
    for comparison of method from the dashboard and the forecasts of "models" submitted to EU Covid 19 Forecasting Hub
    compute ratios of errors (MAE and WIS) between methods and baseline from EU Covid 19 Forecast Hub
    output
    df_errors: DataFrame with errors for each method
    RI:        DataFrame with ratio of MAE of methods and baseline
    WIS:       DataFrame with ratio of WIS of methods and baseline
    coverage (see SI in the paper)
    """
    parse_column = "Country/Region"
    
    #Link to the data of 2022-01-10
    if datasource=="JHU":  
        if typedata=="deaths":
            path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/b849410bace2cc777227f0f1ac747a74fd8dc4be/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"     
        if typedata=="cases":
            path_data = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/b849410bace2cc777227f0f1ac747a74fd8dc4be/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"  
    folder_hub="hub_methods_eu"

    df = pd.read_csv(path_data)
    data_type = "/horison_"+str(int(H/7))+"_week/"  + datasource+ "_"+typedata +"/"  
    
    hub_data_folder = path_results + data_type +  folder_hub
    path_TrendForecast = path_results + data_type + "TrendForecast" + addon 
        
    #-------------------------------forecast of our model---------------------------------------------------
    df_forecasts = get_df_limits(path_TrendForecast+"/Point/*", date_limits,country_list) 
    df_forecasts = df_forecasts[(df_forecasts["target_date"]>=date_limits[0])&(df_forecasts["target_date"]<=date_limits[1])]
    df_forecasts_CI = get_df_limits(path_TrendForecast+"/CI/*", date_limits,country_list) 
    #-----------------------------fix confidence intervals--------------------------------------------------
    type_norm = ["abs","sqrt"]#"abs"#"sqrt"
    df_forecasts_CI = df_forecasts_CI[df_forecasts_CI["confidence_norm"].isin(type_norm)]
    df_forecasts_CI = df_forecasts_CI[(df_forecasts_CI["target_date"]>=date_limits[0])&(df_forecasts_CI["target_date"]<=date_limits[1])] 
    df_forecasts_CI["type"] = "day"
    #-------------------------------retrospective target for baseline estimation----------------------------
    df_target = get_df_limits(path_TrendForecast+"/Target/*", date_limits, country_list, add_days=H)
    #---------------------------------find the common dates from the hub submissions-------------------------
    dict_methods_dfs, models, date_intersection_set = load_hub_models(models,typedata, date_limits, df_forecasts, hub_data_folder)
    #--------------------------------------------------------------------------------------------------------
  
    models_ = ["forecast"] + models 
    models__ = ["forecast"+x for x in type_norm] + models 
    df_errors = pd.DataFrame(columns=["country","forecast_MAE","forecast_MedianAE",
                                      "baseline_MAE","baseline_MedianAE","coverage" ] + 
                             [model+"_MAE" for model in models] + [model+"_MedianAE" for model in models]) 
    
    model_wo_baseline = list(set(models_)-set([baseline]))
    model_wo_baseline_ = list(set(models__)-set([baseline]))
        
        
    RI = pd.DataFrame(columns=["country"] + [model+"_RIMAE" for model in model_wo_baseline] + [model+"_RIMedianAE" for model in model_wo_baseline]) 
   
    WIS = pd.DataFrame(columns=["country"]+[model+"WIS" for model in model_wo_baseline_])    
    coverage = pd.DataFrame(columns=["country"]+[model+"_coverage" for model in models__])  
    
    country_list_intersection = list(set(country_list).intersection(set(dict_methods_dfs[models[0]]["country"])))
    country_list_intersection = sorted(country_list_intersection)
    for numit, country in enumerate(country_list_intersection): 
        
        
        res = read_countries(df, [country], 0, datasource, typedata)
        cumulative_, date_ = res[0][0], res[1][0]
        cumulative_ = repair_increasing(cumulative_)  
        #-------------------------------target based on most recent report---------------------------------------
        target_now = pd.DataFrame()
        target_now["target_uptodate"] = pd.Series(np.diff(np.ravel(cumulative_))).rolling(H).mean()[7:]*7 
        target_now["target_cumul"] =  np.ravel(cumulative_)[7+1:]
        target_now["target_date"] = date_[7+1:]
        target_now["target_date"] =  target_now["target_date"].astype(str)
        #--------------------------------retrospective values for baseline.......................................
        country_target = df_target[df_target["country"]==country].sort_values(by="target_date").merge(target_now[["target_date","target_uptodate","target_cumul"]],on="target_date") 
        #------------------point forecast------------------------------------------------------------------------
        country_forecast = df_forecasts[df_forecasts["country"]==country].sort_values(by="target_date")
        country_forecast = country_forecast.merge(country_target[["target_date","target"]],on="target_date")
        country_forecast = country_forecast.merge(target_now[["target_date","target_uptodate"]],on="target_date")
        country_forecast["forecast"] = np.where(country_forecast["forecast"]>0,country_forecast["forecast"],0)
        country_forecast["fc_AE"] = np.abs(country_forecast["target_uptodate"]-country_forecast["forecast"])
     
        country_forecast = country_forecast.sort_values(by="target_date").reset_index()
        #------------------------CI---------------------------------------------------------------------------
        country_CI_ = df_forecasts_CI[df_forecasts_CI["country"]==country]
       
        country_CI_ = country_CI_[country_CI_["target_date"].isin(list(country_forecast["target_date"]))].reset_index().sort_values(by="target_date")
        country_CI_ = country_CI_.merge(country_target[["target_date","target"]], on="target_date")
        country_CI_ = country_CI_.merge(target_now[["target_date","target_uptodate"]], on="target_date")  
        
        country_CI = pd.DataFrame(sorted(np.unique(country_CI_["target_date"])),columns=["target_date"])

        
        for ci_type in type_norm:  
            df_ = country_CI_[(country_CI_["horizon"]=="w"+str(int(H/7))) &(country_CI_["confidence_norm"]==ci_type)].reset_index() 
            df_ = compute_WIS(df_, "forecast"+ci_type,"day") 
            df_ = compute_coverage(df_, "forecast"+ci_type,"day")  
            country_CI = country_CI.merge(df_[["forecast"+ci_type+"WIS","forecast"+ci_type+"_coverage", "target_date"]], on=["target_date"])
        
        #----------------------------------------------------------------------------------
        if numit in range(0):  #change from 0 to some number to plot forecasts
            ax = plot_forecast_baseline(country,country_forecast,H)
            plot_CI(country, df_, country_forecast,H) 
      
        #-------------------------compute errors for hub models----------------------------
        hub_models = pd.DataFrame()
        hub_models["target_date"] = date_intersection_set
        #print(country)
        for model in models: 
            df_ = dict_methods_dfs[model] 
            if df_["type"][0] == "day":
                target_col = "target_uptodate"
            elif df_["type"][0] == "cumulative":
                target_col = "target_cumul"
 
            model_country_df = df_[df_["country"]==country].merge(target_now[["target_date",target_col]], on="target_date")
            model_country_df =  model_country_df[model_country_df["target_date"].isin(hub_models["target_date"])]  
            model_country_df[model+"AE"] = np.abs(model_country_df["forecast_"+model]-model_country_df[target_col])
 
            if model_country_df.shape[0]==0:
                print(model," does not support", country) 
            try:
                model_country_df = compute_WIS(model_country_df, model, df_["type"][0]) 
                model_country_df = compute_coverage(model_country_df, model, df_["type"][0]) 
            except:
                print("WIS failed for ", model)
                print(model_country_df)
            if numit in range(0): #change from 0 to some number to plot forecasts
                 if ("ensemble" in model) or ("baseline" in model):     
                    model_country_df = model_country_df.sort_values(by=["target_date"]).reset_index(drop=True)
                    ax.plot(model_country_df["target_date"],model_country_df["forecast_"+model], alpha=0.5) 

            hub_models = hub_models.merge(model_country_df[[model+"AE",model+"WIS",model+"AE",model+"_coverage","target_date"]], on="target_date",how="inner") 
        country_target["bl_AE"] = np.nan 
        

        #-----------------------------start country-wise error calculation------------------
        if country_target.shape[0]>H: 
            country_baseline_forecast = country_target.copy()
            country_baseline_forecast["target_date"] = pd.to_datetime(country_baseline_forecast["target_date"]) + timedelta(days=H)
            country_baseline_forecast["target_date"] = country_baseline_forecast["target_date"].astype(str)
            country_baseline_forecast =country_baseline_forecast.rename(columns={"target":"baseline_forecast"}) 
            country_target = country_target.merge(country_baseline_forecast[["target_date","baseline_forecast"]], on=["target_date"], how="inner") 
            country_target["bl_AE"] =  np.abs(country_target["target_uptodate"]-country_target["baseline_forecast"])  
           
            #---------------Merge all the methods AE results-------------------------------- 
 
            AE  = pd.merge(country_target[["bl_AE","target_date"]],
                                   country_forecast[["fc_AE","target_date"]],
                                   on = "target_date", how="inner").merge(hub_models, on = "target_date", how="inner").dropna(axis=0) 
            AE["cover"] = 1*(AE["bl_AE"]>AE["fc_AE"]) 
            errors = pd.DataFrame([[country, np.mean(AE["fc_AE"]),np.nanmedian(AE["fc_AE"]),
                                    np.nanmean(AE["bl_AE"]), np.nanmedian(AE["bl_AE"]), 
                                    np.mean(AE["cover"])] + [np.nanmean(AE[model+"AE"]) for model in models] + 
                                   [np.nanmean(AE[model+"AE"]) for model in models]],
                                    columns = df_errors.columns)
            df_errors = df_errors.append(errors, ignore_index=True)
            #--------------------------------Relative improvement----------------------------
            res = pd.DataFrame() 
            for err in ["MAE","MedianAE"]:
                error_bl = errors[baseline+"_"+err].values 
                for model in model_wo_baseline:    
                    error_model = errors[model+"_"+err].values
                    res[model+"_RI"+err] = error_model/(1+error_bl)   #(error_bl-error_model)/(1+error_bl)   
            ri_0 = errors["coverage"].values[0]
            res["country"] = country     
            RI = RI.append(res,ignore_index=True) 
            #--------------------------------WIS----------------------------------------------
            
            hub_models = hub_models.merge(country_CI, on=["target_date"], how="inner")
            error_bl = np.mean(hub_models[baseline+"WIS"].values)
           
            re_wis = [np.mean(hub_models[model+"WIS"])/(error_bl+1) for model in model_wo_baseline_] #[(error_bl-np.mean(hub_models[model+"WIS"]))/(error_bl+1) for model in model_wo_baseline_]
            WIS = WIS.append(pd.DataFrame([[country]+ re_wis],
                                          columns=WIS.columns), ignore_index=True) 
            coverage = coverage.append(pd.DataFrame([[country]+[np.mean(hub_models[model+"_coverage"]) for model in models__]],
                                          columns=coverage.columns), ignore_index=True) 
    
    return df_errors, RI, WIS, coverage



    
 
    