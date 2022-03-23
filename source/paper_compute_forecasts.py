 
"""
To run the retrospective forecasts based on the data available 
at each day between  "2020-04-01"  and "2021-12-15" 
The results are saved in  TrendForecast folders in the folders 
data/paper/horison_*i*_week/JHU_*type*/ where i=1,2, type is "cases" or "deaths" 

python source/paper_compute_forecasts.py 
"""
    
import multiprocessing
import os 
import time
import warnings 
import getopt, sys
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import time 
import datetime
from datetime import date, timedelta
#os.chdir("source/")
sys.path.append('methods_configuration/')

from precomputing import read_countrydata, read_countries, repair_increasing
from smoothing import piecewise_STL,simple_mirroring 
from forecast_one_country import forecast_one_country
from function_configuration import METHODS  
from visualization import plot_last_forecast
from countries_preselected import countries_JHU, countries_US 

def parse_col_links(data_col):
    if "global" in data_col:
        datasource, parse_column = "JHU", "Country/Region" 
    elif "US" in data_col:
        datasource, parse_column = "JHU_US", "Province_State" 
    if ("confirmed" in data_col) or ("cases" in data_col):
        typedata = "cases"
    else:
        typedata = "deaths"
    return datasource, parse_column, typedata

def get_methods_params(subset,call_from_notebook=False):
    #---------------------------forecasting method selection----------------------------------
    methods_df_path = "methods_configuration/selected_methods_const.csv"
    if call_from_notebook:
        methods_df_path = "../methods_configuration/selected_methods_const.csv"
    methods_pd = pd.read_csv(methods_df_path) 
    methods = [METHODS[x] for x in list(methods_pd["method"].values[subset])]
    kwargs_list = [eval(x) for x in list(methods_pd["parameters"].values[subset])]
    names = list(methods_pd["description"].values[subset])
    cv_flag = list(methods_pd["cv"].astype(int).values[subset])
    return methods, kwargs_list, names, cv_flag

def check_existance(folder):
    if not os.path.isdir(folder): 
        os.makedirs(folder)  

def create_folders(save_folder,hor_forecast,data_type):
    save_path = ""
    for add_ in [save_folder,hor_forecast,data_type]: 
        save_path = save_path + add_
        check_existance(save_path) 
    for add_ in ["Target","CI","Point"]:
        check_existance(save_path+add_) 
    return save_path 

def extrapolate_linearly_weekly(daily_smoothed):
    data_smoothed = H*pd.Series(daily_smoothed[:-H]).rolling(H).mean().values
    if data_smoothed[-1]-data_smoothed[-H]>0:
        weekly_forecast = data_smoothed[-1]+(data_smoothed[-1]-data_smoothed[-H])
    else:
        weekly_forecast = np.exp(np.log(data_smoothed[-1])+(np.log(data_smoothed[-1])-np.log(data_smoothed[-H])))
        weekly_forecast = np.where(weekly_forecast>0,weekly_forecast,0)
    return weekly_forecast 


def save_retrospective_predictions(date, data_col, countries, subset, 
                           smoothing_fun, H=7, call_from_notebook=False, save_folder="data/paper/", typedata="",method_addon=""): 
   
    #--------------prepare df's for saving---------------------------------------------  
    forecast = pd.DataFrame(columns = ["country", "target_date", "forecast"]) 
    q = [0.01,0.025] + list(np.arange(0.05,1,0.05)) + [0.975,0.99]
    q = [str(round(x,3)) for x in q]
    forecast_ci = pd.DataFrame(columns = q + ["date"] + ["confidence_norm","country"])
    target = pd.DataFrame(columns = ["country","target_date", "target"]) 
    hor_forecast = "horison_"+ str(int(H/7))+"_week/" 
    data_type = datasource+ "_"+typedata+"/TrendForecast"+method_addon+"/" 
    path_save = create_folders(save_folder,hor_forecast,data_type)  
  
    #------------------load retrospective data------------------------------------------
    print(date, " started") 
    link_for_forecast  = retro_data_info[retro_data_info["date"]==str(date)[:10]][data_col].values[0]
    target_date= date + timedelta(days=H)
    link_for_target  = retro_data_info[retro_data_info["date"]==str(target_date)[:10]][data_col].values[0]
    df_for_forecast = pd.read_csv(link_for_forecast)   
    df_for_target = pd.read_csv(link_for_target)   
    countries_intersection = set(df_for_forecast[parse_column].unique()).intersection(countries)
    cumulative_list, date_list = read_countries(df_for_forecast, countries_intersection, 0, datasource, typedata=typedata) 
    cumulative_target, date_target = read_countries(df_for_target, countries_intersection, 0, datasource, typedata=typedata) 
    #------------------------------------------------------------------------------------ 
    
    for country, cumulative_cases, cumulative_target_, dates_,dates_target_ in zip(countries_intersection, cumulative_list, cumulative_target, date_list, date_target):   
        cumulative_cases_ = repair_increasing(cumulative_cases) 
        cumulative_target_ = repair_increasing(cumulative_target_)  
        return_dict, _, ci_full = forecast_one_country({}, country, cumulative_cases, dates_,
                                                       methods, kwargs_list, names, smoothing_fun, 
                                                       datasource, H=H, return_val=True, saveplot=False, newly=True) 
        #---------point forecast---------------------------------
        smoothed = np.ravel(return_dict[country][-2]["daily_smoothed"].values[::-1].copy()) 
        weekly_forecast = np.sum(smoothed[-7:])
        
        target_date = return_dict[country][-2]["date"].values[0]  
        forecast = forecast.append(pd.DataFrame([[country, target_date, weekly_forecast]], columns=forecast.columns), ignore_index=True)
        #---------"probabilistic" forecast------------------------
        if len(ci_full)>0:
            forecast_ci = forecast_ci.append(ci_full[ci_full["horizon"]=="w"+str(int(H/7))], ignore_index=True) 
        #---------target------------------------------------------ d
        daily_target = np.diff(cumulative_target_)
      #  weekly_target = np.sum(daily_target[-7:]) 
        weekly_target = 7*np.median(daily_target[-7:]) 
        target_date_ = dates_target_[-1]
        target = target.append(pd.DataFrame([[country, target_date_, weekly_target]], columns=target.columns), ignore_index=True)   
    

    target_path = path_save + "Target/Target"+"_"+str(H)+"_" + str(date)[:10]+ ".csv"  
    forecast_ci_path =  path_save +  "CI/Forecast_CI_h" + str(H)+"_TrendModel_" + str(date)[:10]+".csv" 
    forecast_path =  path_save + "Point/Forecast_point_h"+ str(H)+"_TrendModel_" + str(date)[:10]+ ".csv" 
    target.to_csv(target_path)
    forecast_ci.rename(columns={"date": "target_date"}).to_csv(forecast_ci_path)
    forecast.to_csv(forecast_path)
    print(date, " finished")

    
if __name__ == '__main__':
    
    #------setup--------------------------------------------------
    nthreads = 10#0
    starting_time = "2020-04-01" 
    end_time = "2021-12-15" 
    #-------------------------------------------------------------
    start_date = datetime.datetime.strptime(starting_time, "%Y-%m-%d") 
    end_date = datetime.datetime.strptime(end_time, "%Y-%m-%d")
    dates =  pd.date_range(start_date, end_date, freq='d') 
    methods,kwargs_list,names,cv_flag = get_methods_params([3],False)  
    retro_data_info = pd.read_csv("data/JHU_data_links.csv")   
    retro_data_info.columns = ["date"] + list(retro_data_info.columns)[1:]
    print(retro_data_info)
    #-----------------full scenario---------------------------------
    horisons = [7,14] 
    data_cols = ["global-confirmed", "US-deaths"] 
    countries_list = [countries_JHU, countries_US] 
        
    smoothing_funcs = {"cases": [(piecewise_STL,"")],
                       "deaths":[(piecewise_STL,"")]}    
 
    # -----------------------scenario JHU deaths----------------
    #horisons = [7,14] 
    #data_cols = ["global-deaths"] 
    #countries_list = [countries_JHU]
    #smoothing_funcs = {"deaths":[(piecewise_STL,"")]}    
    # ----------------------------------------------------------------
    processes = [] 
    manager = multiprocessing.Manager()  
    counter = 0  
    
    for H in horisons:
        for data_col, countries in zip(data_cols, countries_list): 
            datasource, parse_column, typedata = parse_col_links(data_col) 
            for (smoothing_fun, method_addon) in smoothing_funcs[typedata]:
                for ii,date in enumerate(dates):
                    counter+=1
                    p = multiprocessing.Process(
                            target=save_retrospective_predictions,
                            args=(date, data_col, countries, [3], 
                                  smoothing_fun, H, False, "data/paper/", typedata, method_addon)) 
                    processes.append(p)
                    p.start()
                    if counter%nthreads==0:
                        for process in processes:
                            process.join() 
                        processes = []
                        manager = multiprocessing.Manager()  
            
    for process in processes:
        process.join() 