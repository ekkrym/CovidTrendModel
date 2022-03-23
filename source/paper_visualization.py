import os
import json
import pandas as pd
import numpy as np
import glob
import seaborn as sns  
import matplotlib.pyplot as plt 

def plot_forecast_baseline(country,country_forecast,H):
    """
    plot a baseline forecast - optional function for evaluation 
    """
    fig,ax = plt.subplots(1,1,figsize=(10,3))
    len_  = len(country_forecast["target_uptodate"])
    
    ax.plot(country_forecast["target_date"],country_forecast["forecast"], color="b",alpha=0.6,lw=2)
    ax.plot(country_forecast["target_date"][H:],country_forecast["target"].values[:-H],color="c",alpha=0.6,lw=2) 
    ax.bar(country_forecast["target_date"], country_forecast["target_uptodate"], color="b",alpha=0.3)
    ax.bar(country_forecast["target_date"], country_forecast["target"], color="r",alpha=0.3)
    ax.legend(["forecast","baseline","target_new", "target_retro"])
     
    ax.set_title(country)
    return ax

     
def plot_CI(country, CI, country_forecast, H, ci_type="sqrt"):
    """
    plot a particular country forecast and CI - optional function for evaluation 
    """
    fig,ax = plt.subplots(1,1,figsize=(10,3))
    len_  = len(country_forecast["target_uptodate"])
    ax.plot(country_forecast["target_date"],country_forecast["forecast"], color="b",alpha=0.6,lw=2)
    #CI = country_CI_.sort_values(by="target_date").reset_index()#country_CI_[country_CI_["confidence_norm"]==ci_type]
    
    ax.plot(CI["target_date"],CI["0.5"].values, color="black",alpha=0.6,lw=2)
    ax.bar(country_forecast["target_date"], country_forecast["target_uptodate"], color="b",alpha=0.3)
    ax.bar(country_forecast["target_date"], country_forecast["target"], color="r",alpha=0.3)
 
    ax.legend(["forecast","forecast_corr","target_new", "target_retro"])

    for i,q in enumerate(["0.01","0.05","0.25"]): 
        m1q = str(1-float(q))   
        qlower = CI[q].astype(float).values
        qupper = CI[m1q].astype(float).values
        ax.fill_between(CI["target_date"], 
                         qlower, qupper, 
                         alpha=0.1,color="b")   
 
    ax.set_title(country+ " CI "+ci_type + " based")
    
 