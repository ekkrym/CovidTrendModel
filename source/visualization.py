import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os
from scipy.interpolate import UnivariateSpline
from precomputing import compute_method_depth

plt.rcParams['axes.facecolor'] = 'white'
from smoothing import simple_mirroring
from precomputing import precompute_smoothing, precompute_forecasts

if not os.path.exists("pics/"):
    os.makedirs("pics/")

    
    
def plot_last_forecast(allcases, smoothed_dat, ci,
                       date_=[], country="", H=7, add_info = ""):
    """
    Plotting daily cases predictions and confidence intervals in one subplot
    and confidence interval for weekly forecast on horizon int(H/7) in another
     
    allcases:     cumulative cases
    smoothed_dat: contains smooth trend and forecast (in the last H positions)
    ci:           DataFrame with 21 CI esimates
    country: country or region
    date_:        dates corresponding to allcases
    H:            forecasts length
    add_info:     optional addition to the name of the figure 
    """ 
    smoothed_forecast = np.diff(smoothed_dat)[-H:]
    len_hist = len(allcases)
    start_plot = len(allcases)-31
    green = "c"
    custom_lines = [Line2D([0], [0], color=green, lw=2, alpha = 0.5),
                Line2D([0], [0], color="blue", lw=2),
                Line2D([0], [0], color="red", lw=2, alpha = 0.5),
                Line2D([0], [0], color="black", lw=2, alpha = 0.5)] 

    fig, (ax1, ax2) = plt.subplots(2,  figsize=(15, 7),dpi=600) 
    ax1.bar(range(start_plot, len_hist), np.diff(allcases, axis=0)[start_plot-1:], color = green, alpha = 0.5) 
    ax1.plot(range(start_plot, len(smoothed_dat)-H), np.diff(smoothed_dat, axis=0)[start_plot-1:-H], c="blue") 
    ax1.bar(range(len_hist,len_hist+H), list(smoothed_forecast), color = "red",  alpha = 0.3)  
    
    #------plot quantiles--------------------------------------------------------   
    qn = len(ci.columns[1:]) 
    q_to_show = ci.columns[1:int((qn-1)/2)+1] 
    for i,q in enumerate(q_to_show):

        m1q = str(1-float(q))  
        qlower = ci[q].astype(float).values[:H]
        qupper = ci[m1q].astype(float).values[:H]
        ax1.fill_between(range(len_hist, len_hist + H), 
                         qlower, qupper, 
                         alpha=0.1,color="b")
        if i%4==0:
            ax1.text(len_hist + H -0.8, qlower[-1], q, fontsize=7)
            ax1.text(len_hist + H -0.8, qupper[-1], m1q, fontsize=7) 

    med = ci["0.5"].values[:H]  
    ax1.plot(range(len_hist, len_hist + H), med, c = "black",  alpha = 0.7, lw=3)
    #---------------------------------------------------------------------------         
    legend = ["daily data","stl", "forecast", "forecast+ q(AE,0.5)"]

    add_info_=""
    ax1.set_title(country+" "+add_info_)
    ax1.grid()
    ax1.set_ylabel("daily cases", fontsize=10); 
    ax1.legend(custom_lines, legend, fontsize=10, loc = 'upper left') 

    if len(date_) > 0: 
        x = range(len(date_[start_plot+1:]),start_plot,-H)
        ax1.set_xticks(x)
        ax1.set_xticklabels(np.array([date_[i] for i in x]), fontsize=8)
        for label in ax1.get_xticklabels():
            label.set_rotation(60)
            label.set_horizontalalignment('right') 
        
    qweakly = ci.values[-1,1:]   
  
    ax2.bar(ci.columns[1:], qweakly, color = "blue", alpha = 0.5)
    ax2.bar(["0.5"], qweakly[11], color = "black", alpha = 0.5)
    ax2.legend(["weekly q", "weekly forecasted"])
    ax2.grid()  
 
    plt.savefig('pics/' + country + "_forecast_" + "CI_smooth_" +"H"+str(H)+add_info+".pdf") 


        
        
def plot_last_forecast_like_dashboard(allcases, smoothed_dat, ci,
                       date_=[], country="", H=7, add_info = ""):
    """
    
    Figure 2 in the paper
    Plotting daily cases and CI in one picture
    
    allcases:     cumulative cases
    smoothed_dat: contains smooth trend and forecast (in the last H positions)
    ci:           DataFrame with 21 CI esimates
    country: country or region
    date_:        dates corresponding to allcases
    H:            forecasts length
    add_info:     optional addition to the name of the figure 
    """ 
    smoothed_forecast = np.diff(smoothed_dat)[-H:]
    len_hist = len(allcases)
    start_plot = len(allcases)-31
    green = "#66C3A6"
    blue = "#8EA2CC"
    custom_lines = [Line2D([0], [0], color=green, lw=2, alpha = 0.9),
                Line2D([0], [0], color=blue, lw=2),
                Line2D([0], [0], color="red", lw=2, alpha = 0.9),
                Line2D([0], [0], color="black", lw=2, alpha = 0.9)] 
    cc  ="#404040"  
    with plt.rc_context({'axes.edgecolor':cc, 'xtick.color':cc, 'ytick.color':cc, 'figure.facecolor':'white'}):
       
        fig, ax1 = plt.subplots(1,  figsize=(5, 2),dpi=300)  
        ax1.bar(range(start_plot, len_hist), np.diff(allcases, axis=0)[start_plot-1:], color = green, alpha = 0.9) 
        ax1.plot(range(start_plot, len(smoothed_dat)-H), np.diff(smoothed_dat, axis=0)[start_plot-1:-H], c="red") 
        ax1.bar(range(len_hist,len_hist+H), list(smoothed_forecast), color = "blue",  alpha = 0.3) 

       #------plot quantiles--------------------------------------------------------  

        qn = len(ci.columns[1:]) 
        q_to_show = ci.columns[1:int((qn-1)/2)+1][::2] 
        for i,q in enumerate(q_to_show): 
            m1q = str(1-float(q))  
            qlower = ci[q].astype(float).values[:H]
            qupper = ci[m1q].astype(float).values[:H]
            ax1.fill_between(range(len_hist, len_hist + H), 
                             qlower, qupper, 
                             alpha=0.1,color="red")
            if True:#i%4==0:
                if i%3==0:
                    ax1.text(len_hist + H -0.8, qlower[-1], q, fontsize=7)

                    ax1.text(len_hist + H -0.8, qupper[-1], m1q, fontsize=7) 

        med = ci["0.5"].values[:H]   

        ax1.plot(range(len_hist, len_hist + H), med, c = "red")
        ax1.set_frame_on(False)
        ax1.axes.get_xaxis().set_visible(True) 
        #---------------------------------------------------------------------------         
        legend = ["daily data","stl", "forecast", "forecast+ q(AE,0.5)"]

        add_info_="" 
        if len(date_) > 0: 
            x = np.arange(len(date_)-1,start_plot,-2*H)
            print(x)
            ax1.set_xticks(x)
            ax1.set_xticklabels(np.array([str(date_[i]) for i in x]), fontsize=8) 

        qweakly = ci.values[-1,1:] 

        plt.savefig('pics/' + country + "_forecast_" + "CI_smooth_" +"H"+str(H)+add_info+".pdf") 


