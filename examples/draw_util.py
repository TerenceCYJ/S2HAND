import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import numpy as np



def draw_2d_error_curve(errs, eval_names, metric_type, fig):
    FONT_SIZE_XLABEL= 20#15
    FONT_SIZE_YLABEL= 20#15
    FONT_SIZE_LEGEND = 18#11.8
    FONT_SIZE_TICK = 18#11.8
    eval_num = len(errs)
    thresholds = np.arange(0, 50, 1)
    results = np.zeros(thresholds.shape+(eval_num,))
    #fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,2,2)
    xlabel = 'Mean distance threshold (mm)'
    ylabel = 'Fraction of frames within distance (%)'
    # color map
    jet = plt.get_cmap('jet') 
    values = range(eval_num)
    if eval_num < 3:
          jet = plt.get_cmap('prism') 
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
    
    l_styles = ['-','--','-','--','-']
    color_style = ['r','g','g','b','b']#
    auc = []
    for eval_idx in range(eval_num):
        #import pdb; pdb.set_trace()
        if errs[eval_idx].shape[0]==0:
            continue
        if metric_type == 'mean-frame':
            err = np.mean(errs[eval_idx], axis=1)
        elif  metric_type == 'max-frame':
            err = np.max(errs[eval_idx], axis=1)
            xlabel = 'Maximum allowed distance to GT (Pixel)'
        elif  metric_type == 'joint':
            err = errs[eval_idx]
            xlabel = 'Distance Threshold (Pixel)'
            ylabel = 'Fraction of joints within distance (%)'
        err_flat = err.ravel()
        for idx, th in enumerate(thresholds):
            results[idx, eval_idx] = np.where(err_flat <= th)[0].shape[0] * 1.0 / err_flat.shape[0]
        colorVal = scalarMap.to_rgba(eval_idx)
        colorVal = color_style[eval_idx]#
        
        ls = l_styles[eval_idx%len(l_styles)]
        if eval_idx == eval_num - 1:
            ls = '-'
        ls = l_styles[eval_idx]#
        ax.plot(thresholds, results[:, eval_idx]*100, label=eval_names[eval_idx], 
                color=colorVal, linestyle=ls)
        #cyj
        auc.append(np.trapz(results[:,eval_idx],thresholds)/49)
        
    plt.xlabel(xlabel, fontsize=FONT_SIZE_XLABEL)
    plt.ylabel(ylabel, fontsize=FONT_SIZE_YLABEL)
    ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    plt.grid(True)
    major_ticks = np.arange(0, 51, 10)                                              
    minor_ticks = np.arange(0, 51, 5)                                               
    ax.set_xticks(major_ticks)                                                       
    ax.set_xticks(minor_ticks, minor=True)   
    major_ticks = np.arange(0, 101, 10)                                              
    minor_ticks = np.arange(0, 101, 5)                                          
    ax.set_yticks(major_ticks)                                                       
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(which='minor', alpha=0.2, linestyle=':', linewidth=0.3)                                                
    ax.grid(which='major', alpha=0.5, linestyle='--', linewidth=0.3)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 90)
    ax.set_title("FreiHAND training set", fontsize = FONT_SIZE_XLABEL)
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    left='off',         # ticks along the top edge are off
    labelsize=FONT_SIZE_TICK)
    print("AUC:",auc)
    fig.tight_layout()   