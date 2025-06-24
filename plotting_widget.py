import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
# import seaborn as sns

def ensure_dir_exists(file_path):
    """
    Ensures that the directory for the given file path exists.
    Creates directories recursively if they do not exist.

    Parameters:
    - file_path (str): Full path to the file (e.g., 'plots/figs/myplot.png')

    Returns:
    - None
    """
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_plot(filename):
    """
    Saves a plot to a file.
    
    Parameters:
    - filename (str): The path/filename to save the plot (e.g., 'plot.png')
    - plot_func (callable): A function that creates the plot using matplotlib
    - *args, **kwargs: Optional arguments to pass to `plot_func`
    
    Returns:
    - None
    """
    
    # Save the figure
    ensure_dir_exists(filename)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)

def before_after_weights(initial_weights, final_weights,title1,title2,fname,cmaps = 'gray_r'):
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title(r'$%s$'% title1)
    plt.imshow(initial_weights, cmap=cmaps, interpolation='nearest',aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.title(r'$%s$'% title2)
    plt.imshow(final_weights, cmap=cmaps, interpolation='nearest',aspect='auto')
    plt.colorbar()
    
    save_plot(fname)
    plt.show()




def plot_activity(activity, title,var_name, fname,c):
    plt.figure(figsize=(10, 5))
    plt.plot(activity,color=c)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(var_name)
    # save_plot(fname)
    plt.show()
    
def plot_activities(activity1, activity2, title,var_name, fname, color1='red', color2='blue'):
    plt.figure(figsize=(10, 5))

    for a in activity1:
        plt.plot(a, color=color1, alpha=0.6)

    for a in activity2:
        plt.plot(a, color=color2, alpha=0.6)

    # Optional: Keep or remove these as needed
    plt.title(title)
    plt.xlabel('Time (a.u.)')
    plt.ylabel(var_name)

    # No labels or legend
    # save_plot(fname)  # Uncomment if saving is needed
    plt.show()
    
def plot_weights_over_time(weights, titles, fname, cmaps='gray_r',title_fontsize=14, tick_fontsize=10, colorbar_fontsize=10):
    """
    Plots weight matrices from different time points in one row.

    Args:
        weights: List of 2D numpy arrays representing weights.
        titles: List of titles for each subplot.
        fname: Filename to save the complete figure.
        cmaps: List of colormaps (or a single colormap) for the subplots.
    """
    num_plots = len(weights)
    fig = plt.figure(figsize=(5 * num_plots, 5))
    gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=[1] * num_plots + [0.05], wspace=0.3)
    vmin = min(w.min() for w in weights)
    vmax = max(w.max() for w in weights)
    vmin = min(-1,vmin)
    vmax = max(1,vmax)
    # im = ax.imshow(w, cmap=cmap, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
    ims = []
    for idx, (w, title) in enumerate(zip(weights, titles)):
        cmap = cmaps[idx] if isinstance(cmaps, list) else cmaps
        ax = fig.add_subplot(gs[0, idx])
        im = ax.imshow(w, cmap=cmap, interpolation='nearest', aspect='auto', vmin=vmin, vmax=vmax)
        ims.append(im)
        ax.set_title(title, fontsize=title_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

    # Shared colorbar
    cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    plt.tight_layout()
    save_plot(fname)
    plt.show()
    
def plot_activity_n_excitability_time(weights, titles, fname, cmaps='hot',title_fontsize=14, tick_fontsize=10, colorbar_fontsize=10):
    """
    Plots weight matrices from different time points in one row.

    Args:
        weights: List of 2D numpy arrays representing weights.
        titles: List of titles for each subplot.
        fname: Filename to save the complete figure.
        cmaps: List of colormaps (or a single colormap) for the subplots.
    """
    num_plots = len(weights)
    plt.figure(figsize=(8 * num_plots, 5))

    for idx, (w, title) in enumerate(zip(weights, titles)):
        cmap = cmaps[idx] if isinstance(cmaps, list) else cmaps

        ax = plt.subplot(1, num_plots, idx + 1)
        im = ax.imshow(w, cmap=cmap, interpolation='nearest', aspect='auto')
        ax.set_title(title, fontsize=title_fontsize)
        ax.tick_params(labelsize=tick_fontsize)

        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=colorbar_fontsize)

    plt.tight_layout()
    save_plot(fname)
    plt.show()
def plot_row_correlations(data_3d, fname,ref_index=0, use_bar_plot=False,font_size=14, tick_fontsize=14):
    """
    Plots the correlation between the reference row and all other rows in the matrix.
    
    Parameters:
        matrix (np.ndarray): 2D array with shape (rows, columns).
        reference_row_index (int): Index of the reference row to correlate with others.
        use_bar_plot (bool): If True, use a bar plot; else use a line plot.
    """
    days, reps, neurons = data_3d.shape
    mean_correlations = []
    std_correlations = []

    for day in range(days):
        rep_correlations = []
        for rep in range(reps):
            corr = np.corrcoef(data_3d[ref_index, rep], data_3d[day, rep])[0, 1]
            rep_correlations.append(corr)
        mean_correlations.append(np.mean(rep_correlations))
        std_correlations.append(np.std(rep_correlations))

    # Plot
    fig,ax  = plt.subplots(figsize=(8, 4))
    if use_bar_plot:
        ax.bar(
            x=range(days),
            height=mean_correlations,
            yerr=std_correlations,
            capsize=5,
            color='blue',
            alpha=0.7,
            edgecolor='black'
        )
    else:
        plt.errorbar(
            x=range(days),
            y=mean_correlations,
            yerr=std_correlations,
            fmt='-o',
            capsize=5,
            ecolor='gray',
            color='blue'
        )
    ax.spines[["right", "top"]].set_visible(False)
    # plt.title("Neuron Correlation with Day 0 Over Days")
    plt.xlabel("Elapsed time (days)",fontsize =font_size)
    plt.ylabel("Ensemble activity corr.",fontsize =font_size)
    plt.xticks(range(days))
    ax.tick_params(labelsize=tick_fontsize)

    # plt.ylim(-1, 1)
    plt.tight_layout()
    save_plot(fname)
    plt.show()

def plot_consecutive_day_correlation(data_3d,fname):
    """
    Plots average neuron correlation (with std) between consecutive days, repetition-wise.

    Parameters:
        data_3d (np.ndarray): 3D array of shape (days, repetitions, neurons)
    """
    days, reps, neurons = data_3d.shape
    mean_corrs = []
    std_corrs = []
    day_labels = []

    for day in range(1, days):
        rep_corrs = []
        for rep in range(reps):
            corr = np.corrcoef(data_3d[day - 1, rep], data_3d[day, rep])[0, 1]
            rep_corrs.append(corr)
        mean_corrs.append(np.mean(rep_corrs))
        std_corrs.append(np.std(rep_corrs))
        day_labels.append(f"{day-1}-{day}")

    # Plot with error bars
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        x=range(1, days),
        y=mean_corrs,
        yerr=std_corrs,
        fmt='-o',
        capsize=5,
        ecolor='gray',
        color='green'
    )
    plt.title("Neuron Correlation Between Consecutive Days")
    plt.xlabel("Day Pair")
    plt.ylabel("Correlation")
    plt.xticks(ticks=range(1, days), labels=day_labels)
    plt.ylim(-1, 1)
    # plt.grid(True)
    save_plot(fname)
    plt.show()
