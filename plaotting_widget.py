import os
import matplotlib.pyplot as plt
import numpy as np
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

def before_after_weights(initial_weights, final_weights,title1,title2,fname,cmaps = 'hot'):
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
    
def plot_weights_over_time(weights, titles, fname, cmaps='hot'):
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

        plt.subplot(1, num_plots, idx + 1)
        plt.title(title)
        plt.imshow(w, cmap=cmap, interpolation='nearest', aspect='auto')
        plt.colorbar()

    plt.tight_layout()
    save_plot(fname)
    plt.show()

def plot_row_correlations(data_3d, fname):
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
            corr = np.corrcoef(data_3d[0, rep], data_3d[day, rep])[0, 1]
            rep_correlations.append(corr)
        mean_correlations.append(np.mean(rep_correlations))
        std_correlations.append(np.std(rep_correlations))

    # Plot
    fig,ax  = plt.subplots(figsize=(8, 4))
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
    plt.xlabel("Elapsed time (days)")
    plt.ylabel("Ensemble activity corr.")
    plt.xticks(range(days))
    # plt.ylim(-1, 1)
    plt.tight_layout()
    save_plot(fname)
    plt.show()

