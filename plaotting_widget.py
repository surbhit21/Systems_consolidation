import os
import matplotlib.pyplot as plt
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




def plot_activity(monitor,var_name, title, fname):
    plt.figure(figsize=(10, 5))
    frs = monitor.get(var_name)
    plt.plot(frs[:])
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(var_name)
    # save_plot(fname)
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
    plt.figure(figsize=(5 * num_plots, 5))

    for idx, (w, title) in enumerate(zip(weights, titles)):
        cmap = cmaps[idx] if isinstance(cmaps, list) else cmaps

        plt.subplot(1, num_plots, idx + 1)
        plt.title(title)
        plt.imshow(w, cmap=cmap, interpolation='nearest', aspect='auto')
        plt.colorbar()

    plt.tight_layout()
    save_plot(fname)
    plt.show()