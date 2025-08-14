import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from Utilities import center_of_mass_rowwise, center_of_mass_columnwise_blocked
import seaborn as sns
from sklearn.decomposition import PCA
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




def plot_activity(activity, title,var_name, fname,c, th = 10):
    plt.figure(figsize=(10, 5))
    plt.plot(activity, color=c)
    plt.axhline(y=th, color='black', linestyle='--', linewidth=1)  # Dashed line at y = th
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(var_name)
    save_plot(fname)
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
    breakpoint()
    num_plots = len(weights)
    fig = plt.figure(figsize=(5 * num_plots, 5))
    gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=[1] * num_plots + [0.05], wspace=0.3)
    vmin = min(w.min() for w in weights)
    vmax = max(w.max() for w in weights)
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
    plt.figure(figsize=(8 * num_plots, 4))

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
def plot_row_correlations(ref_activity,data_2d,xlabs, title, fname, use_bar_plot=False,font_size=14, tick_fontsize=14):
    """
    Plots the correlation between the reference row and all other rows in the matrix.
    
    Parameters:
        matrix (np.ndarray): 2D array with shape (rows, columns).
        reference_row_index (int): Index of the reference row to correlate with others.
        use_bar_plot (bool): If True, use a bar plot; else use a line plot.
    """
    days, neurons = data_2d.shape

    mean_correlations = []
    # breakpoint()
    for day in range(days):
        mean_correlations.append(np.corrcoef(ref_activity,data_2d[day])[0,1])

    # Plot
    fig,ax  = plt.subplots(figsize=(8, 4))
    if use_bar_plot:
        ax.bar(
            x=range(days),
            height = mean_correlations,
            # yerr=std_correlations,
            capsize=5,
            color='blue',
            alpha=0.7,
            edgecolor='black'
        )
    else:
        plt.errorbar(
            x=range(days),
            y=mean_correlations,
            # yerr=std_correlations,
            fmt='-o',
            capsize=5,
            ecolor='gray',
            color='blue'
        )
    ax.spines[["right", "top"]].set_visible(False)
    plt.title(title)
    plt.xlabel("Elapsed time (days)",fontsize =font_size)
    plt.ylabel("Ensemble activity corr.",fontsize =font_size)
    plt.xticks(ticks=range(days), labels=xlabs)
    ax.tick_params(labelsize=tick_fontsize)

    # plt.ylim(-1, 1)
    plt.tight_layout()
    save_plot(fname)
    plt.show()

def plot_corr_matrix(data,fname):
    """
    Plots the correlation matrix of the input data.
    
    Parameters:
        data (np.ndarray): 2D array with shape (rows, columns).
        fname (str): Filename to save the plot.
    """
    corr_matrix = np.corrcoef(data)  # Transpose to get correlations between columns
    print(corr_matrix)
    plt.figure(figsize=(6, 5))
    # Mask the diagonal (set it to NaN so it won't be colored)
    mask = np.eye(corr_matrix.shape[0], dtype=bool)
    min_v,max_v = np.min(corr_matrix), np.max(corr_matrix)
    sns.heatmap(corr_matrix, annot=False, cmap="viridis", vmin=min_v, vmax=max_v, square=True, mask=mask, cbar=True)

    # plt.colorbar(label='Correlation Coefficient')
    plt.title('Correlation Matrix')
    plt.xlabel('Days')
    plt.ylabel('Days')
    save_plot(fname)
    plt.show()

def plot_consecutive_day_correlation(data_2d,fname):
    """
    Plots average neuron correlation (with std) between consecutive days, repetition-wise.

    Parameters:
        data_3d (np.ndarray): 3D array of shape (days, repetitions, neurons)
    """
    days, neurons = data_2d.shape
    mean_corrs = []
    day_labels = []

    for day in range(1, days):
        mean_corrs.append( np.corrcoef(data_2d[day - 1], data_2d[day])[0, 1])
        day_labels.append(f"{day-1}-{day}")

    # Plot with error bars
    plt.figure(figsize=(8, 4))
    plt.errorbar(
        x=range(1, days),
        y=mean_corrs,
        # yerr=std_corrs,
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

def plot_rowwise_com(matrix, num_days,fname,title=''):
    """
    Computes and plots the row-wise center of mass of a 2D matrix.
    
    Parameters:
        matrix (2D array): Input matrix.
        title (str): Plot title.
    """
    # matrix = np.asarray(matrix)
    # rows = np.arange(matrix.shape[0])
    
    # # Compute column-wise center of mass
    # col_sums = matrix.sum(axis=0)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     com = (matrix.T * rows).sum(axis=1) / col_sums
    #     com = np.where(col_sums == 0, np.nan, com)  # Handle zero columns gracefully
    
    # # Plot
    # plt.figure(figsize=(6, 4))
    # days_to_plot = np.arange(off_set, matrix.shape[1], time_in_a_day)
    # com_oi = [com[d] for d in days_to_plot]
    com_means,com_std = center_of_mass_columnwise_blocked(matrix)
    x = [i for i in range(num_days)]
    xlabs = ["Encoding", "Off 1","Off 2","Off 3","Off 4","Off 5","Recall"]
    # breakpoint()
    plt.errorbar(x,com_means,com_std, marker = 'o',color = 'k', label='Center of Mass')
    plt.xticks(ticks=x, labels=xlabs, rotation=45)
    plt.xlabel('Time (days)')
    plt.ylabel('Center of mass of the output weights (# neuron)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(fname)
    plt.show()



# def plot_pca(matrix,lables=None, title='PCA Projection', fname='pca_plot.png', cmap='viridis'):
#     if lables is not None:
#         scatter = plt.scatter(matrix[:, 0], matrix[:, 1], c=lables, cmap=cmap, s=40)
#         plt.legend(*scatter.legend_elements(), title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
#     else:
#         plt.scatter(matrix[:, 0], matrix[:, 1], c=np.arange(matrix.shape[0]), cmap=cmap, s=40)
#         plt.colorbar(label='Column Index')
#     plt.xlabel('PC 1')
#     plt.ylabel('PC 2')
#     plt.title('PCA projection')
#     plt.grid(True)
#     plt.show()
    
def plot_pca_2d(X, labels=None, cmap='tab10', title='PCA projection'):
    """
    Plots the first two principal components of the columns of X.

    Args:
        X: 2D numpy array of shape (n_features, n_samples)
        labels: List or array of labels (length = n_samples) for color coding
        cmap: Matplotlib colormap
        title: Title of the plot
    """
    X = X.T  # Each column is a sample → transpose to shape (n_samples, n_features)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))

    if labels is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, s=40)
        plt.legend(*scatter.legend_elements(), title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(X.shape[0]), cmap=cmap, s=40)
        plt.colorbar(label='Column Index')

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_avg_activity(activities, titles, fname, cmaps='gray', title_fontsize=14, tick_fontsize=10, colorbar_fontsize=10):
    # breakpoint()
    num_plots = int(len(activities) // 2)
    fig,ax = plt.subplots(figsize=(5 * num_plots, 5),ncols =num_plots)
    # ax =ax.split(sharey=True)
    for i in range(num_plots):
        avg_activity = np.mean(activities[i], axis=1)
        # ax[i].plot(avg_activites[i], color=cmaps[i] if isinstance(cmaps, list) else cmaps)
        ax[i % 2].step(range(len(avg_activity)), avg_activity, where='post', linewidth=i // 2 + 1,color=cmaps[i] if isinstance(cmaps, list) else cmaps)
        # print(avg_activity)
    plt.tight_layout()
    save_plot(fname)
    plt.show()