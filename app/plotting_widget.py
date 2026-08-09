import os
# from turtle import pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from Utilities import center_of_mass_rowwise, center_of_mass_columnwise_blocked
import seaborn as sns
from sklearn.decomposition import PCA
# import seaborn as sns
fformat = [ ".pdf", ".png",".svg"]

PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 18
PLOT_TICK_FONTSIZE = 18
PLOT_COLORBAR_FONTSIZE = 18
PLOT_LEGEND_FONTSIZE = 18

# Apply the shared typography to direct Matplotlib calls as well as functions
# that explicitly use the constants below.
matplotlib.rcParams.update({
    "axes.titlesize": PLOT_TITLE_FONTSIZE,
    "axes.labelsize": PLOT_LABEL_FONTSIZE,
    "xtick.labelsize": PLOT_TICK_FONTSIZE,
    "ytick.labelsize": PLOT_TICK_FONTSIZE,
    "legend.fontsize": PLOT_LEGEND_FONTSIZE,
    "legend.title_fontsize": PLOT_LEGEND_FONTSIZE,
    "figure.titlesize": PLOT_TITLE_FONTSIZE,
})


def setBoxColors(bp, c, a=1, flipped=False):
    """Apply one color consistently to all artists in a boxplot."""
    plt.setp(bp["boxes"], color=c, alpha=a)
    plt.setp(bp["caps"], color=c, alpha=a)
    plt.setp(bp["whiskers"], color=c, alpha=a)
    plt.setp(bp["fliers"], color=c, alpha=a)
    plt.setp(bp["medians"], color="w" if flipped else c, alpha=a)


def style_axis(ax, title=None, xlabel=None, ylabel=None,
               title_fontsize=PLOT_TITLE_FONTSIZE,
               label_fontsize=PLOT_LABEL_FONTSIZE,
               tick_fontsize=PLOT_TICK_FONTSIZE):
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)


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

def save_plot(filename, tight_layout=True):
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
    if tight_layout:
        plt.tight_layout()
    print("saving plot to:", filename)
    for ff in fformat:
        plt.savefig(filename+ff, bbox_inches='tight', dpi=300)

def before_after_weights(initial_weights, final_weights,title1,title2,fname,cmaps = 'gray_r'):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    ax.set_title(r'$%s$'% title1, fontsize=PLOT_TITLE_FONTSIZE)
    plt.imshow(initial_weights, cmap=cmaps, interpolation='nearest',aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label("Weights [a.u.]", fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_COLORBAR_FONTSIZE)
    ax = plt.subplot(122)
    ax.set_title(r'$%s$'% title2, fontsize=PLOT_TITLE_FONTSIZE)
    plt.imshow(final_weights, cmap=cmaps, interpolation='nearest',aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label("Weights [a.u.]", fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_COLORBAR_FONTSIZE)
    
    save_plot(fname)
    plt.show()




def plot_activity(activity, title,var_name, fname,c, th = 10):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(activity, color=c)
    ax.axhline(y=th, color='black', linestyle='--', linewidth=1)  # Dashed line at y = th
    style_axis(ax, title=title, xlabel='Time (s)', ylabel=var_name)
    save_plot(fname)
    plt.show()
    
def plot_activities(activity1, activity2, title,var_name, fname, color1='red', color2='blue'):
    plt.figure(figsize=(10, 5))

    for a in activity1:
        plt.plot(a, color=color1, alpha=0.6)

    for a in activity2:
        plt.plot(a, color=color2, alpha=0.6)

    # Optional: Keep or remove these as needed
    style_axis(plt.gca(), title=title, xlabel='Time (a.u.)', ylabel=var_name)

    # No labels or legend
    # save_plot(fname)  # Uncomment if saving is needed
    plt.show()
    
def plot_weights_over_time(
    weights,
    titles,
    fname,
    cmaps='gray_r',
    title_fontsize=PLOT_TITLE_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
    colorbar_fontsize=PLOT_COLORBAR_FONTSIZE,
    plot_title="",
    com=None,
    in_set=0,
    colorbar_label="Weights [a.u.]",
):
    """
    Plots weight matrices from different time points in one row.

    Args:
        weights: List of 2D numpy arrays representing weights.
        titles: List of titles for each subplot.
        fname: Filename to save the complete figure.
        cmaps: List of colormaps (or a single colormap) for the subplots.
        in_set: If 1, add an inset showing weights among the first 25 neurons.
    """
    # breakpoint()
    num_plots = len(weights)
    fig = plt.figure(figsize=(4 * num_plots, 4))
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
        style_axis(ax, title=title, title_fontsize=title_fontsize, tick_fontsize=tick_fontsize)
        n = w.shape[0]
        if com:
            x_coords, y_coords = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            x_com = np.sum(x_coords * w) / np.sum(w)
            y_com = np.sum(y_coords * w) / np.sum(w)
            print(x_com,y_com)
            ax.scatter(y_com, x_com, color='red', s=100, edgecolors='white', label='Center of Mass')
        if in_set == 1:
            inset_size = min(25, w.shape[0], w.shape[1])
            inset_ax = ax.inset_axes([0.58, 0.58, 0.38, 0.38])
            inset_ax.imshow(
                w[:inset_size, :inset_size],
                cmap=cmap,
                interpolation='nearest',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
            )
            inset_ax.set_title("First 25", fontsize=max(8, title_fontsize // 2))
            inset_ax.tick_params(axis='both', which='major', labelsize=max(6, tick_fontsize // 2))
            if inset_size > 1:
                inset_ax.set_xticks([0, inset_size - 1])
                inset_ax.set_yticks([0, inset_size - 1])

    # Shared colorbar
    cax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(ims[0], cax=cax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=PLOT_LABEL_FONTSIZE)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    if plot_title:
        fig.suptitle(plot_title, fontsize=title_fontsize)
        plt.tight_layout(rect=(0, 0, 1, 0.9))
    else:
        plt.tight_layout()
    save_plot(fname)
    plt.show()
    
def active_input_intervals(input_history, threshold=1e-12):
    """Return half-open time intervals during which any external input is active."""
    input_history = np.asarray(input_history)
    if input_history.ndim == 1:
        active = np.abs(input_history) > threshold
    elif input_history.ndim == 2:
        active = np.any(np.abs(input_history) > threshold, axis=1)
    else:
        raise ValueError("input_history must have shape (time,) or (time, inputs).")

    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def plot_activity_n_excitability_time(
    weights,
    titles,
    fname,
    cmaps='hot',
    seqA=[],
    title_fontsize=PLOT_TITLE_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
    colorbar_fontsize=PLOT_COLORBAR_FONTSIZE,
    time_per_day=None,
    day_zero_time=0,
    colorbar_label="Firing rate (Hz)",
    input_history=None,
    input_marker="|",
    ncols=None,
    figsize=None,
    day_labels=None,
):
    """
    Plots weight matrices from different time points in one row.

    Args:
        weights: List of 2D numpy arrays representing weights.
        titles: List of titles for each subplot.
        fname: Filename to save the complete figure.
        cmaps: List of colormaps (or a single colormap) for the subplots.
        colorbar_label: A single label used for every panel, or a list/tuple
            containing one colorbar label per panel. Use ``None`` or an empty
            string to omit a panel's colorbar label.
    """
    num_plots = len(weights)
    if ncols is None:
        ncols = num_plots
    if ncols <= 0:
        raise ValueError("ncols must be positive")
    nrows = int(np.ceil(num_plots / ncols))
    if figsize is None:
        figsize = (8 * ncols, 4 * nrows)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )
    flat_axes = axes.ravel()

    def time_to_day(time_value):
        return (time_value - day_zero_time) / time_per_day

    use_day_axis = time_per_day is not None and time_per_day > 0

    if input_history is not None:
        input_intervals = active_input_intervals(input_history)
    else:
        # Backward-compatible fallback for older callers that only provide
        # alternating stimulation start/stop boundaries in seqA.
        input_intervals = list(zip(seqA[0::2], seqA[1::2]))

    for idx, (w, title) in enumerate(zip(weights, titles)):
        cmap = cmaps[idx] if isinstance(cmaps, list) else cmaps
        # c_map = LinearSegmentedColormap.from_list("custom_cmap", ['#ffffff',cmap])
        ax = flat_axes[idx]
        ax.set_facecolor("white")
        if use_day_axis:
            x_start = time_to_day(0)
            x_end = time_to_day(w.shape[1])
            im = ax.imshow(
                w,
                cmap=cmap,
                interpolation='nearest',
                aspect='auto',
                extent=(x_start, x_end, w.shape[0] - 0.5, -0.5),
            )
        else:
            im = ax.imshow(w, cmap=cmap, interpolation='nearest', aspect='auto')
        style_axis(ax, title=title, title_fontsize=title_fontsize, tick_fontsize=tick_fontsize)
        ax.spines[['right', 'top']].set_visible(False)
        cbar = fig.colorbar(im, ax=ax, pad=0.025)
        if isinstance(colorbar_label, (list, tuple)):
            if len(colorbar_label) != num_plots:
                raise ValueError(
                    "colorbar_label must contain one label per panel"
                )
            panel_colorbar_label = colorbar_label[idx]
        else:
            panel_colorbar_label = colorbar_label
        if panel_colorbar_label:
            cbar.set_label(
                panel_colorbar_label, fontsize=PLOT_LABEL_FONTSIZE
            )
        cbar.ax.tick_params(labelsize=colorbar_fontsize)
        marker_y = -10
        input_onsets = [start for start, _ in input_intervals]
        if use_day_axis:
            input_onsets = [time_to_day(start) for start in input_onsets]
        ax.plot(
            input_onsets,
            np.full(len(input_onsets), marker_y),
            linestyle="none",
            marker=input_marker,
            color="black",
            markersize=10,
            markeredgewidth=1.4,
            clip_on=False,
        )
        if use_day_axis:
            if day_labels is not None:
                day_centers = np.arange(len(day_labels), dtype=float) + 0.5
                ax.set_xticks(day_centers, labels=day_labels)
            else:
                first_day = int(np.ceil(time_to_day(0)))
                last_day = int(np.floor(time_to_day(w.shape[1])))
                ax.set_xticks(np.arange(first_day, last_day + 1, 1))
            ax.set_xlabel('Session', fontsize=PLOT_LABEL_FONTSIZE)
        else:
            ax.set_xlabel('Time (a.u.)', fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_ylabel('Neurons', fontsize=PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    for unused_ax in flat_axes[num_plots:]:
        unused_ax.set_visible(False)
    fig.subplots_adjust(
        left=0.09,
        right=0.90,
        bottom=0.10,
        top=0.94,
        hspace=0.55,
    )
    save_plot(fname, tight_layout=False)
    plt.show()
def plot_row_correlations(ref_activity,data_2d,xlabs, title, fname, use_bar_plot=False,font_size=PLOT_LABEL_FONTSIZE, tick_fontsize=PLOT_TICK_FONTSIZE):
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
    style_axis(ax, title=title, xlabel="Elapsed time (days)", ylabel="Ensemble activity corr.",
               title_fontsize=font_size, label_fontsize=font_size, tick_fontsize=tick_fontsize)
    plt.xticks(ticks=range(days), labels=xlabs)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

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
    ax = sns.heatmap(corr_matrix, annot=True, cmap="viridis", vmin=min_v, vmax=max_v, square=True, mask=mask, cbar=True,annot_kws={"size": 8, "color": "white", "ha": "center", "va": "center"})
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=PLOT_COLORBAR_FONTSIZE)

    # plt.colorbar(label='Correlation Coefficient')
    style_axis(ax, title='Correlation Matrix', xlabel='Days', ylabel='Days')
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
    style_axis(plt.gca(), title="Neuron Correlation Between Consecutive Days",
               xlabel="Day Pair", ylabel="Correlation")
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
    style_axis(plt.gca(), title=title, xlabel='Time (days)',
               ylabel='Center of mass of the output weights (# neuron)')
    plt.grid(True)
    plt.legend(fontsize=PLOT_LEGEND_FONTSIZE)
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
        plt.legend(*scatter.legend_elements(), title="Label", bbox_to_anchor=(1.05, 1),
                   loc='upper left', fontsize=PLOT_LEGEND_FONTSIZE,
                   title_fontsize=PLOT_LEGEND_FONTSIZE)
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(X.shape[0]), cmap=cmap, s=40)
        cbar = plt.colorbar(label='Column Index')
        cbar.ax.tick_params(labelsize=PLOT_COLORBAR_FONTSIZE)

    style_axis(plt.gca(), title=title, xlabel='PC 1', ylabel='PC 2')
    plt.tight_layout()
    plt.show()

def plot_avg_activity(activities, titles, fname, cmaps='gray', title_fontsize=PLOT_TITLE_FONTSIZE, tick_fontsize=PLOT_TICK_FONTSIZE, colorbar_fontsize=PLOT_COLORBAR_FONTSIZE):
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


def plot_mean_std_corr_over_time(
    data_3d,
    ref_time_idx=0,
    xlabels=None,
    title="Mean ± SEM of neuron-ensemble correlation over time (across simulations)",
    fname=None,
    include_ref_bar=False,
    cmap="viridis",            # now accepts a colormap name or object
    font_size=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
    markersize = 10,
    capsize=5,
    marker='o',
    linewidth=3,
    bar_plot = False,
    no_plot = False,
    xlab = "Days"
):
    """
    Compute & plot mean ± sem of correlations over time across simulations (errorbar plot).

    Parameters
    ----------
    data_3d : np.ndarray
        Shape (sims, time, neurons). Axis 0 = simulations.
    ref_time_idx : int
        The reference time index (e.g., 0 for Encoding, -1 for Recall).
        Correlations are computed: corr( data[s, ref_time_idx, :], data[s, t, :] ) for each sim s.
    xlabels : list[str] or None
        Labels for the x-axis points. If None, generated automatically.
    title : str
        Plot title.
    fname : str or None
        If provided, saves to this path; else shows the plot.
    include_ref_bar : bool
        If True, include the reference time point (usually 1.0 correlation).
    cmap : str or matplotlib.colors.Colormap
        Colormap for line or points.
    font_size : int
        Font size for labels/title.
    tick_fontsize : int
        Font size for tick labels.
    capsize : int
        Error bar cap size.
    marker : str
        Marker style for points.
    linewidth : float
        Line width.
    barplot: bool
        plot as barplot or errorbar

    Returns
    -------
    mean_corr : np.ndarray
    sem_corr  : np.ndarray
    per_sim_corr : np.ndarray
    sel_time_idx : list[int]
    """

    def _safe_corr(a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if a.size != b.size:
            raise ValueError(f"Vectors must match in length, got {a.size} vs {b.size}.")
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return np.corrcoef(a, b)[0, 1]

    if data_3d.ndim != 3:
        raise ValueError("data_3d must be 3D with shape (sims, time, neurons).")

    sims, T, N = data_3d.shape
    ref_time_idx = int(ref_time_idx)

    # select time indices
    all_times = list(range(T))
    sel_time_idx = all_times if include_ref_bar else [t for t in all_times if t != (ref_time_idx % T)]

    # compute correlations per simulation
    per_sim_corr = np.full((sims, len(sel_time_idx)), np.nan)
    for s in range(sims):
        ref_vec = data_3d[s, ref_time_idx, :]
        for j, t in enumerate(sel_time_idx):
            per_sim_corr[s, j] = _safe_corr(ref_vec, data_3d[s, t, :])

    mean_corr = np.nanmean(per_sim_corr, axis=0)
    std_corr  = np.nanstd(per_sim_corr, axis=0)
    sem_corr  = std_corr / np.sqrt(sims)
    if no_plot:
        return mean_corr, sem_corr, per_sim_corr, sel_time_idx
    # x labels
    if xlabels is None:
        xlabels = [f"T{t}" for t in sel_time_idx]
    elif len(xlabels) != len(sel_time_idx):
        raise ValueError(f"len(xlabels)={len(xlabels)} must equal {len(sel_time_idx)}.")

    # get color from cmap (e.g., central color of the range)
    cm = plt.get_cmap(cmap)
    

    # plot mean ± sem as errorbar line plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sel_time_idx))
    if bar_plot:
        colors = [cm(i / T) for i in range(T)]
        ax.bar( x, mean_corr, yerr = sem_corr,color=colors)
    else:
        color = cm(0.5)
        ax.errorbar(
            x, mean_corr, yerr=sem_corr, fmt=marker + '-',
            markersize=8, linewidth=linewidth, capsize=capsize,
            color=color, ecolor=color, elinewidth=2, alpha=0.9
        )

    # styling
    ax.spines[["right", "top"]].set_visible(False)
    style_axis(ax, title=title, xlabel=xlab, ylabel="PV Correlation",
               title_fontsize=font_size, label_fontsize=font_size, tick_fontsize=tick_fontsize)
    ax.set_xticks(x, xlabels)
    y_min = mean_corr.min() - sem_corr.max()
    ax.set_ylim([0, 1.1])
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.tight_layout()

    # save or show
    try:
        if fname is not None:
            try:
                save_plot(fname)
            except NameError:
                plt.savefig(fname, dpi=200)
            plt.show()
            plt.close(fig)
        else:
            plt.show()
    except Exception:
        plt.close(fig)

    return mean_corr, sem_corr, per_sim_corr, sel_time_idx



def plot_population_vector_correlations(
    correlation_series,
    sem_series,
    xlabels,
    labels,
    fname=None,
    title="Cell population \n activity correlation",
    colors=None,
    markers=None,
    font_size=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
    linewidth=3,
    capsize=5,
):
    """
    Plot multiple population-vector correlation curves on the same axes.
    """
    if not (len(correlation_series) == len(sem_series) == len(labels)):
        raise ValueError("correlation_series, sem_series, and labels must have the same length.")

    n_points = len(correlation_series[0])
    if len(xlabels) != n_points:
        raise ValueError(f"len(xlabels)={len(xlabels)} must equal number of points={n_points}.")
    for corr, sem in zip(correlation_series, sem_series):
        if len(corr) != n_points or len(sem) != n_points:
            raise ValueError("All correlation and SEM arrays must have the same length.")

    if colors is None:
        colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
    if markers is None:
        markers = ["^", "o", "s", "D"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(n_points)

    for i, (corr, sem, label) in enumerate(zip(correlation_series, sem_series, labels)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.errorbar(
            x,
            corr,
            yerr=sem,
            fmt=marker + "-",
            markersize=8,
            linewidth=linewidth,
            capsize=capsize,
            color=color,
            ecolor=color,
            elinewidth=2,
            alpha=0.9,
            label=label,
        )

    ax.spines[["right", "top"]].set_visible(False)
    style_axis(
        ax,
        title=title,
        xlabel="Days",
        ylabel="PV Correlation",
        title_fontsize=font_size,
        label_fontsize=font_size,
        tick_fontsize=tick_fontsize,
    )
    ax.set_xticks(x, xlabels)
    ax.set_ylim(ymin=0)
    ax.legend(frameon=False, fontsize=tick_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    fig.tight_layout()

    if fname is not None:
        try:
            save_plot(fname)
        except NameError:
            plt.savefig(fname, dpi=200)
        plt.show()
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_first_activity_vs_active_sessions(
    last_activity_all,              # shape: (sims, sessions, neurons)
    threshold,                      # scalar threshold for "active"
    first_session_idx=0,            # which session is "first"
    mode="concat",                  # "concat" | "mean" | "median"
    include_ref_in_count=True,      # if False, counts exclude the first session
    exclude_sessions=None,          # iterable of session indices to ignore in counts (applied after include_ref_in_count)
    fname=None,                     # optional save path
    font_size=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
    alpha=0.6,
    show_linefit=True               # add least-squares fit line to scatter
):
    """
    Plot first-session activity vs. number of sessions active (>threshold), excluding
    neurons not active in the first session (per the chosen mode).

    Modes:
      - "concat": each point is a (sim, neuron) pair where the neuron is active in the
                  first session for that sim.
      - "mean":   one point per neuron, x/y are means across sims that had this neuron
                  active in the first session.
      - "median": same as mean, but uses medians.

    Returns
    -------
    x_vals, y_vals : 1D arrays used in the scatter
    r_pearson      : float (NaN if undefined)
    """

    if last_activity_all.ndim != 3:
        raise ValueError("last_activity_all must be (sims, sessions, neurons)")

    S, T, N = last_activity_all.shape
    fs = first_session_idx % T

    # Active mask; treat NaNs as not active (change if you prefer to drop NaN sims)
    active = np.where(np.isnan(last_activity_all), False, last_activity_all > threshold)  # (S, T, N)

    # Optionally exclude certain sessions from the count
    keep_sessions = np.ones(T, dtype=bool)
    if not include_ref_in_count:
        keep_sessions[fs] = False
    if exclude_sessions is not None:
        for t in exclude_sessions:
            keep_sessions[t % T] = False

    # Counts of active sessions per (sim, neuron), with chosen session filtering
    counts = active[:, keep_sessions, :].sum(axis=1)     # (S, N)

    # First-session activity and activeness mask
    first_activity = last_activity_all[:, fs, :]         # (S, N) raw activity
    first_active   = active[:, fs, :]                    # (S, N) boolean

    if mode == "concat":
        # Keep only (sim, neuron) pairs active in first session for that sim
        mask = first_active
        x_vals = first_activity[mask]
        y_vals = counts[mask]

    elif mode in ("mean", "median"):
        # Aggregate per neuron across sims where it's active in first session
        x_vals = np.full(N, np.nan, dtype=float)
        y_vals = np.full(N, np.nan, dtype=float)
        for i in range(N):
            m = first_active[:, i]
            if not np.any(m):
                continue  # neuron never active in first session in any sim -> drop
            if mode == "mean":
                x_vals[i] = np.nanmean(first_activity[m, i])
                y_vals[i] = np.nanmean(counts[m, i])
            else:  # "median"
                x_vals[i] = np.nanmedian(first_activity[m, i])
                y_vals[i] = np.nanmedian(counts[m, i])

        sel = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals, y_vals = x_vals[sel], y_vals[sel]

    else:
        raise ValueError("mode must be 'concat', 'mean', or 'median'.")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x_vals, y_vals, alpha=alpha, edgecolors='none')

    # Optional least-squares line (only if we have enough variation)
    r_pearson = np.nan
    if show_linefit and x_vals.size >= 2 and np.std(x_vals) > 0:
        # Fit y = a*x + b
        a, b = np.polyfit(x_vals, y_vals, deg=1)
        xs = np.linspace(np.min(x_vals), np.max(x_vals), 200)
        ys = a*xs + b
        ax.plot(xs, ys, linewidth=2)

        # Pearson r
        if np.std(y_vals) > 0:
            r_pearson = np.corrcoef(x_vals, y_vals)[0,1]

    ax.spines[["right", "top"]].set_visible(False)
    style_axis(ax, title="First-session activity vs. # sessions active",
               xlabel="Activity in first session",
               ylabel="# sessions active (> threshold)",
               title_fontsize=font_size, label_fontsize=font_size, tick_fontsize=tick_fontsize)
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()

    if fname:
        try:
            save_plot(fname)  # if you have a helper
        except NameError:
            plt.savefig(fname, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return x_vals, y_vals, r_pearson

def plot_sessions_count_vs_activity_sem(
    last_activity_all,          # (sims, sessions, neurons)
    threshold,                  # scalar threshold for "active"
    first_session_idx=0,        # index of the 'first' session
    include_ref_in_count=True,  # if False, counts exclude the first session
    exclude_sessions=None,      # iterable of session indices to ignore in counts
    activity_source='first',    # 'first' or callable(data_3d)->(S,N) custom per (sim,neuron) activity
    sem_mode='pooled',          # 'pooled' or 'across_sims' (see docstring)
    fname=None,                 # optional save path
    font_size=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE,
    capsize=5,
    colors=None,
    title="Encoding activity predicts offline engram persistence"
):
    """
    Makes a bar plot: x = number of sessions a neuron is active in (>threshold),
    y = mean ± SEM of activity values, using only neurons active in the FIRST session.

    activity_source:
      - 'first' -> use first-session activity as y-values.
      - callable -> function(data_3d)->(S,N) giving a per-(sim,neuron) y-value matrix.
                    Example: lambda X: X[:, -1, :] for last-session activity.

    sem_mode:
      - 'pooled': SEM computed over all pooled (sim,neuron) samples for each count-bin.
      - 'across_sims': compute mean within each simulation (per count-bin) then SEM across sims
                       (sims without members in a bin are skipped for that bin).

    Returns
    -------
    counts_unique : np.ndarray of shape (K,)  # the x tick values
    mean_vals     : np.ndarray of shape (K,)
    sem_vals      : np.ndarray of shape (K,)
    n_per_bin     : np.ndarray of shape (K,)  # number of samples (pooled) or sims used (across_sims)
    """

    if last_activity_all.ndim != 3:
        raise ValueError("last_activity_all must be (sims, sessions, neurons)")
    S, T, N = last_activity_all.shape
    fs = first_session_idx % T

    # Active mask; treat NaNs as not active
    active = np.where(np.isnan(last_activity_all), False, last_activity_all > threshold)  # (S, T, N)

    # Only keep neurons active in FIRST session
    first_active = active[:, fs, :]  # (S, N)

    # Sessions to count
    keep_sessions = np.ones(T, dtype=bool)
    if not include_ref_in_count:
        keep_sessions[fs] = False
    if exclude_sessions is not None:
        for t in exclude_sessions:
            keep_sessions[t % T] = False

    # Number of sessions active (with filtering), per (sim, neuron)
    counts = active[:, keep_sessions, :].sum(axis=1)  # (S, N)

    # y-values (activity)
    if activity_source == 'first':
        y_activity = last_activity_all[:, fs, :]      # (S, N)
    elif callable(activity_source):
        y_activity = activity_source(last_activity_all)
        if y_activity.shape != (S, N):
            raise ValueError("Custom activity_source must return shape (sims, neurons)")
    else:
        raise ValueError("activity_source must be 'first' or a callable")

    # Apply mask: only include pairs active in FIRST session
    mask = first_active
    counts = counts[mask]          # 1D pooled counts per selected pair
    y_vals = y_activity[mask]      # 1D pooled activity per selected pair

    # Unique count bins (sorted)
    counts_unique = np.unique(counts.astype(int))
    # If you want to force a fixed range (e.g., 1..T), replace the line above with:
    # counts_unique = np.arange(1 if include_ref_in_count else 0, keep_sessions.sum() + (1 if include_ref_in_count else 0))

    mean_vals, sem_vals, n_per_bin = [], [], []

    if sem_mode == 'pooled':
        for k in counts_unique:
            ys = y_vals[counts == k]
            n = ys.size
            m = np.nanmean(ys) if n else np.nan
            s = np.nanstd(ys, ddof=1) / np.sqrt(n) if n > 1 else 0.0
            mean_vals.append(m); sem_vals.append(s); n_per_bin.append(n)

    elif sem_mode == 'across_sims':
        # For each sim, compute mean activity of neurons (in that sim) with count == k & active in first
        # Then compute grand mean and SEM across sims with non-empty bins.
        # Build per-sim arrays
        counts_sim = np.where(first_active, active[:, keep_sessions, :].sum(axis=1), np.nan)  # (S,N) with NaN where not first-active
        y_sim = np.where(first_active, y_activity, np.nan)                                   # (S,N) with NaN where not first-active
        for k in counts_unique:
            # mean per sim for neurons with count k
            means_per_sim = []
            for s in range(S):
                msk = np.isfinite(counts_sim[s]) & (counts_sim[s].astype(float) == float(k))
                if np.any(msk):
                    means_per_sim.append(np.nanmean(y_sim[s, msk]))
            means_per_sim = np.array(means_per_sim, dtype=float)
            n = means_per_sim.size
            m = np.nanmean(means_per_sim) if n else np.nan
            s = (np.nanstd(means_per_sim, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            mean_vals.append(m); sem_vals.append(s); n_per_bin.append(n)
    else:
        raise ValueError("sem_mode must be 'pooled' or 'across_sims'")

    counts_unique = counts_unique.astype(int)
    mean_vals = np.array(mean_vals, dtype=float)
    sem_vals  = np.array(sem_vals,  dtype=float)
    n_per_bin = np.array(n_per_bin, dtype=int)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(counts_unique))
    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i % cmap.N) for i in range(len(counts_unique))]
    ax.errorbar(x, mean_vals, yerr=sem_vals, capsize=capsize, alpha=0.9)

    ax.spines[["right", "top"]].set_visible(False)
    style_axis(ax, title=title, xlabel="# of offline sessions active",
               ylabel="Firing rate (Hz)",
               title_fontsize=font_size, label_fontsize=font_size, tick_fontsize=tick_fontsize)
    ax.set_xticks(x, counts_unique)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    fig.tight_layout()

    if fname:
        try:
            save_plot(fname)  # your helper, if present
        except NameError:
            plt.savefig(fname, dpi=200)
        plt.close(fig)
    else:
        plt.show()

    return counts_unique, mean_vals, sem_vals, n_per_bin

# def plot_firing_rate(timepoints, firing_rate, lab, fname = None,
#                      xlabel="Time", ylabel="Firing Rate (Hz)", 
#                      c="r", threshold=8, ticksize=14):
#     """
#     Plot firing rate over time with an active threshold line.

#     Parameters:
#         timepoints : array-like
#             Time axis values.
#         firing_rate : array-like
#             Firing rate values to plot.
#         xlabel, ylabel : str
#             Axis labels.
#         color : str
#             Color for the firing rate curve.
#         threshold : float
#             y-value for the horizontal dashed threshold line.
#         ticksize : int
#             Font size for axis ticks.
#     """
#     fig, ax = plt.subplots(figsize=(8, 4))
#     plt.plot(timepoints, firing_rate, label=lab, color=c)
#     plt.hlines(y=threshold, xmin=timepoints[0], xmax=timepoints[-1],
#                colors='k', linestyles=':', label="active threshold")
    
#     plt.xlabel(xlabel, fontsize=ticksize+2)
#     plt.ylabel(ylabel, fontsize=ticksize+2)
#     plt.tick_params(axis='both', which='major', labelsize=ticksize)
#     plt.legend(fontsize=ticksize)
#     plt.tight_layout()
#     plt.ylim([-2,20])
#     if fname:
#         try:
#             save_plot(fname)  # your helper, if present
#         except NameError:
#             plt.savefig(fname, dpi=200)
#         plt.close(fig)
#     else:
#         plt.show()
#     plt.show()



def plot_firing_rate(timepoints, firing_rate, lab, fname=None,
                     xlabel="Time", ylabel="Firing Rate (Hz)", 
                     c="r", threshold=5, ticksize=PLOT_TICK_FONTSIZE,title_fontsize=PLOT_LABEL_FONTSIZE,
                     ylim=None):
    """
    Plot mean firing rate over time with a 95% CI band and an active threshold line.

    Parameters:
        timepoints : array-like, shape (n_timepoints,)
            Time axis values.
        firing_rate : array-like
            If 1D: firing rate values to plot directly.
            If 2D: shape (n_trials, n_timepoints); mean and 95% CI will be computed across trials.
        lab : str
            Label for the firing rate curve.
        fname : str or None
            If provided, figure will be saved to this path.
        xlabel, ylabel : str
            Axis labels.
        c : str
            Color for the firing rate curve and CI band.
        threshold : float
            y-value for the horizontal dashed threshold line.
        ticksize : int
            Font size for axis ticks.
    """

    firing_rate = np.asarray(firing_rate)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Handle 1D vs 2D input
    if firing_rate.ndim == 1:
        mean_fr = firing_rate
        ci95 = None
    elif firing_rate.ndim == 2:
        # mean across trials
        mean_fr = firing_rate.mean(axis=0)
        # SEM and 95% CI (normal approximation)
        n_trials = firing_rate.shape[0]
        sem = firing_rate.std(axis=0, ddof=1) / np.sqrt(n_trials)
        ci95 = 1.96 * sem
    else:
        raise ValueError("firing_rate must be 1D or 2D (trials x timepoints).")

    # Plot mean
    ax.plot(timepoints, mean_fr, label=lab, color=c)

    # Plot 95% CI band if available
    if ci95 is not None:
        ax.fill_between(timepoints,
                        mean_fr - ci95,
                        mean_fr + ci95,
                        color=c,
                        alpha=0.3,
                        edgecolor='none',
                        label="95% CI")

    # Threshold line
    ax.hlines(y=threshold, xmin=timepoints[0], xmax=timepoints[-1],
              colors='k', linestyles=':', label="active threshold")

    # Cosmetics
    style_axis(ax, xlabel=xlabel, ylabel=ylabel,
               label_fontsize=title_fontsize, tick_fontsize=ticksize)
    if ylim is None:
        ylim = [-2, 20]
    ax.set_ylim(ylim)
    ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
    fig.tight_layout()

    # Save or show
    if fname:
        try:
            save_plot(fname)  # your helper, if present
        except NameError:
            fig.savefig(fname, dpi=200)
        plt.close(fig)
    else:
        plt.show()

def plot_tagged_activity(activity, threshold, method='mean'):
    """
    Plot average activity of tagged vs non-tagged neurons over time.

    Parameters
    ----------
    activity : np.ndarray
        Shape (N, T) matrix (neurons × time)
    threshold : float
        Threshold for tagging neurons.
    method : {'mean', 'max', 'median'}, optional
        How to determine tagging. Default is 'mean'.
    """

    N, T = activity.shape

    # --- Decide tagging based on chosen method ---
    if method == 'mean':
        tag_metric = activity.mean(axis=1)
    elif method == 'max':
        tag_metric = activity.max(axis=1)
    elif method == 'median':
        tag_metric = np.median(activity, axis=1)
    else:
        raise ValueError("method must be 'mean', 'max', or 'median'")

    tagged_neurons = tag_metric >= threshold
    non_tagged_neurons = ~tagged_neurons

    # --- Compute average activity over time ---
    tagged_activity = activity[tagged_neurons].mean(axis=0)
    non_tagged_activity = activity[non_tagged_neurons].mean(axis=0)

    # --- Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(tagged_activity, label=f"Tagged (n={tagged_neurons.sum()})", color="tab:blue", linewidth=2)
    plt.plot(non_tagged_activity, label=f"Non-tagged (n={non_tagged_neurons.sum()})", color="tab:gray", linewidth=2)
    style_axis(plt.gca(), title=f"Average activity over time — threshold={threshold}, method={method}",
               xlabel="Time", ylabel="Activity")
    plt.legend(frameon=False, fontsize=PLOT_LEGEND_FONTSIZE)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return tagged_neurons, tagged_activity, non_tagged_activity


# --- Example usage ---
# Random data example
# N = 50  # neurons
# T = 200 # timepoints
# activity = np.random.rand(N, T)
# threshold = 0.6
# plot_tagged_activity(activity, threshold, method='mean')


def plot_engram_size(
    activity,
    threshold,
    day_labels=None,
    error="sem",          # "std" or "sem"
    jitter=0.05,
    point_alpha=0.5,
    point_size=20,
    line_alpha=0.1,
    line_width=1.0,
    error_color="red",
    capsize=4,
    figsize=(8,6),
    title="Engram size",
    fname=None,
    title_fontsize=PLOT_TITLE_FONTSIZE, 
    tick_fontsize=PLOT_TICK_FONTSIZE,
):
    """
    Strip plot with mean ± error bars per day.

    Parameters
    ----------
    counts : array-like, shape (NUM_SIM, NUM_DAYS)
        Neuron counts per simulation and day.
    day_labels : array-like or None
        Labels for days.
    error : {"std", "sem"}
        Error type to plot.
    jitter : float
        Horizontal jitter for strip plot.
    point_alpha : float
        Transparency of points.
    point_size : int
        Size of strip plot points.
    error_color : str
        Color of error bars.
    capsize : int
        Capsize for error bars.
    figsize : tuple or None
        Figure size.
    title : str
        Plot title.
    """

    counts = (activity > threshold).sum(axis=2)  # shape: (num_sim, num_days)
    num_sim, num_days = counts.shape

    if day_labels is None:
        day_labels = np.arange(num_days)

    if figsize is None:
        figsize = (max(6, 1.2 * num_days), 6)

    # Mean and error
    mean = counts.mean(axis=0)

    if error == "std":
        err = counts.std(axis=0)
    elif error == "sem":
        err = counts.std(axis=0, ddof=1) / np.sqrt(num_sim)
    else:
        raise ValueError("error must be 'std' or 'sem'")

    fig, ax = plt.subplots(figsize=figsize)
    for s in range(num_sim):
        plt.plot(
            day_labels,
            counts[s],
            color="black",
            alpha=line_alpha,
            linewidth=line_width,
            zorder=1
        )
    # Strip plot
    for d in range(num_days):
        x = np.random.normal(day_labels[d], jitter, size=num_sim)
        plt.scatter(
            x,
            counts[:, d],
            alpha=point_alpha,
            s=point_size
        )

    # Error bars (mean ± error)
    plt.errorbar(
        day_labels,
        mean,
        yerr=err,
        fmt="o",
        color=error_color,
        capsize=capsize,
        linewidth=2,
        markersize=6,
        zorder=3
    )
    style_axis(ax, title=title, xlabel="Day", ylabel="Neurons above threshold",
               title_fontsize=title_fontsize,
               label_fontsize=PLOT_LABEL_FONTSIZE,
               tick_fontsize=tick_fontsize)
    plt.ylim(0, counts.max() + 1)
    plt.tight_layout()
    if fname:
        try:
            save_plot(fname)  # your helper, if present
        except NameError:
            plt.savefig(fname, dpi=200)
        # plt.close(fig)
   
    plt.show()
    return counts
    # plt.show()

def plot_average_freezing_boxplot(
    day_freezing,
    fname,
    xlabels=None,
    day0_comparisons=None,
    title="Average freezing by day",
    ylabel="Freezing (%)",
    ylim=(0, 110),
    star_y=70,
    f_size=PLOT_LABEL_FONTSIZE,
    tick_fontsize=PLOT_TICK_FONTSIZE
):
    """Plot average percent freezing across simulations as bar + strip plot."""
    day_freezing = np.asarray(day_freezing, dtype=float)
    if day_freezing.ndim != 2:
        raise ValueError("day_freezing must have shape (sims, days)")

    n_days = day_freezing.shape[1]
    if xlabels is None:
        xlabels = [str(day) for day in range(n_days)]

    plot_data = pd.DataFrame({
        "Day": np.repeat(xlabels, day_freezing.shape[0]),
        "Freezing (%)": day_freezing.T.reshape(-1),
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=plot_data,
        x="Day",
        y="Freezing (%)",
        order=xlabels,
        ax=ax,
        color="#e8eef7",
        edgecolor="#1f1f1f",
        errorbar="se",
        # capsize=0.2,
        err_kws={"color": "#1f1f1f", "linewidth": 2},
    )
    sns.stripplot(
        data=plot_data,
        x="Day",
        y="Freezing (%)",
        order=xlabels,
        ax=ax,
        color="#1f1f1f",
        alpha=0.65,
        size=5,
        jitter=0.18,
    )
    ax.spines[["right", "top"]].set_visible(False)
    style_axis(ax, title=title, xlabel="Day", ylabel=ylabel,
               title_fontsize=f_size, label_fontsize=f_size, tick_fontsize=tick_fontsize)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if day0_comparisons is not None:
        for comparison in day0_comparisons:
            stars = comparison.get("stars", "")
            if not stars:
                continue

            day = comparison["day"]
            if day < 0 or day >= n_days:
                continue

            x = day
            ax.text(x, star_y, stars, ha="center", va="bottom", fontsize=f_size, color="black")
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.tight_layout()
    save_plot(fname)
    plt.close(fig)
