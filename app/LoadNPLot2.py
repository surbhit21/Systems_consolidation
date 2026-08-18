import numpy as np
from plotting_widget import *
import json
import os
import pandas as pd
from Utilities import (
    average_freezing_by_day,
    compare_freezing_to_day0,
    normalize_freezing_to_day0,
    output_firing_rate_to_freezing,
)

# choose the desired format (0 for PDF, 1 for PNG)

def _resolve_sim_folder(sim_folder, data_root="./data"):
    """Return a data-folder path from either an explicit path or a folder name."""
    if os.path.isdir(sim_folder):
        return sim_folder
    if sim_folder.startswith("/data/"):
        sim_folder = sim_folder[len("/data/"):]
    candidate = os.path.join(data_root, sim_folder)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(
        "Could not find simulation folder '{}' or '{}'".format(sim_folder, candidate)
    )


def _resolve_comparison_pairs(n_groups, comparison_pairs, control_index):
    """Validate requested index pairs, or default to control-vs-all pairs."""
    if comparison_pairs is None:
        if control_index < 0 or control_index >= n_groups:
            raise ValueError("control_index is outside the range of plotted groups")
        return [(control_index, idx) for idx in range(n_groups) if idx != control_index]

    resolved_pairs = []
    for pair in comparison_pairs:
        if len(pair) != 2:
            raise ValueError("Each comparison pair must contain exactly two indexes")
        index1, index2 = (int(pair[0]), int(pair[1]))
        if index1 == index2:
            raise ValueError("A comparison pair must contain two different indexes")
        if not (0 <= index1 < n_groups and 0 <= index2 < n_groups):
            raise ValueError(
                "Comparison pair ({}, {}) is outside the range 0-{}".format(
                    index1, index2, n_groups - 1
                )
            )
        resolved_pairs.append((index1, index2))
    return resolved_pairs


def _apply_holm_correction(comparisons, pvalue_to_stars, alpha=0.05):
    """Apply Holm's step-down correction to a list of pairwise results."""
    valid = [
        (idx, comparison["pvalue"])
        for idx, comparison in enumerate(comparisons)
        if not np.isnan(comparison["pvalue"])
    ]
    correction = "holm" if len(valid) > 1 else "none"
    adjusted = {}
    running_max = 0.0
    for rank, (idx, pvalue) in enumerate(sorted(valid, key=lambda item: item[1])):
        corrected = min(1.0, (len(valid) - rank) * pvalue)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max

    for idx, comparison in enumerate(comparisons):
        corrected_pvalue = adjusted.get(idx, np.nan)
        comparison["pvalue_raw"] = comparison["pvalue"]
        comparison["pvalue_corrected"] = corrected_pvalue
        comparison["correction"] = correction
        comparison["stars"] = pvalue_to_stars(corrected_pvalue)
        comparison["significant"] = bool(
            False if np.isnan(corrected_pvalue) else corrected_pvalue < alpha
        )
    return correction


def _load_or_compute_day_freezing(
    sim_folder,
    n_presentations=10,
    normalized=False,
):
    """Load saved day-wise freezing, or compute it from output firing rates."""
    suffix = "" if n_presentations is None else "_last{}presentations".format(n_presentations)
    prefix = "normalized_freezing_by_day" if normalized else "average_freezing_by_day"
    freezing_path = os.path.join(sim_folder, "{}{}.npy".format(prefix, suffix))
    if os.path.exists(freezing_path):
        return np.load(freezing_path)

    params_path = os.path.join(sim_folder, "all_params.json")
    op_path = os.path.join(sim_folder, "FR_history_op.npy")
    if not os.path.exists(params_path) or not os.path.exists(op_path):
        raise FileNotFoundError(
            "Could not load '{}' and could not recompute because all_params.json "
            "or FR_history_op.npy is missing in '{}'.".format(freezing_path, sim_folder)
        )

    with open(params_path, "r") as f:
        params = json.load(f)["simulation_params"]

    freezing_fr_max = params.get("freezing_fr_max", 10.0)
    op_activity = np.load(op_path)
    if op_activity.ndim == 3:
        op_activity = op_activity[:, :, 0]

    freezing_history = output_firing_rate_to_freezing(
        op_activity,
        freezing_fr_max=freezing_fr_max,
    )
    day_freezing = average_freezing_by_day(
        freezing_history,
        ID=params["ID"],
        N_off_days=params["N_off_days"],
        Nrep=params["Nrep"],
        t_off=params["t_off"],
        IR=params["IR"],
        last_n_presentations=n_presentations,
    )
    if normalized:
        day_freezing = normalize_freezing_to_day0(day_freezing)
    return day_freezing


def compare_freezing_on_day(
    sim_folders,
    day,
    labels=None,
    data_root="./data",
    n_presentations=10,
    normalized=False,
    fname=None,
    title=None,
    alpha=0.05,
    annotate_vs_control=True,
    control_index=0,
    comparison_pairs=None,
    show_ns=True,
    bar_colors="#e8eef7",
    xtick_labels=None,
    x_group_labels=None,
):
    """
    Compare freezing percentage on one day across multiple simulation folders.

    Parameters
    ----------
    sim_folders : list[str]
        Folder names under data_root, or explicit data-folder paths.
    day : int
        Day index to compare, e.g. 10 for day 10.
    labels : list[str] or None
        Labels to use on the x-axis. Defaults to folder basenames.
    data_root : str
        Root directory used when sim_folders contains folder names.
    n_presentations : int or None
        Which saved average to load, e.g. 10 loads
        average_freezing_by_day_last10presentations.npy. If the file is missing,
        the function recomputes it from FR_history_op.npy.
    normalized : bool
        If True, compare normalized freezing values instead of raw percent.
    fname : str or None
        If provided, save a bar+strip plot to this path using save_plot().
    title : str or None
        Plot title.
    alpha : float
        Significance threshold used in the optional omnibus test.
    annotate_vs_control : bool
        If True, add pairwise comparisons against the control group.
    control_index : int
        Index of the control group in sim_folders/labels.
    comparison_pairs : list[tuple[int, int]] or None
        Index pairs to test and annotate, e.g. [(0, 1), (1, 2)]. If None,
        compare control_index against every other group. Holm correction is
        applied when more than one pair is tested.
    show_ns : bool
        If True, annotate non-significant pairwise comparisons as "n.s.".
    bar_colors : str or list[str]
        Outline color for all boxes, or one color per simulation/group.
    xtick_labels : list[str] or None
        Optional labels to display on the x-axis. Must match the number of
        plotted groups if provided.
    x_group_labels : list[tuple] or list[dict] or None
        Optional grouped labels drawn below the x-axis. Each entry can be
        ("LTP erasure", 1, 2) or {"label": "LTP erasure", "start": 1, "end": 2}.

    Returns
    -------
    plot_data : pandas.DataFrame
        Long-format values with columns Simulation, SimulationFolder, Day,
        Freezing, and SimIndex.
    stats_summary : dict
        Mean/SEM per group plus an optional across-group statistical test.
    """
    if labels is not None and len(labels) != len(sim_folders):
        raise ValueError("labels must have the same length as sim_folders")

    rows = []
    group_values = {}
    for idx, sim_folder in enumerate(sim_folders):
        resolved_folder = _resolve_sim_folder(sim_folder, data_root=data_root)
        label = labels[idx] if labels is not None else os.path.basename(resolved_folder)
        day_freezing = _load_or_compute_day_freezing(
            resolved_folder,
            n_presentations=n_presentations,
            normalized=normalized,
        )
        if day < 0 or day >= day_freezing.shape[1]:
            raise ValueError(
                "Requested day {} but '{}' has {} days".format(
                    day, resolved_folder, day_freezing.shape[1]
                )
            )

        values = np.asarray(day_freezing[:, day], dtype=float)
        group_values[label] = values
        for sim_idx, value in enumerate(values):
            rows.append({
                "Simulation": label,
                "SimulationFolder": resolved_folder,
                "Day": day,
                "Freezing": value,
                "SimIndex": sim_idx,
            })

    plot_data = pd.DataFrame(rows)
    summary = {}
    for label, values in group_values.items():
        valid = values[~np.isnan(values)]
        summary[label] = {
            "n": int(valid.size),
            "mean": float(np.nanmean(values)),
            "sem": float(np.nanstd(values) / np.sqrt(valid.size)) if valid.size else np.nan,
            "values": values.tolist(),
        }

    stats_summary = {
        "day": int(day),
        "n_presentations": n_presentations,
        "normalized": bool(normalized),
        "groups": summary,
    }

    def pvalue_to_stars(pvalue):
        if np.isnan(pvalue):
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        if pvalue < 0.05:
            return "*"
        return "n.s." if show_ns else ""

    pairwise_comparisons = []
    try:
        from scipy import stats
        valid_groups = [values[~np.isnan(values)] for values in group_values.values()]
        if len(valid_groups) == 2:
            result = stats.mannwhitneyu(valid_groups[0], valid_groups[1], alternative="two-sided")
            test = "mannwhitneyu"
        elif len(valid_groups) > 2:
            result = stats.kruskal(*valid_groups)
            test = "kruskal"
        else:
            result = None
            test = "not_enough_groups"

        if result is not None:
            stats_summary["test"] = {
                "name": test,
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
                "alpha": alpha,
                "significant": bool(result.pvalue < alpha),
            }
        else:
            stats_summary["test"] = {"name": test}

        group_labels = list(group_values.keys())
        if annotate_vs_control and len(group_labels) > 1:
            index_pairs = _resolve_comparison_pairs(
                len(group_labels), comparison_pairs, control_index
            )
            for index1, index2 in index_pairs:
                label1, label2 = group_labels[index1], group_labels[index2]
                values1 = group_values[label1]
                values2 = group_values[label2]
                values1 = values1[~np.isnan(values1)]
                values2 = values2[~np.isnan(values2)]
                if values1.size == 0 or values2.size == 0:
                    pvalue = np.nan
                    statistic = np.nan
                else:
                    result = stats.mannwhitneyu(values1, values2, alternative="two-sided")
                    pvalue = float(result.pvalue)
                    statistic = float(result.statistic)
                pairwise_comparisons.append({
                    "group1": label1,
                    "group2": label2,
                    "index1": index1,
                    "index2": index2,
                    "n1": int(values1.size),
                    "n2": int(values2.size),
                    "test": "mannwhitneyu",
                    "statistic": statistic,
                    "pvalue": pvalue,
                    "stars": pvalue_to_stars(pvalue),
                    "alpha": alpha,
                    "significant": bool(False if np.isnan(pvalue) else pvalue < alpha),
                })
            correction = _apply_holm_correction(
                pairwise_comparisons, pvalue_to_stars, alpha=alpha
            )
            stats_summary["comparisons"] = pairwise_comparisons
            stats_summary["multiple_comparison_correction"] = correction
    except ImportError:
        stats_summary["test"] = {"name": "scipy_unavailable"}
        stats_summary["comparisons"] = []

    if fname is not None:
        ylabel = "Freezing (% of day 0)" if normalized else "Freezing (%)"
        if title is None:
            title = "Freezing on day {}".format(day)
        group_order = list(group_values.keys())
        fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(group_order)), 4))
        if isinstance(bar_colors, str):
            bar_palette = [bar_colors] * len(group_order)
        else:
            if len(bar_colors) != len(group_order):
                raise ValueError(
                    "bar_colors must be a single color or have one color per group. "
                    "Got {} colors for {} groups.".format(len(bar_colors), len(group_order))
                )
            bar_palette = list(bar_colors)
        for group_idx, (group_label, box_color) in enumerate(zip(group_order, bar_palette)):
            bp = ax.boxplot(
                group_values[group_label],
                positions=[group_idx],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            setBoxColors(bp, box_color)
            plt.setp(bp["boxes"], facecolor="none", linewidth=2)
        sns.stripplot(
            data=plot_data,
            x="Simulation",
            y="Freezing",
            order=group_order,
            ax=ax,
            color="#1f1f1f",
            alpha=0.65,
            size=5,
            jitter=0.18,
            marker="^",
        )
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylim([0,100])
        style_axis(ax, title=title, xlabel="", ylabel=ylabel)
        if xtick_labels is not None:
            if len(xtick_labels) != len(group_order):
                raise ValueError(
                    "xtick_labels must have one label per group. "
                    "Got {} labels for {} groups.".format(len(xtick_labels), len(group_order))
                )
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(xtick_labels)

        if x_group_labels is not None:
            xaxis_transform = ax.get_xaxis_transform()
            bracket_y = -0.18
            tick_y = -0.14
            text_y = -0.28
            for group_label in x_group_labels:
                if isinstance(group_label, dict):
                    label = group_label["label"]
                    start = group_label["start"]
                    end = group_label["end"]
                else:
                    label, start, end = group_label
                ax.plot(
                    [start, end],
                    [bracket_y, bracket_y],
                    transform=xaxis_transform,
                    color="black",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.plot(
                    [start, start],
                    [bracket_y, tick_y],
                    transform=xaxis_transform,
                    color="black",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.plot(
                    [end, end],
                    [bracket_y, tick_y],
                    transform=xaxis_transform,
                    color="black",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.text(
                    (start + end) / 2,
                    text_y,
                    label,
                    transform=xaxis_transform,
                    ha="center",
                    va="top",
                    fontsize=PLOT_LABEL_FONTSIZE,
                    clip_on=False,
                )
        # ax.tick_params(axis="x", rotation=30)
        if annotate_vs_control and pairwise_comparisons:
            y_values = plot_data["Freezing"].to_numpy(dtype=float)
            y_max = np.nanmax(y_values)
            y_min = np.nanmin(y_values)
            y_range = max(y_max - y_min, 1.0)
            bracket_height = 0.04 * y_range
            y_step = 0.12 * y_range
            y_start = y_max + 0.08 * y_range
            annotation_count = 0
            for comparison in pairwise_comparisons:
                stars = comparison["stars"]
                if not stars:
                    continue
                x0 = comparison["index1"]
                x1 = comparison["index2"]
                y = y_start + annotation_count * y_step
                ax.plot(
                    [x0, x0, x1, x1],
                    [y, y + bracket_height, y + bracket_height, y],
                    color="black",
                    linewidth=1.5,
                    clip_on=False,
                )
                ax.text(
                    (x0 + x1) / 2,
                    y + bracket_height,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=PLOT_LABEL_FONTSIZE,
                    color="black",
                    clip_on=False,
                )
                annotation_count += 1
            ax.set_ylim([0, 100])
        if x_group_labels is not None:
            fig.tight_layout(rect=[0, 0.12, 1, 1])
        else:
            fig.tight_layout()
        save_plot(fname)
        plt.close(fig)

    return plot_data, stats_summary


def load_goto_fig2c_crossover_latency(
    data_folder="./Published_data/Goto_Science_2021",
    file_names=None,
    condition_order=None,
):
    """
    Load digitized Goto et al. Fig. 2C cross-over latency data.

    The expected files are CSVs, e.g.
    Fig2C_CNT_D2Recall.csv and Fig2C_CALI_D2Recall.csv. Each CSV is expected
    to have no header and two columns: a digitized point label and the
    cross-over latency value for one animal. The condition labels are supplied
    explicitly via condition_order, in the same order as file_names.

    Parameters
    ----------
    data_folder : str
        Folder containing the digitized CSV files.
    file_names : list[str] or None
        CSV filenames to load. Relative filenames are resolved inside
        data_folder; absolute or explicit paths are used directly. If None,
        defaults to the two Fig. 2C day-2 recall files.
    condition_order : list[str] or None
        Condition labels corresponding to file_names. Must have the same length
        as file_names. Also defines the plotting order.

    Returns
    -------
    plot_data : pandas.DataFrame
        Long-format data with columns Condition, SourceFile, Point,
        CrossoverLatency, and AnimalIndex.
    summary : dict
        Per-condition n, mean, SEM, and raw values.
    """
    if file_names is None:
        file_names = [
            "Fig2C_CNT_D2Recall.csv",
            "Fig2C_CALI_D2Recall.csv",
        ]
    if condition_order is None:
        condition_order = ["Control", "CALI"]
    if len(file_names) != len(condition_order):
        raise ValueError(
            "file_names and condition_order must have the same length. "
            "Got {} file_names and {} condition labels.".format(
                len(file_names),
                len(condition_order),
            )
        )

    rows = []
    for csv_file, condition in zip(file_names, condition_order):
        if not os.path.isabs(csv_file) and not os.path.exists(csv_file):
            csv_file = os.path.join(data_folder, csv_file)
        if not os.path.exists(csv_file):
            raise FileNotFoundError("Could not find '{}'".format(csv_file))
        raw = pd.read_csv(csv_file, header=None, names=["Point", "CrossoverLatency"])
        raw["CrossoverLatency"] = pd.to_numeric(raw["CrossoverLatency"], errors="coerce")
        raw = raw.dropna(subset=["CrossoverLatency"]).reset_index(drop=True)
        raw["CrossoverLatency"] = raw["CrossoverLatency"].clip(lower=0)
        for animal_idx, row in raw.iterrows():
            rows.append({
                "Condition": condition,
                "SourceFile": csv_file,
                "Point": row["Point"],
                "CrossoverLatency": float(row["CrossoverLatency"]),
                "AnimalIndex": int(animal_idx),
            })

    plot_data = pd.DataFrame(rows)
    if plot_data.empty:
        raise ValueError(
            "CSV files were found, but no numeric latency values could be read."
        )

    summary = {}
    for condition, condition_data in plot_data.groupby("Condition", sort=False):
        values = condition_data["CrossoverLatency"].to_numpy(dtype=float)
        valid = values[~np.isnan(values)]
        summary[condition] = {
            "n": int(valid.size),
            "mean": float(np.nanmean(values)),
            "sem": float(np.nanstd(values) / np.sqrt(valid.size)) if valid.size else np.nan,
            "values": values.tolist(),
        }
    return plot_data, summary


def plot_goto_fig2c_crossover_latency(
    data_folder="./Published_data/Goto_Science_2021",
    file_names=None,
    condition_order=None,
    fname=None,
    title="Goto et al. Fig. 2C",
    ylabel="Change in cross-over latency (s)",
    annotate=True,
    show_ns=True,
    bar_colors="#f2e7da",
    control_index=0,
    comparison_pairs=None,
    xtick_labels=None,
    x_group_labels=None,
):
    """
    Plot digitized Goto et al. Fig. 2C cross-over latency data.

    Parameters
    ----------
    data_folder : str
        Folder containing digitized CSV files.
    file_names : list[str] or None
        CSV filenames to load. Relative filenames are resolved inside
        data_folder. If None, defaults to the two Fig. 2C day-2 recall files.
    condition_order : list[str] or None
        Condition labels corresponding to file_names. Must have the same length
        as file_names and defines the x-axis order.
    fname : str or None
        If provided, save the figure using save_plot(fname).
    title : str
        Plot title.
    ylabel : str
        Y-axis label. Use "Cross-over latency (s)" if your digitized values are
        raw latencies rather than change scores.
    annotate : bool
        If True, add a Mann-Whitney U significance annotation for two groups.
    show_ns : bool
        If True, annotate non-significant comparisons as "n.s.".
    bar_colors : str or list[str]
        Fill color for all boxes, or one color per condition.
    control_index : int
        Index of the control condition used for pairwise annotations.
    comparison_pairs : list[tuple[int, int]] or None
        Index pairs to test and annotate, e.g. [(0, 1), (1, 2)]. If None,
        compare control_index against every other condition. Holm correction
        is applied when more than one pair is tested.
    xtick_labels : list[str] or None
        Optional labels to display on the x-axis. Must match condition_order
        length if provided. This lets the plotted ticks be shorter than the
        condition names used in the stats summary.
    x_group_labels : list[tuple] or list[dict] or None
        Optional grouped labels drawn below the x-axis. Each entry can be
        ("CALI", 1, 2) or {"label": "CALI", "start": 1, "end": 2}.

    Returns
    -------
    plot_data : pandas.DataFrame
        Long-format data returned by load_goto_fig2c_crossover_latency().
    stats_summary : dict
        Summary statistics and optional Mann-Whitney U test.
    """
    plot_data, group_summary = load_goto_fig2c_crossover_latency(
        data_folder=data_folder,
        file_names=file_names,
        condition_order=condition_order,
    )
    order = list(plot_data["Condition"].drop_duplicates())

    stats_summary = {
        "data_folder": data_folder,
        "file_names": list(file_names) if file_names is not None else [
            "Fig2C_CNT_D2Recall.csv",
            "Fig2C_CALI_D2Recall.csv",
        ],
        "condition_order": order,
        "groups": group_summary,
    }

    def pvalue_to_stars(pvalue):
        if np.isnan(pvalue):
            return ""
        if pvalue < 0.001:
            return "***"
        if pvalue < 0.01:
            return "**"
        if pvalue < 0.05:
            return "*"
        return "n.s." if show_ns else ""

    pairwise_comparisons = []
    if annotate and len(order) > 1:
        index_pairs = _resolve_comparison_pairs(len(order), comparison_pairs, control_index)
        try:
            from scipy import stats
            for index1, index2 in index_pairs:
                label1, label2 = order[index1], order[index2]
                values1 = plot_data.loc[
                    plot_data["Condition"] == label1, "CrossoverLatency"
                ].to_numpy(dtype=float)
                values2 = plot_data.loc[
                    plot_data["Condition"] == label2, "CrossoverLatency"
                ].to_numpy(dtype=float)
                result = stats.mannwhitneyu(
                    values1,
                    values2,
                    alternative="two-sided",
                )
                pairwise_comparisons.append({
                    "test": "mannwhitneyu",
                    "group1": label1,
                    "group2": label2,
                    "index1": index1,
                    "index2": index2,
                    "n1": int(values1.size),
                    "n2": int(values2.size),
                    "statistic": float(result.statistic),
                    "pvalue": float(result.pvalue),
                    "stars": pvalue_to_stars(float(result.pvalue)),
                })
            correction = _apply_holm_correction(
                pairwise_comparisons, pvalue_to_stars, alpha=0.05
            )
            stats_summary["comparisons"] = pairwise_comparisons
            stats_summary["multiple_comparison_correction"] = correction
            if len(pairwise_comparisons) == 1:
                stats_summary["comparison"] = pairwise_comparisons[0]
        except ImportError:
            stats_summary["comparison"] = {"test": "scipy_unavailable"}
            stats_summary["comparisons"] = []

    if fname is not None:
        if isinstance(bar_colors, str):
            bar_palette = [bar_colors] * len(order)
        else:
            if len(bar_colors) != len(order):
                raise ValueError(
                    "bar_colors must be a single color or have one color per condition. "
                    "Got {} colors for {} conditions.".format(len(bar_colors), len(order))
                )
            bar_palette = list(bar_colors)
        fig, ax = plt.subplots(figsize=(max(4, 1.5 * len(order)), 4))
        for condition_idx, (condition, box_color) in enumerate(zip(order, bar_palette)):
            condition_values = plot_data.loc[
                plot_data["Condition"] == condition, "CrossoverLatency"
            ].to_numpy(dtype=float)
            bp = ax.boxplot(
                condition_values,
                positions=[condition_idx],
                widths=0.6,
                patch_artist=True,
                showfliers=False,
            )
            setBoxColors(bp, box_color, flipped=True)
            plt.setp(bp["boxes"], facecolor=box_color, linewidth=2)
        sns.stripplot(
            data=plot_data,
            x="Condition",
            y="CrossoverLatency",
            order=order,
            ax=ax,
            color="#1f1f1f",
            alpha=0.7,
            size=5,
            jitter=0.18,
        )
        ax.spines[["right", "top"]].set_visible(False)
        style_axis(ax, title=title, xlabel="", ylabel=ylabel)
        if xtick_labels is not None:
            if len(xtick_labels) != len(order):
                raise ValueError(
                    "xtick_labels must have one label per condition. "
                    "Got {} labels for {} conditions.".format(len(xtick_labels), len(order))
                )
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(xtick_labels)

        if x_group_labels is not None:
            xaxis_transform = ax.get_xaxis_transform()
            bracket_y = -0.18
            tick_y = -0.14
            text_y = -0.28
            for group_label in x_group_labels:
                if isinstance(group_label, dict):
                    label = group_label["label"]
                    start = group_label["start"]
                    end = group_label["end"]
                else:
                    label, start, end = group_label
                ax.plot(
                    [start, end],
                    [bracket_y, bracket_y],
                    transform=xaxis_transform,
                    color="black",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.plot(
                    [start, start],
                    [bracket_y, tick_y],
                    transform=xaxis_transform,
                    color="black",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.plot(
                    [end, end],
                    [bracket_y, tick_y],
                    transform=xaxis_transform,
                    color="black",
                    linewidth=1.2,
                    clip_on=False,
                )
                ax.text(
                    (start + end) / 2,
                    text_y,
                    label,
                    transform=xaxis_transform,
                    ha="center",
                    va="top",
                    fontsize=PLOT_LABEL_FONTSIZE,
                    clip_on=False,
                )

        if annotate and pairwise_comparisons:
            y_values = plot_data["CrossoverLatency"].to_numpy(dtype=float)
            y_max = np.nanmax(y_values)
            y_min = np.nanmin(y_values)
            y_range = max(y_max - y_min, 1.0)
            bracket_height = 0.04 * y_range
            y_step = 0.12 * y_range
            y_start = y_max + 0.10 * y_range
            annotation_count = 0
            for comparison in pairwise_comparisons:
                stars = comparison["stars"]
                if not stars:
                    continue
                x0 = comparison["index1"]
                x1 = comparison["index2"]
                y = y_start + annotation_count * y_step
                ax.plot(
                    [x0, x0, x1, x1],
                    [y, y + bracket_height, y + bracket_height, y],
                    color="black",
                    linewidth=1.5,
                    clip_on=False,
                )
                ax.text(
                    (x0 + x1) / 2,
                    y + bracket_height,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=PLOT_LABEL_FONTSIZE,
                    color="black",
                    clip_on=False,
                )
                annotation_count += 1
            if annotation_count:
                ax.set_ylim(top=y_start + annotation_count * y_step + 0.12 * y_range)

        if x_group_labels is not None:
            fig.tight_layout(rect=[0, 0.12, 1, 1])
        else:
            fig.tight_layout()
        save_plot(fname)
        plt.close(fig)

    return plot_data, stats_summary


def PlotAll(input_data_folder="./data/IPBlock[4, 5]_fast_drift_with_limited11_IP_lowI", op_plot_folder="./plots/IPBlock[4, 5]_fast_drift_with_limited11_IP_lowI"):
    # input_data_folder = "./data/CNT_fast_drift_with_limited7_IP_lowI"
    # op_plot_folder = "./plots/CNT_fast_drift_with_limited7_IP_lowI"
    # Path to your JSON file
    json_path = "{}/all_params.json".format(input_data_folder)
    dop = 2
    with open(json_path, "r") as f:
        sim_params = json.load(f)

    # breakpoint()
    sim_params = sim_params["simulation_params"]
    # Explicitly assign variables
    n = sim_params["n"]
    # n_inp = sim_params["n_inp"]
    # n_ctx = sim_params["n_ctx"]
    E_fl = sim_params["E_fl"]
    FC_inp = sim_params["FC_inp"]
    seqA = sim_params["t_series"]
    E_fl_ctx = sim_params["E_fl_ctx"]
    threshold = sim_params["threshold"]
    ID = sim_params["ID"]
    N_off_days = sim_params["N_off_days"]
    off_days = sim_params["off_days"]
    # E_mod = sim_params["E_mod"]
    # t_off = sim_params["t_off"]
    IR = sim_params["IR"]
    Nrep = sim_params["Nrep"]
    t_off = sim_params["t_off"]
    time_per_day = Nrep * (t_off + IR) + ID
    freezing_presentations_to_average = [3, 4, 5, 6,10]
    # start_seed = sim_params["start_seed"]
    # max_e = sim_params["max_e"]
    total_time = sim_params["total_time"]
    # dt = sim_params["dt"]
    # import
    # threshold = 2 
    

    A_0 = sim_params["E_mod"]#2.4
    t1 = np.arange(0,11,1)
    tau = sim_params["tau_IE"]
    A_t = np.zeros_like(t1)
    A_t = A_0 *(np.exp(-(t1-1)/tau))
    # print(A_t,t1)
    A_t[0] = 0
    fig,ax = plt.subplots(figsize = (6,3))
    ax.plot(t1,A_t,'o-k')
    ax.hlines(y=0,xmin=0,xmax=10,linestyle='--',color = 'k',alpha = 0.5)
    ax.set_xlabel("Days",fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(r"$\Delta \mathrm{e}_{i}^{ACC}$",fontsize=PLOT_LABEL_FONTSIZE)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xticks(t1)
    ax.set_yticks(np.arange(0, A_0+0.5, 1.))
    ax.tick_params(labelsize=PLOT_TICK_FONTSIZE)
    # ax.set_yticks(fontsize = 18)
    # ax.set_yticks(fontsize = 18)
    plt.tight_layout()
    save_plot("{}/excitability_boost_decay".format(op_plot_folder))
    plt.close()
    last_activity_all = np.load("{}/last_activity.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    last_activity_ctx_all = np.load("{}/last_activity_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    FR_history_all = np.load("{}/FR_history.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    EX_history_all = np.load("{}/EX_history.npy".format(input_data_folder)) # shape:
    FR_op_history_all = np.load("{}/FR_history_op.npy".format(input_data_folder))
    freezing_fr_max = sim_params.get("freezing_fr_max", 10.0)
    print("Using freezing_fr_max:", freezing_fr_max)
    freezing_path = "{}/freezing_history.npy".format(input_data_folder)
    if os.path.exists(freezing_path):
        freezing_history_all = np.load(freezing_path)
    else:
        freezing_history_all = output_firing_rate_to_freezing(
            FR_op_history_all[:, :, 0],
            freezing_fr_max=freezing_fr_max,
        )
    recall_op_path = "{}/recall_activity_op.npy".format(input_data_folder)
    if os.path.exists(recall_op_path):
        recall_op_activity_all = np.load(recall_op_path)
        recall_freezing_all = output_firing_rate_to_freezing(
            recall_op_activity_all[:, :, 0],
            freezing_fr_max=freezing_fr_max,
        )
        recall_days = sim_params.get("recall_days", [])
        n_recall = sim_params.get("N_recall", recall_freezing_all.shape[1])
        if len(recall_days) > 0 and recall_freezing_all.shape[1] == len(recall_days) * n_recall:
            recall_labels = [ 
                "D{}-{}".format(day, rep + 1)
                for day in recall_days
                for rep in range(n_recall)
            ]
        else:
            recall_labels = [str(i + 1) for i in range(recall_freezing_all.shape[1])]
        np.save("{}/recall_freezing.npy".format(input_data_folder), recall_freezing_all)
        plot_average_freezing_boxplot(
            recall_freezing_all,
            fname="{}/recall_freezing_boxplot".format(op_plot_folder),
            xlabels=recall_labels,
            title="Frozen-weight recall",
            ylabel="Freezing (%)",
            ylim=(0, 100),
        )
    # breakpoint()
    # breakpoint()

    last_activity_all_ctx = np.load("{}/last_activity_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    FR_history_all_ctx = np.load("{}/FR_history_ctx.npy".format(input_data_folder)) # shape: (sims, time, neurons)
    EX_history_all_ctx = np.load("{}/EX_history_ctx.npy".format(input_data_folder)) # shape: 
    input_history = np.load("{}/input_history.npy".format(input_data_folder),allow_pickle=True)
    FR_history_th = (FR_history_all > threshold).astype(float)*FR_history_all
    FR_history_th_ctx = (FR_history_all_ctx > threshold).astype(float)*FR_history_all_ctx
    last_activity_th = (last_activity_all > threshold).astype(float)*last_activity_all
    last_activity_th_ctx = (last_activity_all_ctx > threshold).astype(float)*last_activity_all_ctx
    plot_engram_size(last_activity_all, threshold=threshold, title = "Engram size (HPC)",fname="{}/engram_size_HPC".format(op_plot_folder))
    plot_engram_size(last_activity_all_ctx, threshold=threshold,title = "Engram size (ACC)", fname="{}/engram_size_ACC".format(op_plot_folder))

    # breakpoint()
    # total_time = 22000
    plot_corr_matrix(last_activity_th[0], fname="{}/corr_matrix".format(op_plot_folder))
    plot_corr_matrix(last_activity_th_ctx[0], fname="{}/corr_matrix_ctx".format(op_plot_folder))
    timepoints = np.arange(0,total_time,1)*1
    plot_firing_rate(timepoints, freezing_history_all,lab = "Freezing",
                    xlabel="Time (s)", ylabel="Freezing (%)", c="r",fname= "{}/freezing_level".format(op_plot_folder),threshold=80, ylim=[0, 200])
    for n_presentations in freezing_presentations_to_average:
        suffix = "last{}presentations".format(n_presentations)
        day_freezing = average_freezing_by_day(
            freezing_history_all,
            ID=ID,
            N_off_days=N_off_days,
            Nrep=Nrep,
            t_off=t_off,
            IR=IR,
            last_n_presentations=n_presentations,
        )
        np.save("{}/average_freezing_by_day_{}.npy".format(input_data_folder, suffix), day_freezing)
        freezing_day_stats = compare_freezing_to_day0(day_freezing)
        freezing_day_stats = {
            "presentations_averaged": n_presentations,
            "comparisons": freezing_day_stats,
        }
        with open("{}/average_freezing_by_day_stats_{}.json".format(input_data_folder, suffix), "w") as f:
            json.dump(freezing_day_stats, f, indent=4)
        plot_average_freezing_boxplot(
            day_freezing,
            fname="{}/average_freezing_by_day_boxplot_{}".format(op_plot_folder, suffix),
            xlabels=[str(day) for day in range(N_off_days)],
            day0_comparisons=freezing_day_stats["comparisons"],
            title="Average freezing by day (last {} presentations)".format(n_presentations),
        )

        normalized_day_freezing = normalize_freezing_to_day0(day_freezing)
        np.save(
            "{}/normalized_freezing_by_day_{}.npy".format(input_data_folder, suffix),
            normalized_day_freezing,
        )
        normalized_freezing_day_stats = compare_freezing_to_day0(normalized_day_freezing)
        normalized_freezing_day_stats = {
            "presentations_averaged": n_presentations,
            "comparisons": normalized_freezing_day_stats,
        }
        with open("{}/normalized_freezing_by_day_stats_{}.json".format(input_data_folder, suffix), "w") as f:
            json.dump(normalized_freezing_day_stats, f, indent=4)
        plot_average_freezing_boxplot(
            normalized_day_freezing,
            fname="{}/normalized_freezing_by_day_boxplot_{}".format(op_plot_folder, suffix),
            xlabels=[str(day) for day in range(N_off_days)],
            day0_comparisons=normalized_freezing_day_stats["comparisons"],
            title="Freezing normalized to day 0 (last {} presentations)".format(n_presentations),
            ylabel="Freezing (% of day 0)",
            ylim=(0, 140),
            star_y=125,
        )
    # breakpoint()
    plot_firing_rate(timepoints, FR_op_history_all[:, :, 0],lab = "Output neuron",
                    xlabel="Time (s)", ylabel="Firing Rate (Hz)", c="r",fname= "{}/OP_neuron_activity".format(op_plot_folder),threshold=8)
    sim_to_plot = 3
    plot_activity_n_excitability_time([FR_history_th[sim_to_plot].T,FR_history_th_ctx[sim_to_plot].T],
                        titles=['Neuronal Activity (HPC) \n Late IP blocking',
                                    'Neuronal Activity (ACC) \n Late IP blocking'],
                        seqA=seqA,
                        fname="{}/Activity".format(op_plot_folder),
                        cmaps=['OrRd', 'Blues'],
                        time_per_day=time_per_day,
                        day_zero_time=ID,
                        input_history=input_history[sim_to_plot])


    plot_activity_n_excitability_time([EX_history_all[sim_to_plot].T,EX_history_all_ctx[sim_to_plot].T,input_history[sim_to_plot].T],
                        titles=['Neuronal Excitability (HPC)',
                                    'Neuronal Excitability (ACC)',
                                    'Input Activity'],
                        seqA=seqA,
                        fname="{}/Excitability".format(op_plot_folder),
                        cmaps=['Oranges', 'Blues','Grays'],
                        time_per_day=time_per_day,
                        day_zero_time=ID,
                        colorbar_label=None,
                        input_history=input_history[sim_to_plot])
    # labs = ["FC"] t [f"Off {it1}" for i in range(N_off_days)]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="./plots/Reimagined/Rec_w",
    #                        cmap='gray_r')

    cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    xlabs = [f"{i}" for i in range(1, N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std".format(op_plot_folder),
        cmap = "Oranges",
        marker = "^"
    )
    mean_corr_ctx, std_corr_ctx, per_sim_corr_ctx, idx_ctx = plot_mean_std_corr_over_time(
        last_activity_ctx_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_ctx".format(op_plot_folder),
        cmap = "Blues"
    )
    mean_DR = np.mean(1 - mean_corr)
    mean_DR_ctx = np.mean(1 - mean_corr_ctx)
    print("excitability boosts:", E_fl, E_fl_ctx)
    print("Normalized drift rate:", mean_DR)
    print("Normalized drift rate:", mean_DR_ctx)

    plot_population_vector_correlations(
        [mean_corr, mean_corr_ctx],
        [std_corr, std_corr_ctx],
        xlabels=[i for i in range(1, N_off_days)],
        labels=["HPC", "ACC"],
        fname="{}/encoding_vs_others_mean_std_HPC_ACC".format(op_plot_folder),
        title="Cell population \n activity correlation",
        colors=["tab:orange", "tab:blue"],
        markers=["^", "o"],
    )
    xlabs = [f"{i-dop}" for i in range(N_off_days)]
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std".format(op_plot_folder),
        cmap = "Oranges",
        marker = "^"
    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ctx_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std_ctx".format(op_plot_folder),
        cmap = "Blues"
    )

    xlabs = [f"{i}" for i in range(N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std".format(op_plot_folder),
        cmap = "Oranges",
        marker = "^"

    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ctx_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std_ctx".format(op_plot_folder),
        cmap = "Blues",
        

    )


    labs = [f"Day {i}" for i in off_days]
    all_weights_files = [
        "rec_weights.npy",
        "rec_ctx_weights.npy",
        "mtl_op_weights.npy",
        "ctx_op_weights.npy",
        # "mtl_ctx_weights.npy",
        # "ctx_mtl_weights.npy"
    ]
    for weights_file in all_weights_files:
        weights_all = np.load("{}/{}".format(input_data_folder, weights_file))
        use_inset = 1 if weights_file in ["rec_weights.npy", "rec_ctx_weights.npy"] else 0
        plot_weights_over_time(weights_all[-1,off_days],
                            titles=  labs,
                            fname="{}/{}".format(op_plot_folder, weights_file.split('.')[0]),
                            cmaps='gray_r',
                            in_set=use_inset)
    # rec_weights_all = np.load("{}/rec_weights.npy".format(input_data_folder))
    # plot_weights_over_time(rec_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/Rec_w".format(op_plot_folder),
    #                     cmaps='gray_r')

    # rec_ctx_weights_all = np.load("{}/rec_ctx_weights.npy".format(input_data_folder))
    # plot_weights_over_time(rec_ctx_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/Rec_w_ctx".format(op_plot_folder),
    #                     cmaps='gray_r')

    # mtl_op_weights_all = np.load("{}/mtl_op_weights.npy".format(input_data_folder))
    # plot_weights_over_time(mtl_op_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/mtl_op_w".format(op_plot_folder),
    #                     cmaps='gray_r')
    # ctx_op_weights_all = np.load("{}/ctx_op_weights.npy".format(input_data_folder))
    # plot_weights_over_time(ctx_op_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/ctx_op_w".format(op_plot_folder),
    #                     cmaps='gray_r')
    # mtl_ctx_weights_all = np.load("{}/mtl_ctx_weights.npy".format(input_data_folder))
    # plot_weights_over_time(mtl_ctx_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/mtl_ctx_w".format(op_plot_folder),
    #                     cmaps='gray_r')
    # ctx_mtl_weights_all = np.load("{}/ctx_mtl_weights.npy".format(input_data_folder))
    # plot_weights_over_time(ctx_mtl_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/ctx_mtl_w".format(op_plot_folder),
    #                     cmaps='gray_r')



def PlotAll3R(input_data_folder="../data/3R_CNT_fast_dCNT_fast_drift_wo_IP_lowIrift_with_limited7_IP_lowI", op_plot_folder="../plots/CNT_fast_drift_wo_IP_lowI"):
    # input_data_folder = "./data/CNT_fast_drift_wo_IP_lowI"
    # op_plot_folder = "./plots/CNT_fast_drift_wo_IP_lowI"
    # Path to your JSON file
    json_path = "{}/all_params.json".format(input_data_folder)
    dop = 2
    with open(json_path, "r") as f:
        sim_params = json.load(f)

    # breakpoint()
    sim_params = sim_params["simulation_params"]
    # Explicitly assign variables
    seqA = sim_params["t_series"]
    # E_fl_acc = sim_params["E_fl_acc"]
    threshold = 2#sim_params["threshold"]
    ID = sim_params["ID"]
    N_off_days = sim_params["N_off_days"]
    off_days = sim_params["off_days"]
    IR = sim_params["IR"]
    total_time = sim_params["total_time"]
    
    import numpy as np

    FR_HPC_history_all = np.load(f"{input_data_folder}/FR_HPC_history_all.npy", allow_pickle=True)
    FR_RSC_history_all = np.load(f"{input_data_folder}/FR_RSC_history_all.npy", allow_pickle=True)
    FR_ACC_history_all = np.load(f"{input_data_folder}/FR_ACC_history_all.npy", allow_pickle=True)
    FR_op_history_all = np.load(f"{input_data_folder}/FR_op_history_all.npy", allow_pickle=True)

    EX_HPC_history_all = np.load(f"{input_data_folder}/EX_HPC_history_all.npy", allow_pickle=True)
    EX_RSC_history_all = np.load(f"{input_data_folder}/EX_RSC_history_all.npy", allow_pickle=True)
    EX_ACC_history_all = np.load(f"{input_data_folder}/EX_ACC_history_all.npy", allow_pickle=True)

    last_activity_HPC_all = np.load(f"{input_data_folder}/last_activity_HPC_all.npy", allow_pickle=True)
    last_activity_RSC_all = np.load(f"{input_data_folder}/last_activity_RSC_all.npy", allow_pickle=True)
    last_activity_ACC_all = np.load(f"{input_data_folder}/last_activity_ACC_all.npy", allow_pickle=True)

    input_history_all = np.load(f"{input_data_folder}/input_history_all.npy", allow_pickle=True)
    rec_HPC_weights_all = np.load(f"{input_data_folder}/rec_HPC_weights_all.npy", allow_pickle=True)
    rec_RSC_weights_all = np.load(f"{input_data_folder}/rec_RSC_weights_all.npy", allow_pickle=True)
    rec_ACC_weights_all = np.load(f"{input_data_folder}/rec_ACC_weights_all.npy", allow_pickle=True)
    HPC_RSC_weights_all = np.load(f"{input_data_folder}/HPC_RSC_weights_all.npy", allow_pickle=True)
    RSC_ACC_weights_all = np.load(f"{input_data_folder}/RSC_ACC_weights_all.npy", allow_pickle=True)
    HPC_ACC_weights_all = np.load(f"{input_data_folder}/HPC_ACC_weights_all.npy", allow_pickle=True)

    # HPC_OP_weights_all = np.load(f"{input_data_folder}/HPC_OP_weights.npy", allow_pickle=True)
    ACC_OP_weights_all = np.load(f"{input_data_folder}/ACC_OP_weights_all.npy", allow_pickle=True)
    ACC_HPC_weights_all = np.load(f"{input_data_folder}/ACC_HPC_weights_all.npy", allow_pickle=True)
        
    
    FR_history_th_hpc = (FR_HPC_history_all > threshold).astype(float)*FR_HPC_history_all
    FR_history_th_acc = (FR_ACC_history_all > threshold).astype(float)*FR_ACC_history_all
    FR_history_th_rsc = (FR_RSC_history_all > threshold).astype(float)*FR_RSC_history_all
    
    last_activity_HPC_all_th = (last_activity_HPC_all > threshold).astype(float)*last_activity_HPC_all
    last_activity_RSC_all_th = (last_activity_RSC_all > threshold).astype(float)*last_activity_RSC_all
    last_activity_ACC_all_th = (last_activity_ACC_all > threshold).astype(float)*last_activity_ACC_all
    
    plot_engram_size(last_activity_HPC_all, threshold=threshold, title = "Engram size (HPC)",fname="{}/engram_size_HPC".format(op_plot_folder))
    plot_engram_size(last_activity_RSC_all, threshold=threshold,title = "Engram size (RSC)", fname="{}/engram_size_ACC".format(op_plot_folder))
    plot_engram_size(last_activity_ACC_all, threshold=threshold,title = "Engram size (ACC)", fname="{}/engram_size_ACC".format(op_plot_folder))

    plot_corr_matrix(last_activity_HPC_all_th[0], fname="{}/corr_matrix_hpc".format(op_plot_folder))
    plot_corr_matrix(last_activity_RSC_all_th[0], fname="{}/corr_matrix_rsc".format(op_plot_folder))
    plot_corr_matrix(last_activity_ACC_all_th[0], fname="{}/corr_matrix_acc".format(op_plot_folder))
    
    timepoints = np.arange(0,total_time,1)*1
    plot_firing_rate(timepoints, FR_op_history_all[:, :, 0],lab = "Output neuron",
                    xlabel="Time (s)", ylabel="Firing Rate (Hz)", c="r",fname= "{}/OP_neuron_activity".format(op_plot_folder),threshold=8)
    # breakpoint()

    plot_activity_n_excitability_time([FR_history_th_hpc[-1].T,FR_history_th_rsc[-1].T,FR_history_th_acc[-1].T],
                        titles=['Neuronal Activity (HPC)',
                                'Neuronal Activity (RSC)',
                                'Neuronal Activity (ACC)'],
                        seqA=seqA,
                        fname="{}/Activity".format(op_plot_folder),
                        cmaps=['Oranges', 'Greens',"Blues"])


    plot_activity_n_excitability_time([EX_HPC_history_all[-1].T,EX_RSC_history_all[-1].T,EX_ACC_history_all[-1].T],
                        titles=['Neuronal Excitability (HPC)',
                                'Neuronal Excitability (RSC)',
                                'Neuronal Excitability (ACC)'],
                        seqA=seqA,
                        fname="{}/Excitability".format(op_plot_folder),
                        cmaps=['Greens', 'Greens','Greens'],
                        colorbar_label=None)
    # labs = ["FC"] t [f"Off {it1}" for i in range(N_off_days)]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="./plots/Reimagined/Rec_w",
    #                        cmap='gray_r')

    cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    xlabs = [f"{i}" for i in range(1, N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr", use_bar_plot=True)
    mean_corr_hpc, std_corr_hpc, per_sim_corr_hpc, idx_hpc = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_hpc".format(op_plot_folder),
        cmap = "Oranges",
        marker = "^"
    )
    mean_corr_rsc, std_corr_rsc, per_sim_corr_rsc, idx_rsc = plot_mean_std_corr_over_time(
        last_activity_RSC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_rsc".format(op_plot_folder),
        cmap = "Blues"
    )
    mean_corr_acc, std_corr_acc, per_sim_corr_acc, idx_acc = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=0,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=False,
        title="Cell population \n activity correlation",
        fname="{}/encoding_vs_others_mean_std_acc".format(op_plot_folder),
        cmap = "Greens"
    )
    mean_DR_hpc = np.mean(1 - mean_corr_hpc)
    mean_DR_rsc = np.mean(1 - mean_corr_rsc)
    mean_DR_acc = np.mean(1 - mean_corr_acc)
    # print("excitability boosts:", E_fl, E_fl_acc)
    print("Normalized drift rate (HPC):", mean_DR_hpc)
    print("Normalized drift rate (RSC):", mean_DR_rsc)
    print("Normalized drift rate (ACC):", mean_DR_acc)

    xlabs = [f"{i-dop}" for i in range(N_off_days)]
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_HPC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std_hpc".format(op_plot_folder),
        cmap = "Oranges",
        marker = "^"
    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=dop,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/off1_vs_others_mean_std_acc".format(op_plot_folder),
        cmap = "Greens"
    )

    xlabs = [f"{i}" for i in range(N_off_days)]
    Title = "Ensemble similarity"
    # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr", use_bar_plot=True)
    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_HPC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std_hpc".format(op_plot_folder),
        cmap = "Oranges",
        marker = "^"

    )

    mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
        last_activity_ACC_all ,                # shape: (sims, time, neurons)
        ref_time_idx=-1,         # Encoding
        xlabels=xlabs,         # must match number of non-ref times
        include_ref_bar=True,
        title="Cell population \n activity correlation",
        fname="{}/recall_vs_others_mean_std_acc".format(op_plot_folder),
        cmap = "Greens",
        

    )

    # cbars = ["fff5f0ff","fdcab5ff","fc8a6aff","f96044ff","e83429ff","c3161bff","980c13ff",]
    # xlabs = ["Off 1","Off 2","Off 3"]
    # Title = "Ensemble similarity"
    # # breakpoint()
    # # plot_row_correlations(last_activity[0,0],last_activity[0,1:], xlabs=xlabs,title=Title,fname="./plots/Reimagined/encoding_corr", use_bar_plot=True)
    # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    #     last_activity_all[:,:-1,:] ,                # shape: (sims, time, neurons)
    #     ref_time_idx=0,         # Encoding
    #     xlabels=xlabs,         # must match number of non-ref times
    #     include_ref_bar=False,
    #     title="Cell population \n activity correlation",
    #     fname="{}/encoding_vs_offline_mean_std".format(op_plot_folder),
    #     cmap = "Oranges",
    #     marker = "^"
    # )
    # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    #     last_activity_all_acc[:,:-1,:] ,                # shape: (sims, time, neurons)
    #     ref_time_idx=0,         # Encoding
    #     xlabels=xlabs,         # must match number of non-ref times
    #     include_ref_bar=False,
    #     title="Cell population \n activity correlation",
    #     fname="{}/encoding_vs_offline_mean_std_acc".format(op_plot_folder),
    #     cmap = "Greens"
    # )

    # xlabs = [f"Off {i+1}" for i in range(N_off_days)]
    # # Title = "Ensemble similarity"
    # # # plot_row_correlations(last_activity[0,-1],last_activity[0,:-1], xlabs=xlabs,title=Title,fname="./plots/Reimagined//Recall_corr", use_bar_plot=True)
    # # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    # #     last_activity_all[:,1:,:],                # shape: (sims, time, neurons)
    # #     ref_time_idx=-1,         # Encoding
    # #     xlabels=xlabs,         # must match number of non-ref times
    # #     include_ref_bar=False,
    # #     title="Cell population \n activity correlation",
    # #     fname="{}/recall_vs_offline_mean_std".format(op_plot_folder),
    # #     cmap = "Oranges",
    # #     marker = "^"

    # # )

    # # mean_corr, std_corr, per_sim_corr, idx = plot_mean_std_corr_over_time(
    # #     last_activity_all_acc[:,1:,:],                # shape: (sims, time, neurons)
    # #     ref_time_idx=-1,         # Encoding
    # #     xlabels=xlabs,         # must match number of non-ref times
    # #     include_ref_bar=False,
    # #     title="Cell population \n activity correlation",
    # #     fname="{}/recall_vs_offline_mean_std_acc".format(op_plot_folder),
    # #     cmap = "Greens"

    # # )

    # S, T, N = last_activity_all.shape

    # # Treat NaNs as "not active" (change if you prefer to ignore them)
    # active = np.where(np.isnan(last_activity_all), False, last_activity_all > threshold)  # (S, T, N)

    # first_active = active[:, 0, :]          # (S, N) first session
    # other_active = active[:, 1:, :]         # (S, T-1, N) all sessions after the first

    # # Intersection & union with the first session, per sim & session
    # intersection = np.logical_and(first_active[:, None, :], other_active).sum(axis=-1)   # (S, T-1)
    # union        = np.logical_or (first_active[:, None, :], other_active).sum(axis=-1)   # (S, T-1)

    # # 1) Raw overlap counts
    # overlap_counts = intersection                                                # (S, T-1)

    # # 2) Fraction of first-session actives recovered later (recall of first set)
    # first_counts = first_active.sum(axis=-1)[:, None]                            # (S, 1)
    # overlap_frac_first = np.divide(
    #     intersection, first_counts,
    #     out=np.zeros_like(intersection, dtype=float), where=first_counts > 0
    # )                                                                           # (S, T-1)

    # mean_frac_first = overlap_frac_first.mean(axis=0)             # (T-1,)
    # std_frac_first  = overlap_frac_first.std(axis=0, ddof=0)  
    # sem_frac_first  = std_frac_first/np.sqrt(S)           # (T-1,)

    # fig, ax = plt.subplots(figsize=(8, 4))
    # x = np.arange(1,T - 1)
    # font_size = 14
    # tick_fontsize = 12
                                            
    # ax.bar(x, mean_frac_first[:-1], yerr=sem_frac_first[:-1], capsize=5,  edgecolor='black', alpha=0.9)
    # # Cosmetics
    # ax.spines[["right", "top"]].set_visible(False)
    # ax.set_title("Cell overlap fraction with encoding", fontsize=font_size)
    # ax.set_xlabel("Session", fontsize=font_size)
    # ax.set_ylabel("Overlap fraction", fontsize=font_size)
    # ax.set_xticks(x, labels=xlabs)
    # ax.tick_params(labelsize=tick_fontsize)
    # ax.set_ylim(0, 0.5)
    # # ax.grid(True, axis='y', linestyle='--', alpha=0.35)
    # fig.tight_layout()
    # plt.savefig("{}/overlap_frac_first_mean_sem".format(op_plot_folder))
    # plt.show()


    # x1, y1, r1 = plot_first_activity_vs_active_sessions(
    #     last_activity_all, threshold=threshold,
    #     first_session_idx=0,
    #     mode="concat",
    #     include_ref_in_count=True,
    #     fname="{}/first_activity_vs_counts_concat".format(op_plot_folder)
    # )
    # print(r1)

    # counts_x, mean_y, sem_y, n = plot_sessions_count_vs_activity_sem(
    #     last_activity_all, threshold=threshold,
    #     first_session_idx=0,
    #     include_ref_in_count=True,
    #     sem_mode='pooled',
    #     fname="{}/first_act_vs_sessions_pooled".format(op_plot_folder)
    # )
    # labs = ["Encoding","Day 1","Day 2","Day 3","Day 4"]
    # plot_weights_over_time(rec_weights_all[0],
    #                        titles=  labs,
    #                        fname="{}/Rec_w".format(op_plot_folder),
    #                        cmaps='gray_r')

    # plot_weights_over_time(rec_acc_weights_all[0],
    #                        titles=  labs,
    #                        fname="{}/Rec_w_acc".format(op_plot_folder),
    #                        cmaps='gray_r')

    labs = [f"Day {i+1}" for i in off_days]
    plot_weights_over_time(rec_HPC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_hpc".format(op_plot_folder),
                        cmaps='gray_r')
    plot_weights_over_time(rec_RSC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_rsc".format(op_plot_folder),
                        cmaps='gray_r')
    plot_weights_over_time(rec_ACC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/Rec_w_acc".format(op_plot_folder),
                        cmaps='gray_r')

    # plot_weights_over_time(mtl_op_weights_all[-1,off_days],
    #                     titles=  labs,
    #                     fname="{}/mtl_op_w".format(op_plot_folder),
    #                     cmaps='gray_r')
    plot_weights_over_time(ACC_OP_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/acc_op_w".format(op_plot_folder),
                        cmaps='gray_r')
    
    plot_weights_over_time(HPC_RSC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/hpc_rsc_w".format(op_plot_folder),
                        cmaps='gray_r')
    plot_weights_over_time(RSC_ACC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/rsc_acc_w".format(op_plot_folder),
                        cmaps='gray_r')
    plot_weights_over_time(HPC_ACC_weights_all[-1,off_days],
                        titles=  labs,
                        fname="{}/hpc_acc_w".format(op_plot_folder),
                        cmaps='gray_r')


if __name__ == "__main__":
    PlotAll()
