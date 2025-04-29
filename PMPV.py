import pandas as pd
import numpy as np
from numpy import pi
from numpy import sin
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.patches as patch
from matplotlib.colors import LinearSegmentedColormap
import json

import Main


class Figure:
    def __init__(self, title=None, x=16, y=9, nrows=1, ncols=1, height_ratios=None, width_ratios=None,
                 left=0.10, right=0.98, top=0.90, bottom=0.10, wspace=0.05, hspace=0.1):
        self.fig = (plt.figure(figsize=(16, 9)))
        self.fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=wspace, hspace=hspace)
        self.fig.canvas.manager.set_window_title(title=title)
        self.gs = grid_spec.GridSpec(nrows=nrows, ncols=ncols, figure=self.fig,
                                     height_ratios=height_ratios, width_ratios=width_ratios)
        self.axis = [self.fig.add_subplot(self.gs[r, c]) for r in range(nrows) for c in range(ncols)]

    def set_axis_x(self, icol, ticks=None, label_size=0, show_axis=True, top=None):
        if ticks is None:
            ticks = [0, 1]
        (ax := self.axis[icol]).get_xaxis().set_visible(show_axis)
        ax.set_xlim([min(ticks), top if top else max(ticks)])
        ax.set_xticks(ticks)
        ax.tick_params(axis='x', labelsize=label_size)

    def set_axis_y(self, icol, ticks=None, data_keys=None, label_size=0, show_axis=True, bot_offset=0, top_offset=0):
        if ticks is None:
            ticks = [0, 1]
        (ax := self.axis[icol]).get_yaxis().set_visible(show_axis)
        lower = min(ticks) - bot_offset
        upper = max(ticks) + top_offset
        if lower == upper:
            eps = np.finfo(float).eps  # Smallest positive float delta
            upper += eps
        ax.set_ylim([lower, upper])
        ax.set_yticks(ticks, data_keys if show_axis else [])
        ax.tick_params(axis='y', labelsize=label_size)

    def set_axis_title(self, icol, title):
        self.axis[icol].set_title(title)

    def set_vlines(self, icol, *lines):
        ax = self.axis[icol]
        if not lines:
            print("set_vlines warning: Missing 'lines' arg.")
        else:
            for i, line in enumerate(lines):
                try:
                    color, x_line, linestyle = line
                    ax.vlines(x_line, ymin=-0.5, ymax=len(ax.get_yticks()) - 0.5,
                              colors=color, linewidth=0.5, linestyle=linestyle)
                except ValueError as e:
                    print(f"{e}: The 'lines' arg tuple [i={i}] does not have 3 correct components: "
                          f"(color[str], x[num], linestyle[str]).")

    def set_legend(self, icol, handles, labels, bbox_to_anchor):
        self.axis[icol].legend(
            handles=handles,
            labels=labels,
            ncol=len(labels),
            handletextpad=0.5,
            handlelength=1.0,
            columnspacing=-0.5,
            fontsize=7,
            loc='lower center',
            bbox_to_anchor=bbox_to_anchor
        )


def load_json(file_path) -> (dict, str):
    """Load data from the JSON file"""
    try:
        with open(file_path) as file:
            return json.load(file), None
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        return None, str(e)


def sort_df(dictionary, asc=True, dvd="tTime", dvs="record"):
    """
    Sorts a DataFrame created from a dictionary based on a ratio of two columns.
    Handles exceptions to prevent crashes by validating inputs and handling errors.
    """
    try:
        # Attempt to create DataFrame from dictionary
        df = pd.DataFrame.from_dict(dictionary)
        if df.empty:
            print("Warning: Input dictionary results in an empty DataFrame.")
            return df

        # Check for required columns
        required_columns = [dvd, dvs]
        for col in required_columns:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame")

        # Handle non-numeric columns by converting to numeric (coerce errors to NaN)
        df[dvd], df[dvs] = pd.to_numeric(df[dvd], errors='coerce'), pd.to_numeric(df[dvs], errors='coerce')
        # Replace zeros in divisor to avoid division by zero
        df[dvs] = df[dvs].replace(0, np.nan)

        # Calculate ratio and drop rows with invalid values (NaN)
        df["ratio"] = df[dvd] / df[dvs]
        df.dropna(subset=["ratio"], inplace=True)

        # Sort by ratio
        df.sort_values(by="ratio", ascending=asc, inplace=True)
        return df

    except KeyError as e:
        print(f"Error: {e} Check column names.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()


def color_map(colors):
    return LinearSegmentedColormap.from_list(
        'cmap',
        list(zip(np.linspace(0, 1, len(colors)), colors)),
        N=256
    )


def ridgeline(fig, icol, data, step=0.0625, cmap=plt.get_cmap('viridis'),
              top=None, label_size=(None, None)):
    def vline_color_gradient(position):
        a = sin(2 * pi * position - 4.75)
        b = 1 + 2 * a
        return -(a * b / 4) + 0.75

    title, categories = data
    pad, x_min, x_max = 0, 0, 0
    color = cmap(np.linspace(0, 1, len(categories)))[::-1]

    for i, (cat_name, cat_data) in enumerate(categories.items()):
        x, y = (np.array(list(cat_data.keys()), dtype=float),
                np.array(list(cat_data.values()), dtype=float))

        x_min, x_max = min(x_min, x.min()), max(x_max, x.max())
        dvs = pd.to_numeric(top, errors='coerce') - x_max
        fig.axis[icol].fill_between(x, y + pad, pad, color=color[i], ec='0.5', lw=0.2, zorder=-i)
        fig.axis[icol].plot(x / dvs + x_max, y.cumsum() * step + pad, color='0.0', lw=0.5)

        vlines = np.linspace(i, i + 1, 5)
        gradient = np.array([vline_color_gradient(pos) for pos in vlines]).round(2).astype(str)
        pad += step
        fig.axis[icol].hlines(
            vlines * step, x_max, top,
            color=gradient, lw=0.25)

    fig.set_axis_x(
        icol=icol, ticks=np.append(np.arange(x_min, x_max, round(x_max / 17, 0)), x_max),
        label_size=label_size[0], top=top)
    fig.set_axis_y(icol=icol, ticks=[step * n for n, _ in enumerate(categories)], data_keys=categories.keys(),
                   label_size=label_size[1], show_axis=not icol, top_offset=2.5 * step)
    fig.axis[icol].text(0.005, 0.005, "Time[ms]", fontsize=label_size[0], transform=fig.axis[icol].transAxes)
    fig.set_vlines(icol, ('black', [x_max], '-'), ('red', [33.333], '--'), ('green', [16.666], '--'))
    fig.set_axis_title(icol=icol, title=title)


def stacked_barh(fig, icol, data, cmap,
                 label_size=None):
    title, categories = data
    color = color_map(cmap)
    min_key, max_key = (float(k) for k in pd.DataFrame(categories.to_dict()).index[[0, -1]])

    fig.set_axis_x(icol=icol, ticks=np.linspace(0, 100, 11), label_size=label_size[0])
    fig.set_axis_x(icol=icol + 2, ticks=np.linspace(min_key, max_key, 5), label_size=label_size[0])
    fig.axis[icol].text(1.005, -0.01, "%", fontsize=label_size[0], transform=fig.axis[icol].transAxes)
    fig.set_axis_y(icol=icol, ticks=list(range(len(categories))), data_keys=categories.keys(), label_size=label_size[1],
                   show_axis=not icol, bot_offset=0.5, top_offset=0.5)
    fig.set_axis_y(icol=icol + 2, ticks=[0.0], data_keys=["Time[ms]"], label_size=label_size[1],
                   show_axis=not icol, bot_offset=0.5, top_offset=0.5)

    fig.set_axis_title(icol=icol, title=title)
    for i in np.linspace(min_key, max_key, int(max_key+1)):
        fig.axis[icol+2].barh(y=0, width=1, left=i, color=color(i / max_key))
    for i, (cat_name, cat_data) in enumerate(categories.items()):
        left = 0
        vals, v_len = list(cat_data.values()), len(cat_data.values())
        for n, val in enumerate(vals):
            fig.axis[icol].barh(y=i, width=val*100, left=left*100, label=cat_name, color=color(n / v_len))
            left += val


def barh(fig, icol, i_bar, data, cmap,
         label_size=None):
    def perf_barh():
        fig.set_axis_x(icol=icol, ticks=np.linspace(-128, 128, 17, dtype=int), label_size=label_size)
        fig.axis[icol].text(0.005, 0.005, "Time[ms]", fontsize=label_size, transform=fig.axis[icol].transAxes)
        fig.set_vlines(icol, ('black', [0], '-'), ('red', [-33.333, 33.333], '--'), ('green', [-16.666, 16.666], '--'))
        fig.set_axis_title(icol=icol, title="Average Displayed and FrameTime & Lows")
        for i, (cat_name, cat_data) in enumerate(categories.items()):
            v_len = len(cat_data.values())
            for c, val in enumerate(cat_data.values()):
                fig.axis[icol].barh(i, (-1) ** i_bar * val, color=color(c / v_len))

    def util_barh():
        fig.set_axis_x(icol=icol, ticks=np.linspace(-100, 100, 21, dtype=int), label_size=label_size)
        fig.axis[icol].text(1.005, -0.01, "%", fontsize=label_size, transform=fig.axis[icol].transAxes)
        fig.set_vlines(icol, ('black', [0], '-'))
        fig.set_axis_title(icol=icol, title="Average GPU and CPU usage")
        for i, (cat_name, cat_data) in enumerate(categories.items()):
            if title == "CPUWait" or title == "GPUWait":
                height = 0.3
            else:
                height = 0.8
            fig.axis[icol].barh(i, (-1) ** i_bar * cat_data, color=color(i_bar), height=height)
            #fig.axis[icol].barh(i, (-1) ** i_bar * cat_data, color=color(i_bar))

    title, categories = data
    fig.set_axis_y(icol=icol, ticks=list(range(len(categories))), data_keys=categories.keys(), label_size=label_size,
                   show_axis=not icol, bot_offset=0.5, top_offset=0.5)
    color = color_map(cmap[title])

    if icol:
        util_barh()
    else:
        perf_barh()


def main(data):
    sorted_df = sort_df(data, asc=False)

    n_datasets = len(sorted_df)
    fig_height = int(max(6.0, n_datasets * 0.5))
    keys = ['tTime', 'dist', 'delta_dist', 'mean', 'lows', 'waitTime']
    time_df, dist_df, var_df, mean_df, low_df, wait_df = (
        pd.DataFrame.from_dict(sorted_df[key].to_dict(), orient='index')
        for key in keys
    )

    wait_df = wait_df.div(time_df[0], axis=0) * 100

    for col in ['FrameTime', 'DisplayedTime']:
        low_df[col] = [dict(sorted({**d, '1.00': mean_df.loc[idx, col]}.items(),
                                   key=lambda x: float(x[0])))
                       for idx, d in low_df[col].items()]

    dist_plot, var_plot, mean_plot = (Figure(title='Distribution Summary', y=fig_height, ncols=2),
                                      Figure(title='Stability', y=fig_height, ncols=2, nrows=2,
                                             height_ratios=[49, 1], bottom=0.04),
                                      Figure(title='Performance Summary', y=fig_height, ncols=2))

    cmap_dict = dict(
        FrameTime=[
            (0.75, 0.81, 1.00, 1.00),
            (0.35, 0.51, 1.00, 1.00),
            (0.00, 0.25, 1.00, 1.00)
        ],
        DisplayedTime=[
            (1.00, 0.82, 0.50, 1.00),
            (1.00, 0.73, 0.25, 1.00),
            (1.00, 0.65, 0.00, 1.00)
        ],
        CPUUtilization=[
            (0.00, 0.25, 1.00, 1.00),
            (0.00, 0.00, 0.00, 0.00),
            (0.00, 0.00, 0.00, 0.00)
        ],
        GPUUtilization=[
            (1.00, 0.65, 0.00, 1.00),
            (0.00, 0.00, 0.00, 0.00),
            (0.00, 0.00, 0.00, 0.00)
        ],
        CPUWait=[
            (1.00, 0.00, 0.00, 0.50),
            (0.00, 0.00, 0.00, 0.00),
            (0.00, 0.00, 0.00, 0.00)
        ],
        GPUWait=[
            (0.00, 1.00, 0.00, 0.50),
            (0.00, 0.00, 0.00, 0.00),
            (0.00, 0.00, 0.00, 0.00)
        ],
        Variability=[
            (0.44, 0.68, 0.28, 1.00),
            (0.60, 0.80, 0.20, 1.00),
            (1.00, 0.75, 0.00, 1.00),
            (0.93, 0.49, 0.19, 1.00),
            (0.75, 0.00, 0.00, 1.00)
        ]
    )

    mean_patch_list = []
    for col_pair in zip(cmap_dict['FrameTime'], cmap_dict['DisplayedTime']):
        mean_patch_list += [patch.Patch(facecolor=col_tup) for col_tup in col_pair]

    for icol, i_dist in enumerate(dist_df.items()):
        ridgeline(fig=dist_plot, icol=icol, data=i_dist, step=0.0625, label_size=(6, 6), top=75)

    for icol, i_var in enumerate(var_df.items()):
        stacked_barh(fig=var_plot, icol=icol, data=i_var, cmap=cmap_dict['Variability'], label_size=(6, 6))

    for i_bar, i_mean in enumerate(low_df.items()):

        barh(mean_plot, 0, i_bar, i_mean, cmap_dict, label_size=6)
        mean_plot.set_legend(
            icol=0, handles=reversed(mean_patch_list),
            labels=['', 'Avg    ', '', 'Low 5%    ', '', 'Low 1%'], bbox_to_anchor=(0.5, -0.1))

    pa1 = patch.Patch(facecolor='blue')
    pa2 = patch.Patch(facecolor='orange')
    pb1 = patch.Patch(facecolor='green')
    pb2 = patch.Patch(facecolor='red')

    for i_bar, i_mean in enumerate(pd.concat([mean_df, wait_df], axis=1)[
                                       [
                                           'CPUUtilization',
                                           'GPUUtilization',
                                           'CPUWait',
                                           'GPUWait'
                                       ]].items()):
        barh(mean_plot, 1, i_bar, i_mean, cmap_dict, label_size=6)
        mean_plot.set_legend(
            icol=1, handles=[pa1, pa2],
            labels=['CPU%    ', 'GPU%    '], bbox_to_anchor=(0.5, -0.1))

    plt.show()


if __name__ == '__main__':
    config, error = load_json('config.json')

    if error is not None:
        print(f'Config Load error: {error}\n'
              f'Closing Present Mon Performance Viewer')
        quit()

    # Output_27-02-25_T141345
    # Output_01-03-25_T183524
    paths, error = Main.open_files()
    perf_data, error = load_json(paths[0])
    if error is not None:
        print(f'Data Load error: {error}\n'
              f'Closing Present Mon Performance Viewer')
        quit()

    main(perf_data)