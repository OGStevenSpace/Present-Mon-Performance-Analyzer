import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import matplotlib.patches as patch
from matplotlib.collections import PatchCollection
import json
import Main


# define an object that will be used by the legend
class MulticolorPatch(object):
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler(object):
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(plt.Rectangle([width / len(orig_handle.colors) * i - handlebox.xdescent,
                                          -handlebox.ydescent],
                                         width / len(orig_handle.colors),
                                         height,
                                         facecolor=c,
                                         edgecolor='none'))

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


def load_config():
    """Load configuration from the JSON file."""
    with open('config.json', "r") as file:
        return json.load(file)


def reshape(array):
    """Reshape the array by flattening nested dictionaries and setting up a new DataFrame structure."""
    reshaped_data = []

    for col in array:
        reshaped_column = [parse_dict(row) if isinstance(row, dict) else row for row in array[col]]
        reshaped_data.append(reshaped_column)

    reshaped_df = pd.DataFrame(reshaped_data).rename(columns=lambda x: reshaped_data[0][x]).drop(0).reset_index(
        drop=True)
    return reshaped_df


def parse_dict(dictionary):
    """Parse a dictionary into a DataFrame row."""
    return pd.DataFrame(dictionary.values(), index=dictionary.keys()).transpose()


def dist_plot(gs, fig, data_list, step=0.05, y_tick=2):
    rows = len(data_list.keys())
    cols = 2
    top = rows * 0.05 - 0.05
    axes = [fig.add_subplot(gs[0, j]) for j in range(cols)]

    x, y = [], []

    for i_k, k in enumerate(data_list.keys()):
        for i_d, dist in enumerate(data_list[k]):

            x = list(data_list[k][dist])
            pad = [top - 0.05 * i_k] * len(x)
            x = [sum(x) for x in zip(x, pad)]
            y = list(data_list[k][dist].keys())
            axes[i_d].fill_between(y, x, 0)
            axes[i_d].fill_between(y, pad, 0, color='white', ec='0.8', lw=0.2)
            axes[i_d].title.set_text(dist)

    # Calculate ticks and common limits
    ticks_x = list(range(0, len(x) // y_tick, y_tick))
    ticks_y = [step * n for n in range(rows)]
    data_keys = list(data_list.keys())

    for ax in axes:
        ax.vlines([-33.333, 33.333], ymin=-0.5, ymax=rows - 0.5, colors='red', linewidth=0.5)
        ax.vlines([-16.666, 16.666], ymin=-0.5, ymax=rows - 0.5, colors='green', linewidth=0.5)
    # Configure each axis
    y_axis_config(axes[0], ticks_y, data_keys, top, step, show_y_axis=True)
    y_axis_config(axes[1], ticks_y, data_keys, top, step, show_y_axis=False)
    x_axis_config(axes[0], 0, 67, 4)
    x_axis_config(axes[1], 0, 67, 4)


def color_map(colors_list, qtl_size, ws=True, stack=True):
    """Generate color map based on frame quantiles colors from config."""
    if ws:
        colors = [[255, 255, 255, 1]] + colors_list
    else:
        colors = colors_list
    color_df = pd.DataFrame(colors, columns=['R', 'G', 'B', 'A']).div([255, 255, 255, 1])
    if stack:
        color_range = np.linspace(0, len(colors) - 1, qtl_size // 2 + 1)
    else:
        color_range = np.linspace(0, len(colors) - 1, qtl_size + 1)

    result = []
    for val in color_range:
        frac, lower_idx = np.modf(val)
        upper_idx = min(lower_idx + 1, len(colors) - 1)
        blended_color = color_df.iloc[int(lower_idx)] * (1 - frac) + color_df.iloc[int(upper_idx)] * frac
        result.append(blended_color)

    if stack:
        result += list(reversed(result))[1:] if qtl_size % 2 == 0 else list(reversed(result))

    return result


def y_axis_config(axis, ticks_y, data_keys, top, step, show_y_axis=True):
    """Configures the properties of a given axis."""
    axis.get_yaxis().set_visible(show_y_axis)
    axis.set_ylim([0.0, top + 3 * step])
    axis.set_yticks(ticks=ticks_y, labels=reversed(data_keys) if show_y_axis else [])
    axis.tick_params(axis='both', which='major', labelsize=6)


def plot_bar(ax, y_position, data, color, offset=0.4):
    """Helper function to plot horizontal bars for FrameTime and DisplayedTime data."""
    v_color = ['royalblue', 'orange']
    ax.barh(y_position, data, color=color, left=data.cumsum() - data, height=0.4)
    ax.vlines(0, y_position - offset, y_position + offset, colors=v_color[y_position % 2], lw=4)
    ax.vlines(data[:5].sum(), y_position - offset + 0.15, y_position + offset - 0.15, colors='red')


def bar_dist_plot(gs, fig, data, color):
    """Generates a horizontal bar distribution plot for FrameTime and DisplayedTime."""
    ax = fig.add_subplot(gs[0, 2])
    dist_df = pd.concat([data[key].transpose() for key in data.keys()])
    key_list = list(data.keys())
    extended_keys = [key for key in key_list for _ in range(2)]

    dist_df = dist_df.diff(axis=1).fillna(dist_df).reset_index()
    dist_df['fileName'] = pd.Series(extended_keys)

    for i, file_name in enumerate(reversed(key_list)):
        frame_data = dist_df.query("fileName == @file_name and index == 'FrameTime'").iloc[0, 1:-1]
        display_data = dist_df.query("fileName == @file_name and index == 'DisplayedTime'").iloc[0, 1:-1]

        y_position = i * 2
        ax.hlines(y=y_position - 0.5, xmin=0, xmax=72, colors='0.8', lw=1)
        plot_bar(ax, y_position, frame_data, color)
        plot_bar(ax, y_position + 1, display_data, color)

    ax.get_yaxis().set_visible(False)
    ax.vlines([-33.333, 33.333], ymin=-0.5, ymax=len(key_list) * 2 - 0.5, colors='red', linewidth=0.5)
    ax.vlines([-16.666, 16.666], ymin=-0.5, ymax=len(key_list) * 2 - 0.5, colors='green', linewidth=0.5)
    ax.set_ylim([-0.5, len(extended_keys) + 3.5])
    x_axis_config(ax, 0, 72, 8)
    ax.set_title('Percentiles')

    return ax


def bar_perf(gs, fig, mean, lows, order):
    """Generate a performance bar chart showing only FrameTime and DisplayedTime."""
    ax = fig.add_subplot(gs)
    # Filter for 'FrameTime' and 'DisplayedTime' in the mean DataFrame
    mean_df = pd.concat([mean[key].transpose() for key in mean.keys()])
    lows_df = pd.concat([lows[key].transpose() for key in lows.keys()])

    ft_df = pd.concat(
        [
            lows_df.loc[lows_df.index.isin(['FrameTime'])],
            mean_df.loc[mean_df.index.isin(['FrameTime'])].rename(columns={0: 1})
        ],
        axis=1
    ).sort_index(axis=1, ascending=order).iloc[::-1]

    dt_df = -pd.concat(
        [
            lows_df.loc[lows_df.index.isin(['DisplayedTime'])],
            mean_df.loc[mean_df.index.isin(['DisplayedTime'])].rename(columns={0: 1})

        ],
        axis=1
    ).sort_index(axis=1, ascending=order).iloc[::-1]
    ft_colors = pd.DataFrame(
        [
            [0.75, 0.81, 1.00, 1.00],
            [0.35, 0.51, 1.00, 1.00],
            [0.00, 0.25, 1.00, 1.00]
        ]
    ).sort_index(axis=0, ascending=order)
    dt_colors = pd.DataFrame(
        [
            [1.00, 0.82, 0.50, 1.00],
            [1.00, 0.73, 0.25, 1.00],
            [1.00, 0.65, 0.00, 1.00]
        ]
    ).sort_index(axis=0, ascending=order)

    # Plot the bar chart for FrameTime
    y_positions = range(len(ft_df))
    for i, col in enumerate(ft_df):
        ax.barh(y_positions, ft_df[col].values.flatten(), color=ft_colors.iloc[i, :])
        ax.barh(y_positions, dt_df[col].values.flatten(), color=dt_colors.iloc[i, :])

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(not order)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(reversed(mean.keys()))
    ax.tick_params(axis='y', which='major', labelsize=6)

    return ax


def var_bar(gs, fig, var, color):

    rows = len(var.keys())
    cols = 2
    axes = [fig.add_subplot(gs[0, j]) for j in range(cols)]

    for i, v in enumerate(reversed(var)):
        for j, col in enumerate(v):
            bars = np.cumsum(v[col].values.flatten())
            axes[j].title.set_text(col)
            for c, bar in enumerate(reversed(bars)):
                axes[j].barh(i, bar, color=color[c])

    axes[0].set_yticks(range(rows))
    axes[0].set_yticklabels(reversed(list(var.keys())))
    axes[0].tick_params(axis='both', which='major', labelsize=6)
    for ax in axes[1:]:
        ax.get_yaxis().set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=6)
    return axes


def util_bar(gs, figure, mean, wait):
    """Generate a performance bar chart showing only FrameTime and DisplayedTime."""
    ax = figure.add_subplot(gs)
    # Filter for 'FrameTime' and 'DisplayedTime' in the mean DataFrame
    mean_df = pd.concat([mean[key].transpose() for key in mean.keys()])
    wait_df = pd.concat([wait[key].transpose() for key in wait.keys()])
    cpu_df = mean_df.loc[mean_df.index.isin(['CPUUtilization'])].iloc[::-1]
    gpu_df = -mean_df.loc[mean_df.index.isin(['GPUUtilization'])].iloc[::-1]
    cpu_wait = wait_df.loc[wait_df.index.isin(['CPUWait'])].iloc[::-1]
    gpu_wait = -wait_df.loc[wait_df.index.isin(['GPUWait'])].iloc[::-1]

    # Plot the bar chart for FrameTime
    y_pos = range(len(cpu_df))
    y_pos_add = [i - 0.25 for i in y_pos]
    for i, col in enumerate(cpu_df):
        ax.barh(y_pos, cpu_df[col].values.flatten())
        ax.barh(y_pos, gpu_df[col].values.flatten())
        ax.barh(y_pos, cpu_wait[col].values.flatten(), height=0.4)
        ax.barh(y_pos, gpu_wait[col].values.flatten(), height=0.4)

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(reversed(mean.keys()))
    ax.tick_params(axis='y', which='major', labelsize=6)

    return ax


def x_axis_config(ax, bot, top, step):
    ticks_x = np.arange(bot, top, step)
    ax.get_xaxis().set_visible(True)
    ax.set_xlim([bot, top])
    ax.set_xticks(ticks_x)
    ax.tick_params(axis='x', which='major', labelsize=6)


def main(array, config, reshape_data=True, sort=False, asc=True):
    """Main function to handle configuration and generate plots."""
    data = reshape(array) if reshape_data else array
    fig_dist = plt.figure(figsize=(10, 6))
    gs_dist = grid_spec.GridSpec(1, 3, figure=fig_dist)

    fig_perf = plt.figure(figsize=(10, 6))
    gs_perf = grid_spec.GridSpec(1, 3, figure=fig_perf)

    fig_var = plt.figure(figsize=(10, 6))
    gs_var = grid_spec.GridSpec(1, 2, figure=fig_var)

    c_mean = data.iloc[3, :] / data.iloc[2, :]
    data = data.transpose()
    data['c_mean'] = c_mean

    if sort:
        data = data.sort_values(by='c_mean', ascending=asc).transpose()
    else:
        data = data.transpose()

    dist_plot(gs_dist, fig_dist, data.iloc[14])
    perc_chart = bar_dist_plot(
        gs_dist,
        fig_dist,
        data.iloc[8],
        color_map(
            config["frame_quantiles_colors"],
            len(config["frame_quantiles"])
        )
    )

    perc_chart.legend()


    fps_chart = bar_perf(gs_perf[0, 0], fig_perf, 1000/data.iloc[4], 1000/data.iloc[7], False)
    fps_chart.set_title('FPS')
    fps_chart.vlines([-30, 30], ymin=-0.4, ymax=len(data.keys()) - 0.6, colors='red', linewidth=0.5)
    fps_chart.vlines([-60, 60], ymin=-0.4, ymax=len(data.keys()) - 0.6, colors='green', linewidth=0.5)
    x_axis_config(fps_chart, -120, 120, 30)

    ms_chart = bar_perf(gs_perf[0, 1], fig_perf, data.iloc[4], data.iloc[7], True)
    ms_chart.set_title('DisplayedTimes | FrameTimes')
    ms_chart.vlines([-33.333, 33.333], ymin=-0.4, ymax=len(data.keys())-0.6, colors='red', linewidth=0.5)
    ms_chart.vlines([-16.666, 16.666], ymin=-0.4, ymax=len(data.keys())-0.6, colors='green', linewidth=0.5)

    pa1 = patch.Patch(facecolor=(0.00, 0.25, 1.00, 1.00))
    pa2 = patch.Patch(facecolor=(1.00, 0.65, 0.00, 1.00))
    #
    pb1 = patch.Patch(facecolor=(0.35, 0.51, 1.00, 1.00))
    pb2 = patch.Patch(facecolor=(1.00, 0.73, 0.25, 1.00))
    #
    pc1 = patch.Patch(facecolor=(0.75, 0.81, 1.00, 1.00))
    pc2 = patch.Patch(facecolor=(1.00, 0.82, 0.50, 1.00))

    ms_chart.legend(
        handles=[pa1, pa2, pb1, pb2, pc1, pc2],
        labels=['', 'Avg    ', '', 'Low 5%    ', '', 'Low 1%'],
        ncol=6,
        handletextpad=0.5,
        handlelength=1.0,
        columnspacing=-0.5,
        fontsize=7,
        loc='lower center',
        bbox_to_anchor=(0, -0.12)
    )

    x_axis_config(ms_chart, -72, 72, 8)

    util_plot = util_bar(gs_perf[0, 2], fig_perf, data.iloc[4], data.iloc[5]/data.iloc[3]*100)
    util_plot.set_title('Util Average & Wait Time %')
    x_axis_config(util_plot, -100, 100, 20)

    pa1 = patch.Patch(facecolor='blue')
    pa2 = patch.Patch(facecolor='orange')
    #
    pb1 = patch.Patch(facecolor='green')
    pb2 = patch.Patch(facecolor='red')

    util_plot.legend(
        handles=[pa1, pa2, pb1, pb2],
        labels=['CPU%    ', 'GPU%    ', 'CPU Wait%    ', 'GPU Wait%'],
        ncol=4,
        handletextpad=0.5,
        handlelength=1.0,
        columnspacing=-0.5,
        fontsize=7,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.12)
    )

    var_bar(gs_var,
            fig_var,
            data.iloc[13],
            color_map(
                config["frame_delta_colors"],
                config["delta_bins"]["bins"],
                False,
                False)
            )

    fig_dist.subplots_adjust(left=0.2, right=0.99, top=0.90, bottom=0.10, wspace=0)
    fig_dist.canvas.manager.set_window_title('Distribution Summary')

    fig_perf.subplots_adjust(left=0.2, right=0.99, top=0.90, bottom=0.10, wspace=0)
    fig_perf.canvas.manager.set_window_title('Performance Summary')

    fig_var.subplots_adjust(left=0.2, right=0.99, top=0.90, bottom=0.10, wspace=0)
    fig_var.canvas.manager.set_window_title('Stability')

    plt.show()


if __name__ == '__main__':
    Main.main(False)
