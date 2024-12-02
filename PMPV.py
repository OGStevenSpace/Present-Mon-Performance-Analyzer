import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import Main


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
            axes[i_d].fill_between(y, pad, 0, color='white', ec='0.8')
            axes[i_d].title.set_text(dist)

    # Calculate ticks and common limits
    ticks_x = list(range(0, len(x) // y_tick, y_tick))
    ticks_y = [step * n for n in range(rows)]
    data_keys = list(data_list.keys())

    # Configure each axis
    configure_axis(axes[0], ticks_x, ticks_y, data_keys, top, step, show_y_axis=True)
    configure_axis(axes[1], ticks_x, ticks_y, data_keys, top, step, show_y_axis=False)
    x_axis_config(axes[0], 0, 68, 4)
    x_axis_config(axes[1], 0, 68, 4)

    fig.tight_layout(pad=-0.5)
    fig.subplots_adjust(left=0.1, right=0.99)


def color_map(config):
    """Generate color map based on frame quantiles colors from config."""
    colors = [[255, 255, 255, 1]] + config["frame_quantiles_colors"]
    color_df = pd.DataFrame(colors, columns=['R', 'G', 'B', 'A']).div([255, 255, 255, 1])
    qtl_size = len(config["frame_quantiles"])
    color_range = np.linspace(0, len(colors) - 1, qtl_size // 2 + 1)

    result = []
    for val in color_range:
        frac, lower_idx = np.modf(val)
        upper_idx = min(lower_idx + 1, len(colors) - 1)
        blended_color = color_df.iloc[int(lower_idx)] * (1 - frac) + color_df.iloc[int(upper_idx)] * frac
        result.append(blended_color)

    return result + list(reversed(result))[1:] if qtl_size % 2 == 0 else result + list(reversed(result))


def configure_axis(axis, ticks_x, ticks_y, data_keys, top, step, show_y_axis=True):
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


def bar_dist_plot(grid_spec, figure, data, color):
    """Generates a horizontal bar distribution plot for FrameTime and DisplayedTime."""
    ax = figure.add_subplot(grid_spec[0, 2])
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
    ax.set_ylim([-0.5, len(extended_keys) + 3.5])
    x_axis_config(ax, 0, 72, 8)
    ax.set_title('Percentiles')


def bar_perf(grid_spec, figure, mean, lows, order):
    """Generate a performance bar chart showing only FrameTime and DisplayedTime."""
    ax = figure.add_subplot(grid_spec)
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


def util_bar(grid_spec, figure, mean):
    """Generate a performance bar chart showing only FrameTime and DisplayedTime."""
    ax = figure.add_subplot(grid_spec)
    # Filter for 'FrameTime' and 'DisplayedTime' in the mean DataFrame
    mean_df = pd.concat([mean[key].transpose() for key in mean.keys()])

    cpu_df = mean_df.loc[mean_df.index.isin(['CPUUtilization'])].iloc[::-1]

    gpu_df = -mean_df.loc[mean_df.index.isin(['GPUUtilization'])].iloc[::-1]

    # Plot the bar chart for FrameTime
    y_positions = range(len(cpu_df))
    for i, col in enumerate(cpu_df):
        ax.barh(y_positions, cpu_df[col].values.flatten())
        ax.barh(y_positions, gpu_df[col].values.flatten())

    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    ax.set_yticks(y_positions)
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
    fig_dist.canvas.manager.set_window_title('Distribution Summary')
    gs_dist = grid_spec.GridSpec(1, 3, figure=fig_dist)

    fig_perf = plt.figure(figsize=(10, 6))
    fig_perf.canvas.manager.set_window_title('Performance Summary')
    gs_perf = grid_spec.GridSpec(1, 3, figure=fig_perf)

    c_mean = data.iloc[3, :] / data.iloc[2, :]
    data = data.transpose()
    data['c_mean'] = c_mean

    if sort:
        data = data.sort_values(by='c_mean', ascending=asc).transpose()
    else:
        data = data.transpose()

    dist_plot(gs_dist, fig_dist, data.iloc[14])
    bar_dist_plot(gs_dist, fig_dist, data.iloc[8], color_map(config))
    fig_dist.tight_layout(pad=-0.5)
    fps_chart = bar_perf(gs_perf[0, 0], fig_perf, 1000/data.iloc[4], 1000/data.iloc[7], False)
    fps_chart.set_title('FPS')
    x_axis_config(fps_chart, -270, 271, 30)
    ms_chart = bar_perf(gs_perf[0, 1], fig_perf, data.iloc[4], data.iloc[7], True)
    ms_chart.set_title('DisplayedTimes | FrameTimes')
    x_axis_config(ms_chart, -72, 73, 8)
    util_plot = util_bar(gs_perf[0, 2], fig_perf, data.iloc[4])
    util_plot.set_title('Util Average')
    x_axis_config(util_plot, -100, 101, 20)
    fig_perf.tight_layout(pad=-1)
    plt.show()
