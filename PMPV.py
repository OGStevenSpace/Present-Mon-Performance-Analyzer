import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec


def load_data(array, config, reshape_data=True):
    """Processes data array, reshaping if specified, and passes it to the main plotting function."""
    processed_data = reshape(array) if reshape_data else array
    main(processed_data, config)


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

    # Calculate ticks and common limits
    ticks_x = list(range(0, len(x) // y_tick, y_tick))
    ticks_y = [step * n for n in range(rows)]
    data_keys = list(data_list.keys())

    # Configure each axis
    configure_axis(axes[0], ticks_x, ticks_y, data_keys, top, step, show_y_axis=True)
    configure_axis(axes[1], ticks_x, ticks_y, data_keys, top, step, show_y_axis=False)

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
    axis.get_xaxis().set_visible(True)
    axis.get_yaxis().set_visible(show_y_axis)
    axis.set_xlim([0.0, 67])
    axis.set_ylim([0.0, top + 3 * step])
    axis.set_xticks(ticks_x)
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
        ax.hlines(y=y_position - 0.5, xmin=0, xmax=67, colors='0.8', lw=1)
        plot_bar(ax, y_position, frame_data, color)
        plot_bar(ax, y_position + 1, display_data, color)

    ax.get_yaxis().set_visible(False)
    ax.set_ylim([-0.5, len(extended_keys) + 3.5])
    ax.set_xlim([0, 67])
    ax.tick_params(axis='both', which='major', labelsize=6)


def main(data, config):
    """Main function to handle configuration and generate plots."""
    fig = plt.figure(figsize=(10, 6))
    fig.canvas.manager.set_window_title('Test')
    gs = grid_spec.GridSpec(1, 3, figure=fig)

    c_mean = data.iloc[3, :] / data.iloc[2, :]
    data = data.transpose()
    data['c_mean'] = c_mean
    data = data.sort_values(by='c_mean').transpose()

    dist_plot(gs, fig, data.iloc[14])
    bar_dist_plot(gs, fig, data.iloc[8], color_map(config))

    plt.show()
