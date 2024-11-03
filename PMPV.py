import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec


def from_data(array):

    output = reshape(array)
    dist_plot(output.iloc[14])

    return 0


def flatten(array):
    print(array)
    for i in array:
        print(i)
    return 0


def reshape(array):
    out_list = []
    for col in array:
        temp_list = []
        for r_idx, row in enumerate(array[col]):
            if isinstance(row, dict):
                parsed_data = parse_array(row)
                temp_list.insert(r_idx, parsed_data)
                # temp_list.append(parsed_data)
            else:
                temp_list.insert(r_idx, row)
        out_list.append(temp_list)

    output = pd.DataFrame(out_list)

    output = output.rename(columns=output.loc[0, :]).drop(0).reset_index(drop=True)
    return output


def parse_array(value):

    df = pd.DataFrame(value.values(), index=list(value.keys())).transpose()
    return df


def dist_plot(data_list, step=0.05):
    rows = len(data_list.keys())
    cols = 2
    fig, axis = plt.subplots(1, cols)
    top = rows * 0.05 - 0.05

    for i_k, k in enumerate(data_list.keys()):
        for i_d, dist in enumerate(data_list[k]):

            flag_vis_y = i_d == 0
            x = list(data_list[k][dist])
            pad = [top - 0.05 * i_k] * len(x)
            x = [sum(x) for x in zip(x, pad)]
            y = list(data_list[k][dist].keys())
            axis[i_d].fill_between(y, x, 0)
            axis[i_d].fill_between(y, pad, 0, color='white', ec='0.8')
            axis[i_d].get_xaxis().set_visible(True)
            axis[i_d].get_yaxis().set_visible(flag_vis_y)
            axis[i_d].set_xlim([0.0, 67])
            axis[i_d].set_ylim([0.0, top + 3 * step])

    fig.tight_layout(pad=-0.5)
    plt.show()
    return 0
