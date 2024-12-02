import pandas as pd
import random as rd
from concurrent.futures import ProcessPoolExecutor, as_completed


class Tree:
    def __init__(self, data=None, level=0):
        self.left = None
        self.right = None
        self.data = data
        self.level = level


def split_node(node, size_threshold, axis):
    """Splits a tree node if it meets the size threshold, returning left and right nodes."""
    if len(node.data) > size_threshold:
        min_val = node.data[axis].min()
        max_val = node.data[axis].max()
        split_point = rd.uniform(min_val, max_val)

        left_node = Tree(data=node.data[node.data[axis] > split_point], level=node.level + 1)
        right_node = Tree(data=node.data[node.data[axis] <= split_point], level=node.level + 1)

        return left_node, right_node
    return None, None


def build_single_tree(log, size_threshold):
    """Builds a single tree and calculates depth contributions."""
    forest = [Tree(data=log)]
    leaf_nodes_data = []

    i = 0
    while i < len(forest):
        node = forest[i]
        axis = ['AnimationError', 'FrameTime'][i % 2]
        left, right = split_node(node, size_threshold, axis)

        if left and right:
            node.left = left
            node.right = right
            forest.extend([left, right])
        else:
            leaf_nodes_data.append(node.data.assign(lod=node.level))

        i += 1

    # Concatenate dataframes and calculate normalized depth contribution for each tree
    depth_data = pd.concat(leaf_nodes_data).sort_index()
    max_depth = forest[-1].level
    return depth_data['lod'] / max_depth


def parallel_build_trees(log, num_trees=10, size_factor=0.001):
    """Parallelizes the construction of multiple trees and aggregates depth contributions."""
    size_threshold = len(log) * size_factor
    depth_contributions = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(build_single_tree, log, size_threshold) for _ in range(num_trees)]
        for future in as_completed(futures):
            depth_contributions.append(future.result())

    # Calculate the final 'lod' value as the mean of depth contributions across all trees
    return pd.DataFrame(depth_contributions).mean()


if __name__ == '__main__':
    # Load data
    log = pd.read_csv('SampleDataLog.csv')
    log['AnimationError'] = abs(log['FrameTime'] - log['FrameTime'].shift(-1)).fillna(0)
    print("AnimationError Done")
    output = log.assign(lod=0)
    print("OutputCreation Done")
    # Build trees and calculate the lod values
    lod_values = parallel_build_trees(log)
    print("Forest Done")
    output['lod'] = lod_values  # < 0.4  # Assign final lod values based on threshold
    print("OutputAssigned Done")
    output.to_csv('forestOutput.csv', mode='a', index=False)
