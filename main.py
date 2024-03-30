import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_longest_path(df):
    df_sorted = df.sort_values(by=['X', 'Y', 'W']).reset_index(drop=True)
    points = df_sorted.values.tolist()
    n = len(points)
    dp = [1] * n
    prev = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if points[i][1] > points[j][1] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j

    max_index = dp.index(max(dp))
    lis_path = []
    while max_index != -1:
        lis_path.append(points[max_index])
        max_index = prev[max_index]

    lis_path.reverse()
    return lis_path


class SegmentTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (4 * size)

    def update(self, node, node_left, node_right, index, value):
        if node_left == node_right:
            self.tree[node] = value
        else:
            mid = (node_left + node_right) // 2
            if index <= mid:
                self.update(2 * node, node_left, mid, index, value)
            else:
                self.update(2 * node + 1, mid + 1, node_right, index, value)
            self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, node, node_left, node_right, query_left, query_right):
        if query_left > node_right or query_right < node_left:
            return 0
        if query_left <= node_left and node_right <= query_right:
            return self.tree[node]
        mid = (node_left + node_right) // 2
        return max(self.query(2 * node, node_left, mid, query_left, query_right),
                   self.query(2 * node + 1, mid + 1, node_right, query_left, query_right))


def preprocess(df, lis_path):
    y_ranks = {y: i for i, y in enumerate(sorted(set(df['Y'])))}
    lis_set = set(tuple(point) for point in lis_path)
    return df, y_ranks, lis_set


def update_weights_with_segment_tree(df, y_ranks, lis_set):
    tree = SegmentTree(len(y_ranks))
    for index, row in df.iterrows():
        if tuple(row[['X', 'Y', 'W']]) not in lis_set:
            y_rank = y_ranks[row['Y']]
            max_sum_before = tree.query(1, 0, len(y_ranks) - 1, 0, y_rank - 1)
            tree.update(1, 0, len(y_ranks) - 1, y_rank, max_sum_before + row['W'])
            df.at[index, 'W'] = max_sum_before + 1  # Update weight to 1 more than max sum before


def increase_weights_non_lis(df, lis_path):
    df, y_ranks, lis_set = preprocess(df, lis_path)
    update_weights_with_segment_tree(df, y_ranks, lis_set)
    return df


file_path = ('./data.xlsx')
df_loaded = pd.read_excel(file_path)

longest_path = generate_longest_path(df_loaded)

# Assuming df is your DataFrame and lis_path is the list of points in the LIS
new_df = increase_weights_non_lis(df_loaded, longest_path)

new_df.rename(columns={'W': 'Part A Results'}, inplace=True)
new_df['Part B Results'] = new_df['Part A Results']

new_df.to_excel(file_path, index=False)
