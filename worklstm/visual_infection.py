"""
    Visualize infection.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pprint


def infection_process(data_df, city_list, region_nums):
    res = []
    region_name_list = []
    for i, city in enumerate(city_list):
        order = sorted(range(region_nums[i]), key=lambda x: str(x))
        for j, idx in enumerate(order):
            target_region = idx  # str(idx)
            df = data_df[data_df['region'] == target_region].reset_index(drop=True)
            df = df[df['city'] == city].reset_index(drop=True)
            if i == 0 and j == 0:
                df = df[['date', 'infect']]
            else:
                df = df[['infect']]

            df = df.rename(columns={'infect': '%s_%d' % (city, idx)})
            region_name_list.append("%s_%d" % (city, idx))

            res.append(df)
            # print('City %s_%d: %s' % (city, idx, df.values.shape))
    fin_df = pd.concat(res, axis=1)
    return fin_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=903)  # 区域数目
    parser.add_argument('--region_names_file', type=str,  # 各区域名字，直接用该后处理文件夹内容
                        default='../../proj_baiduAI/dataset/data_processed_all/region_names.txt')
    parser.add_argument('--hist_file', type=str,
                        default='../../proj_baiduAI/dataset/data_processed_all/infection.csv')  # 历史感染人数，直接用该后处理文件内容
    parser.add_argument('--pred_file', type=str,
                        default='../outputs/submission.csv')  # 预测感染人数，直接用提交内容
    parser.add_argument('--output_path', type=str,  # 生成图片保存路径
                        default='./')
    parser.add_argument('--savefig_name', type=str,  # 生成图片名称
                        default='visual_infection_incr.png')

    args = parser.parse_args()

    hist_df = pd.read_csv(args.hist_file)
    pred_df = pd.read_csv(args.pred_file,
                          sep=',',
                          header=None,
                          names=['city', 'region', 'date', 'infect'])

    city_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    region_nums = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]

    pred_df = infection_process(pred_df, city_list, region_nums)

    # Mask Construction 构造禁用城市字典
    with open(args.region_names_file, 'r') as f:
        for line in f:
            region_names = line.strip().split()

    # Mask Defination. Set corresponding area 0 means it wouldn't display in the figure.
    valid_mask = np.ones_like(region_names)
    vaild_dict = dict(zip(region_names, valid_mask))
    # pprint.pprint(vaild_dict)

    # An example to ban a city 禁用一个城市的例子
    ban_list = ['A','C','D','E','F','G','H','I','J','K']
    for key in vaild_dict:
        if key[0] in ban_list:
            vaild_dict[key] = 0

    NUM_COLORS = np.sum(region_nums)
    color = cm.rainbow(np.linspace(0, 1, NUM_COLORS))

    date_hist = hist_df['date'].values.astype('str')  # Date of history
    hist_df = hist_df.drop(['date'], axis=1)

    date_pred = pred_df['date'].values.astype('str')  # Date of future
    pred_df = pred_df.drop(['date'], axis=1)

    plt.figure(figsize=(30, 22))
    for idx, key in enumerate(hist_df):
        infect = hist_df[key].values
        if vaild_dict[key]:
            plt.plot(date_hist, infect, label=key, c=color[idx])
    for idx, key in enumerate(pred_df):
        infect = pred_df[key].values
        if vaild_dict[key]:
            plt.plot(date_pred, infect, c=color[idx], linestyle='--')

    plt.title('Infection Increment')
    plt.xticks(rotation=45)
    plt.legend(ncol=20, framealpha=0.2, loc='upper center')

    fig_path = os.path.join(args.output_path, args.savefig_name)
    plt.savefig(fig_path)
    # plt.show()