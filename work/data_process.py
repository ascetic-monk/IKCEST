# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""data processing
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

def get_grid_dict(city_path, city_name):
    d = {}
    with open(os.path.join(city_path, 'grid_attr.csv'), 'r') as f:
        for line in f:
            items = line.strip().split(',')
            axis = ",".join(items[0:2])
            ID = items[2]
            d[axis] = "_".join([city_name, ID])

    # d = {'x,y': ID}
    return d

def coord2ID(data_path, city_name, output_path):
    city_path = os.path.join(data_path, "city_%s" % city_name)
    grid_dict = get_grid_dict(city_path, city_name)

    trans_filename = os.path.join(city_path, "transfer.csv")
    output_file = os.path.join(output_path, "%s_transfer.csv" % (city_name))
    with open(trans_filename, 'r') as f, open(output_file, 'w') as writer:
        for line in f:
            items = line.strip().split(',')
            start_axis = ",".join(items[1:3])
            end_axis = ",".join(items[3:5])
            index = items[5]
            try:
                start_ID = grid_dict[start_axis]
                end_ID = grid_dict[end_axis]
            except KeyError: # remove no ID axis
                continue

            writer.write("%s,%s,%s,%s\n" % (items[0], start_ID, end_ID, index))

def calc_index_in_one_day(data_path, city_name):
    trans_filename = os.path.join(data_path, "%s_transfer.csv" % (city_name))
    transfer = pd.read_csv(trans_filename, 
            header=None,
            names=['hour', 's_region', 'e_region', 'index'])
        
    df = transfer.groupby(['s_region', 'e_region'])['index'].sum().reset_index()
    df = df[['s_region', 'e_region', 'index']]
    #  df = df.T
    #  df_list.append(df)
    return df

def process_city_migration(data_path, city_name):
    filename = os.path.join(data_path, "city_%s" % city_name, "migration.csv")
    migration = pd.read_csv(filename, 
                            sep=',', 
                            header=None,
                            names=['date', 's_city', 'e_city', city_name])

    # only use moving in "city" data, ignore moving out data
    df = migration[migration.e_city == city_name]
    df = df[["date", city_name]]

    # calculate total move in data of "city"
    df = df.groupby('date')[city_name].sum().reset_index()
    return df

def migration_process(data_path, city_list, output_path):
    for city_name in city_list:
        coord2ID(data_path, city_name, output_path)
        transfer = calc_index_in_one_day(output_path, city_name)
        migration = process_city_migration(data_path, city_name)

        df_list = []
        for i in range(len(migration)):
            df = transfer.copy()
            date = migration.date[i]
            index = migration[city_name][i]
            df['index'] = df['index'] * index
            df['date'] = date
            df = df[['date', 's_region', 'e_region', 'index']]
            df_list.append(df)

        df = pd.concat(df_list, axis=0)

        df.to_csv(os.path.join(output_path, '%s_migration.csv' % city_name), 
                header=None,
                index=None,
                float_format = '%.4f')


def adj_matrix_process(data_path, city_list, region_nums, output_path):
    total_region_num = np.sum(region_nums)
    adj_matrix = np.zeros((total_region_num, total_region_num))

    offset = 0
    for i, city in enumerate(city_list):
        filename = os.path.join(output_path, "%s_migration.csv" % city)
        migration = pd.read_csv(filename, 
                                sep=',', 
                                header=None,
                                names=['date', 's_region', 'e_region', 'index'])

        matrix = np.zeros((region_nums[i], region_nums[i]))
        order = sorted(range(region_nums[i]), key=lambda x:str(x))
        for j, idx in enumerate(order):
            target_region = "%s_%d" % (city, idx)
            # only use moving in "city" data, ignore moving out data
            df = migration[migration['e_region'] == target_region]

            df = df.groupby('s_region')['index'].mean().reset_index()
            #  res = df['index'].values.reshape(-1)
            for k, o in enumerate(order):
                s_region_id = "%s_%d" % (city, o)
                try:
                    value = df[df['s_region'] == s_region_id]['index'].values[0]
                except:
                    value = 0.0
                if s_region_id == target_region:
                    value = 0.0
                matrix[j, k] = value

        # merge two adj_matrix
        adj_matrix[offset:(offset + region_nums[i]), offset:(offset + region_nums[i])] = matrix
        offset += region_nums[i]

    file_to_save = os.path.join(output_path, 'adj_matrix.npy')
    print("saving result to %s" % file_to_save)
    np.save(file_to_save, adj_matrix)

def infection_process(data_path, city_list, region_nums, output_path):
    res = []
    region_name_list = []
    for i, city in enumerate(city_list):
        filename = os.path.join(data_path, "city_%s" % city, "infection.csv")
        migration = pd.read_csv(filename, 
                                sep=',', 
                                header=None,
                                names=["city", "region", "date", "infect"])

        order = sorted(range(region_nums[i]), key=lambda x:str(x))
        for j, idx in enumerate(order):
            target_region = idx #str(idx)
            df = migration[migration['region'] == target_region].reset_index(drop=True)
            if i == 0 and j == 0:
                df = df[['date', 'infect']]
            else:
                df = df[['infect']]

            df = df.rename(columns={'infect': '%s_%d' % (city, idx)})
            region_name_list.append("%s_%d" % (city, idx))

            res.append(df)
    df = pd.concat(res, axis=1)

    file_to_save = os.path.join(output_path, "infection.csv")
    print("saving result to %s" % file_to_save)
    # format: [date, A, B, C, D, E]
    df.to_csv(file_to_save, index=False)

    region_name_file = os.path.join(output_path, "region_names.txt")
    with open(region_name_file, 'w') as f:
        names = ' '.join(region_name_list)
        f.write(names + '\n')

def region_migration_process(data_path, city_list, region_nums, output_path):
    res = []
    #  import ipdb; ipdb.set_trace()
    for i, city in enumerate(city_list):
        filename = os.path.join(output_path, "%s_migration.csv" % city)
        migration = pd.read_csv(filename, 
                                sep=',', 
                                header=None,
                                names=['date', 's_region', 'e_region', 'index'])

        order = sorted(range(region_nums[i]), key=lambda x:str(x))
        for j, idx in enumerate(order):
            target_region = "%s_%d" % (city, idx)
            df = migration[migration['e_region'] == target_region]

            df = df.groupby('date')['index'].sum().reset_index()

            if i == 0 and j == 0:
                df = df[['date', 'index']]
            else:
                df = df[['index']]

            df = df.rename(columns={'index': target_region})

            res.append(df)

    df = pd.concat(res, axis=1)

    file_to_save = os.path.join(output_path, "region_migration.csv")
    print("saving result to %s" % file_to_save)
    # format: [date, A, B, C, D, E]
    df.to_csv(file_to_save, index=False, float_format = '%.2f')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./dataset/train_data')
    parser.add_argument('--output_path', type=str, default='./dataset/data_processed')
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    city_list = ["A", "B", "C", "D", "E"]
    region_nums = [118, 30, 135, 75, 34]

    print("migration process")

    migration_process(args.data_path, city_list, args.output_path)
    adj_matrix_process(args.data_path, city_list, region_nums, args.output_path)
    infection_process(args.data_path, city_list, region_nums, args.output_path)
    region_migration_process(args.data_path, city_list, region_nums, args.output_path)

