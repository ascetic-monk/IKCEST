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
"""Dataset
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
import random
import matplotlib.pyplot as plt
sys.path.append("../")


def data_split(dataset, args):
    indices = np.arange(0, len(dataset))
    city_num = dataset.city_num
    if args.val_num <= 0:
        train_num = len(dataset) - 1
        train_indices = indices[:train_num]
        test_indices = indices[-1:]
        return Subset(dataset, train_indices), None, Subset(dataset, test_indices)
    else:
        train_num = len(dataset) - args.val_num - 1

        train_indices = indices[:train_num]
        valid_indices = indices[train_num:train_num + args.val_num]
        test_indices = indices[-1:]
        return Subset(dataset, train_indices, aug=True), \
               Subset(dataset, valid_indices), Subset(dataset, test_indices)


class BaseDataset(object):
    """BaseDataset"""

    def __init__(self):
        pass

    def __getitem__(self, idx):
        """getitem"""
        raise NotImplementedError

    def __len__(self):
        """len"""
        raise NotImplementedError


class Subset(BaseDataset):
    """
    Subset of a dataset at specified indices.
    """

    def __init__(self, dataset, indices, aug=False):
        self.dataset = dataset
        self.indices = indices
        self.aug = aug

    def __getitem__(self, idx):
        """getitem"""
        if self.aug:
            data = self.dataset[self.indices[idx]]
            # np.random()
            # data[:-1] = (1 + 0.2 * (np.random.random((data.shape[0]-1, data.shape[1], data.shape[2])) - 0.5)) * data[:-1]
            # if random.random() > 0.95:
            #     data[:-1] = data[:-1][::-1]
            return data
        else:
            return self.dataset[self.indices[idx]]

    def __len__(self):
        """len"""
        return len(self.indices)


class InfectDataset(BaseDataset):
    def __init__(self, args):
        self.args = args
        self.input_file = self.args.input_file
        self.label_file = self.args.label_file
        self.region_names_file = self.args.region_names_file

        self.city_num = self.args.city_num
        self.feat_dim = self.args.feat_dim
        self.n_pred = self.args.n_pred
        self.n_his = self.args.n_his
        self.histogram = None

        self.data = self.process()

    def process(self):
        X = pd.read_csv(self.input_file)
        X = X.fillna(0.0)
        Y = pd.read_csv(self.label_file)

        with open(self.region_names_file, 'r') as f:
            for line in f:
                region_names = line.strip().split()

        # scaling
        SCALE = 100
        for name in region_names:
            X[name] = X[[name]].apply(lambda x: x / SCALE)
            Y[name] = Y[[name]].apply(lambda x: x / SCALE)

        print("region migration: ", X.head())
        print("infect: ", Y.head())

        X = X.drop(columns=['date'])
        Y = Y.drop(columns=['date'])

        date_num = len(Y)
        train_num = date_num - self.n_pred

        df = pd.DataFrame(columns=X.columns)
        cnt = 0
        # (?, n_his, city_num, node_feat_dim)
        for i in range(date_num - self.n_his - self.n_pred + 1):
            df = df.append(Y[i:(i + self.n_his)])

            # df = df.append([np.exp(i / 45)])
            df = df.append(Y[(i + self.n_his):(i + self.n_his + self.n_pred)])

        # plt.savefig('hist.jpg')
        # plt.show()
        # plt.plot(np.arange(0,45,1), np.sum(Y.values,axis=1))
        # plt.savefig('plot.jpg')
        # plt.show()

        # for testing
        df = df.append(Y[-self.n_his:])
        # df = df.append([np.exp((date_num - self.n_his) / 45)])
        df = df.append(Y[-self.n_pred:])  # unused, for padding

        data = df.values.reshape(-1, self.n_his + self.n_pred, self.city_num, 1)
        # data = np.reshape(data.transpose(0, 2, 1, 3), (-1, self.n_his + self.n_pred, self.feat_dim))
        tv_num = data.shape[0] - self.city_num

        pos_enc = np.exp(0.5 * np.arange(0, date_num - self.n_his - self.n_pred + 2)/30)
        pos_enc = np.tile(pos_enc, (self.city_num, 1, 1, 1)).transpose(3, 1, 0, 2)
        # pos_enc = np.reshape(np.tile(pos_enc, (self.city_num, 1)).transpose(1, 0),
        #                      (-1, 1, 1))
        data = np.concatenate([data, pos_enc], axis=1)
        data_stat = data[:, :, :, :]
        data_stat = np.reshape(data_stat.transpose(0, 2, 1, 3), (-1, self.n_his+self.n_pred+1, self.feat_dim))
        data_eli, data_weight = self.elimination(data_stat)
        data[:, -1:, :, :] = np.reshape(data_weight, (-1, self.city_num, 1, self.feat_dim)).transpose(0, 2, 1, 3)
        # weight_0 = data_stat.shape[0] - data_eli.shape[0]
        # data = np.append(data_eli, data_stat, axis=0)
        if self.args.ylog:
            data[:, :-1, :, :] = np.log(100*data[:, :-1, :, :]+1)/6.-0.4
            data_stat = np.log(100*data_eli+1)/6.-0.4
            self.histogram = plt.hist(data_stat[:,-1,0], bins=8)
            # self.histogram[]
        else:
            raise NotImplementedError
            # self.histogram = plt.hist(np.reshape(np.log(data[:,-2,:]+1), (-1, )), bins=8)
        return data

    def elimination(self, data):
        data_eli = []
        for i in range(data.shape[0]):
            d = data[i, :-1, :]
            if np.sum(d[:-1, :]) != 0:
                data_eli.append(d)
            else:
                data[i, -1:, :] /= 10
        return np.array(data_eli), data[:, -1:, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return np.expand_dims(self.data[idx], 0)
        else:
            return self.data[idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=392)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=10)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=3)  # equal to num_layers
    parser.add_argument('--Kt', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--opt', type=str, default='ADAM')
    parser.add_argument('--inf_mode', type=str, default='sep')
    parser.add_argument('--ylog', type=bool, default=True)
    parser.add_argument('--region_names_file', type=str,
                        default='../dataset/data_processed/region_names.txt')
    parser.add_argument('--input_file', type=str,
                        default='../dataset/data_processed/region_migration.csv')
    parser.add_argument('--label_file', type=str,
                        default='../dataset/data_processed/infection.csv')
    parser.add_argument('--adj_mat_file', type=str,
                        default='../dataset/data_processed/adj_matrix.npy')
    parser.add_argument('--output_path', type=str, default='./outputs/')
    parser.add_argument('--val_num', type=int, default=3)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--train_all', action='store_true')
    args = parser.parse_args()

    dataset = InfectDataset(args)
    print("num examples: %s" % len(dataset))

    train, valid, test = data_split(dataset, args)
    print("Train examples: %s" % len(train))
    print("Test examples: %s" % len(test))

    if valid is not None:
        print("Valid examples: %s" % len(valid))

    #  for i in range(3):
    #      print(dataset[[1,2,3]].shape)
    #      time.sleep(5)
