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
"""MAIN FILE
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import pandas as pd

import paddle.fluid as fluid
import paddle.fluid.layers as fl
import pgl
from pgl.utils.logger import log

from Dataset import InfectDataset, data_split
from dataloader import DataLoader
from graph import GraphFactory
from model import Model

def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = InfectDataset(args)
    log.info("num examples: %s" % len(dataset))

    train_dataset, valid_dataset, test_dataset = data_split(dataset, args)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=-1, shuffle=False)

    log.info("Train examples: %s" % len(train_dataset))
    log.info("Test examples: %s" % len(test_dataset))

    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset, batch_size=-1, shuffle=False)
        log.info("Valid examples: %s" % len(valid_dataset))
    else:
        valid_loader = None

    gf = GraphFactory(args)

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()
    train_program = fluid.Program()
    startup_program = fluid.Program()
    train_program.random_seed = args.seed
    startup_program.random_seed = args.seed

    with fluid.program_guard(train_program, startup_program):
        gw = pgl.graph_wrapper.GraphWrapper(
            "gw",
            node_feat=[('norm', [None, 1], "float32")],
            edge_feat=[('weights', [None, 1], "float32")])

        model = Model(args, gw)
        model.forward()

    infer_program = train_program.clone(for_test=True)
    infer_program.random_seed = args.seed

    with fluid.program_guard(train_program, startup_program):
        if args.opt == 'RMSProp':
            train_op = fluid.optimizer.RMSPropOptimizer(args.lr).minimize(
                model.loss)
        elif args.opt == 'ADAM':
            train_op = fluid.optimizer.Adam(args.lr).minimize(model.loss)

    exe = fluid.Executor(place)
    exe.run(startup_program)

    global_step = 0
    best = 100
    print_loss = []
    for epoch in range(1, args.epochs + 1):
        for idx, x_batch in enumerate(train_loader):
            global_step += 1
            x = np.array(x_batch[:, 0:args.n_his, :, :], dtype=np.float32)
            graph = gf.build_graph(x)
            feed = gw.to_feed(graph)
            feed['input'] = np.array(
                    x_batch[:, 0:args.n_his + 1, :, :], dtype=np.float32)
            b_loss = exe.run(train_program,
                                   feed=feed,
                                   fetch_list=[model.loss])

            print_loss.append(b_loss[0])

            if global_step % 5 == 0:
                log.info("epoch %d | step %d | loss %.6f" %
                         (epoch, global_step, np.mean(print_loss)))
                print_loss = []

            if global_step % 10 == 0 and valid_loader is not None:
                predicts = inference(exe, infer_program, 
                        model, valid_loader, gf, gw, args, future_days=args.n_pred)

                result = evaluate(valid_loader, predicts)
                message = "valid result: "
                for key, value in result.items():
                    message += "| %s %s " % (key, value)
                log.info(message)

                if result['rmsle'] < best:
                    predicts = inference(exe, infer_program, 
                            model, test_loader, gf, gw, args, future_days=30)
                    save_to_submit(predicts, args)
                    best = result['rmsle']

    log.info("best valid result: %s" % best)


def save_to_submit(predicts, args):
    # (n_pred, 1, city_num, 1) --> (n_pred, city_num)
    predicts = np.squeeze(predicts)
    predicts = predicts * 1000
    predicts = predicts.transpose(1,0).reshape(-1,).astype("int64")
    #  log.info(predicts)
    predicts = pd.DataFrame({"ret": predicts})

    submit = pd.read_csv(args.submit_file, 
                          header=None,
                          names=["cityid", "regionid", "date", "cnt"])

    submit = pd.concat([submit, predicts], axis=1)
    submit = submit.drop(columns=['cnt'])

    submit.to_csv(os.path.join(args.output_path, 'submission.csv'), 
            index=False, header=False)

def evaluate(dataloader, y_pred):
    # log.info(["y_pred", np.squeeze(y_pred)])
    result = {}
    y_true_list = []
    for x_batch in dataloader:
        y_true_list.append(x_batch[:, args.n_his:(args.n_his + args.n_pred), :, :])

    # y_true: (n_pred, len, city_num, 1)
    y_true = np.concatenate(y_true_list, axis=0).transpose(1, 0, 2, 3)
    # log.info(["y_true:", np.squeeze(y_true)])
    #  log.info(y_true)

    diff = np.log(y_pred + 1) - np.log(y_true + 1)
    rmsle = np.sqrt(np.mean(diff**2))
    result['rmsle'] = rmsle
    return result

def inference(exe, program, model, dataloader, gf, gw, args, future_days=30):
    pred_list = []
    for x_batch in dataloader:
        test_seq = np.copy(x_batch[:, 0:args.n_his + 1, :, :]).astype(np.float32)
        step_list = []
        for j in range(future_days):
            graph = gf.build_graph(test_seq[:, 0:args.n_his, :, :])
            feed = gw.to_feed(graph)
            feed["input"] = test_seq
            pred = exe.run(program, feed=feed, fetch_list=[model.pred])
            if isinstance(pred, list):
                pred = np.array(pred[0])

            pred[pred < 0] = 0.0
            test_seq[:, 0:args.n_his - 1, :, :] = test_seq[:, 1:args.n_his, :, :]
            test_seq[:, args.n_his - 1, :, :] = pred

            step_list.append(pred)

        pred_list.append(step_list)

    # pred_array: (n_pred, len(dataloader), city_num, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=392)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=20)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--Ks', type=int, default=1)  #equal to num_layers
    parser.add_argument('--Kt', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--keep_prob', type=float, default=1.0)
    parser.add_argument('--opt', type=str, default='ADAM')
    parser.add_argument('--region_names_file', type=str, 
            default='../dataset/data_processed/region_names.txt')
    parser.add_argument('--input_file', type=str, 
            default='../dataset/data_processed/region_migration.csv')
    parser.add_argument('--label_file', type=str, 
            default='../dataset/data_processed/infection.csv')
    parser.add_argument('--adj_mat_file', type=str, 
            default='../dataset/data_processed/adj_matrix.npy')
    parser.add_argument('--submit_file', type=str, 
            default='../dataset/train_data/submission.csv')
    parser.add_argument('--output_path', type=str, default='../outputs/')
    parser.add_argument('--val_num', type=int, default=3)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_cuda', action='store_true')
    args = parser.parse_args()

    blocks = [[1, 32, 64], [64, 32, 128]]
    args.blocks = blocks
    log.info(args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)

