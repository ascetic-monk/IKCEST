from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pandas as pd
import sys
import os
sys.path.append("../")
from worklstm.dataset_lstm import InfectDataset, data_split
from worklstm.LogWriter import LogWriter
# from work.dataloader import DataLoader
from torch.utils.data import DataLoader
import time


def dataload(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    dataset = InfectDataset(args)

    train_dataset, valid_dataset, test_dataset = data_split(dataset, args)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.city_num,
                             shuffle=False)

    if valid_dataset is not None:
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False)
    else:
        valid_loader = None
    return train_loader, valid_loader, test_loader


class Weighted_MSE(nn.Module):
    def __init__(self, size_average=True):
        super(Weighted_MSE, self).__init__()
        self.size_average = size_average

    def forward(self, pred, truth, weights):
        if self.size_average:
            return torch.mean(weights*(truth-pred)**2)
        else:
            return torch.sum(weights*(truth-pred)**2)


class WeightedGenerate( ):
    def __init__(self, hist, use_cuda, args):
        if args.ylog:
            self.threshold = hist[1]
        else:
            self.threshold = np.exp(hist[1])-1
        self.weightsth = hist[0]
        self.weightsth[0] /= 2
        self.weightsth[-1] /= 2
        self.weightsth = 1./self.weightsth
        self.weightsth = self.weightsth/np.sum(self.weightsth)
        self.use_cuda = use_cuda

    def generate(self, y):
        if self.use_cuda:
            weight = torch.zeros(y.size(0), y.size(1), y.size(2)).cuda()
        else:
            weight = torch.zeros(y.size(0), y.size(1), y.size(2))

        for idx in range(len(self.threshold)-1):
            weight += self.weightsth[idx] * \
                      (y >= self.threshold[idx]) * (y < self.threshold[idx+1])
        return weight

    def generate_mode(self, x, y):
        total_x = torch.mean(x, axis=1)
        total_y = torch.mean(y, axis=1)
        decrease = (total_x>total_y)
        decrease_to_down = decrease * (total_y<self.threshold[1])
        return (40-12)*decrease_to_down+(12-1)*decrease+1
        # return (40-5)*decrease_to_down+(5-1)*decrease+1


class Gate(nn.Module):
    def __init__(self, args):
        super(Gate, self).__init__()
        self.linear1 = nn.Linear(args.n_his+1, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.linear1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        x = self.sigmoid(self.linear3(x))
        return x


class Sequence(nn.Module):
    def __init__(self, use_cuda, in_feat=1, out_feat=1, hid_feat=64):
        super(Sequence, self).__init__()
        self.lstm1 = nn.LSTMCell(1, hid_feat)
        self.lstm2 = nn.LSTMCell(hid_feat, hid_feat)
        self.linear = nn.Linear(2*hid_feat+1, 1)
        self.use_cuda = use_cuda
        self.hid_feat = hid_feat

    def forward(self, input, future = 1):
        outputs = []
        if not self.use_cuda:
            h_t = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double)
            c_t = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double)
            h_t2 = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double)
            c_t2 = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double)
        else:
            h_t = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double).cuda()
            c_t = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double).cuda()
            h_t2 = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double).cuda()
            c_t2 = torch.zeros(input.size(0), self.hid_feat, dtype=torch.double).cuda()

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t[:,0,:], (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(torch.cat([input_t[:,0,:],h_t, h_t2],1))
            # outputs += [output]
        for i in range(future):# if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(torch.cat([output, h_t, h_t2],1))
            # outputs += [output]
        # outputs = torch.stack(outputs, 1)
        return output.unsqueeze(1)


class Gate_Sequence(nn.Module):
    def __init__(self, use_cuda, args):
        super(Gate_Sequence, self).__init__()
        self.sequence = Sequence(use_cuda)
        if args.gate:
            self.gate = Gate(args)
        self.args = args

    def forward(self, x):
        factor = x.squeeze()
        _input = x[:, :self.args.n_his, :]
        if self.args.gate:
            output = self.sequence(_input) * self.gate(factor).unsqueeze(2)
        else:
            output = self.sequence(_input)
        return output


def train(model, criterion, optimizer, lr_schedual, dataloader, args):
    best = 100
    train_loader, valid_loader, test_loader = dataloader['train'], dataloader['valid'], dataloader['test']
    weight_generation = WeightedGenerate(train_loader.dataset.dataset.histogram, args.use_cuda, args)
    for epo in range(args.epochs):
        model.train()
        loss_train = 0
        lr_schedual.step()
        for idx, data in enumerate(train_loader):
            if args.use_cuda:
                data = data.cuda()
            # y = model(data[:, :args.n_his, :])
            weights_date = data[:, -1:, :]
            factor = torch.cat([data[:, :args.n_his, :], data[:, -1:, :]], axis=1)
            y = model(factor)
            weights = weight_generation.generate(data[:, args.n_his:args.n_his+args.n_pred, :])
            weights_pat = weight_generation.generate_mode(data[:, :args.n_his, :],
                                                 data[:, args.n_his:args.n_his+args.n_pred, :]).unsqueeze(2)

            loss = criterion(y, data[:, args.n_his:args.n_his+args.n_pred, :],
                             weights*weights_date*weights_pat)
            # loss = criterion(y, data[:, -args.n_pred:, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_train += loss.cpu().detach().numpy()
        # args.logwriter.write("epoch %d | loss %.6f" % (epo, loss_train/len(train_loader)))
        print("epoch %d | loss %.6f" % (epo, loss_train/len(train_loader)))
        model.eval()
        predicts = inference(model, valid_loader, args, future_days=1)
        result = evaluate(valid_loader, predicts, args)
        # args.logwriter.write("valid result: %.6f" % (result['rmsle']))
        print("valid result: %.6f" % (result['rmsle']))
        if result['rmsle'] < best and epo > args.epochs//2:
            predicts = inference(model, test_loader, args, future_days=30)
            save_to_submit(predicts, args)
            best = result['rmsle']
    # args.logwriter.write("best result: %.6f" % (best))
    print("best result: %.6f" % (best))


def save_to_submit(predicts, args):
    # (n_pred, 1, city_num, 1) --> (n_pred, city_num)
    predicts = np.squeeze(predicts)
    if args.ylog:
        predicts = np.exp((predicts+0.4)*6) - 1
    else:
        predicts = predicts * 100
    predicts = predicts.transpose(1, 0).reshape(-1,).astype("int64")
    #  log.info(predicts)
    predicts = pd.DataFrame({"ret": predicts})

    submit = pd.read_csv(args.submit_file,
                         header=None,
                         names=["cityid", "regionid", "date", "cnt"])

    submit = pd.concat([submit, predicts], axis=1)
    submit = submit.drop(columns=['cnt'])

    submit.to_csv(os.path.join(args.output_path, 'submission.csv'),
                  index=False, header=False)


def evaluate(dataloader, y_pred, args):
    # log.info(["y_pred", np.squeeze(y_pred)])
    result = {}
    y_true_list = []
    for x_batch in dataloader:
        y_true_list.append(x_batch[:, args.n_his:(args.n_his + args.n_pred), :])

    # y_true: (n_pred, len, city_num, 1)
    y_true = np.concatenate(y_true_list, axis=0).transpose(1, 0, 2)
    # log.info(["y_true:", np.squeeze(y_true)])
    #  log.info(y_true)
    if args.ylog:
        diff = (y_pred - y_true) * 6
    else:
        diff = np.log(100*y_pred + 1) - np.log(100*y_true + 1)
    rmsle = np.sqrt(np.mean(diff**2))
    result['rmsle'] = rmsle
    return result


def inference(model, dataloader, args, future_days=30):
    pred_list = []
    for x_batch in dataloader:
        if args.use_cuda:
            x_batch = x_batch.cuda()
        test_seq = (torch.cat([x_batch[:, 0:args.n_his, :], x_batch[:, -1:, :]], axis=1)).double()
        step_list = []
        for j in range(future_days):
            pred = model(test_seq)
            if args.ylog:
                pred[pred < -0.4] = 0.0
            else:
                pred[pred < 0] = 0.0
            test_seq[:, 0:args.n_his - 1, :] = test_seq[:, 1:args.n_his, :]
            if args.ylog:
                test_seq[:, args.n_his - 1, :] = pred[:, 0, :]
            else:
                test_seq[:, args.n_his - 1, :] = pred[:, 0, :]
            step_list.append(pred[:, 0, :].cpu().detach().numpy())
        pred_list.append(step_list)
        # break
    pred_array = np.concatenate(pred_list, axis=1)
    # pred_array = np.array(pred_list)
    return pred_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--city_num', type=int, default=903)
    parser.add_argument('--feat_dim', type=int, default=1)
    parser.add_argument('--n_his', type=int, default=10)
    parser.add_argument('--n_pred', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--save', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ylog', type=bool, default=True)
    parser.add_argument('--gate', type=bool, default=False)
    parser.add_argument('--region_names_file', type=str,
            default='../../proj_baiduAI/dataset/data_processed_all/region_names.txt')
    parser.add_argument('--input_file', type=str,
            default='../../proj_baiduAI/dataset/data_processed_all/region_migration.csv')
    parser.add_argument('--label_file', type=str,
            default='../../proj_baiduAI/dataset/data_processed_all/infection.csv')
    parser.add_argument('--adj_mat_file', type=str,
            default='../../proj_baiduAI/dataset/data_processed_all/adj_matrix.npy')
    parser.add_argument('--submit_file', type=str,
            default='../../proj_baiduAI/dataset/train_data_all/submission.csv')
    parser.add_argument('--output_path', type=str, default='../outputs/')
    parser.add_argument('--val_num', type=int, default=5)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use_cuda', action='store_true',default=True)
    args = parser.parse_args()

    # set random seed to 0
    np.random.seed(0)
    torch.manual_seed(0)
    # load data and make training set
    train_loader, valid_loader, test_loader = dataload(args)
    dataloader = {'train':train_loader, 'valid': valid_loader, 'test':test_loader}
    os.environ["CUDA_VISIBLE_DEVICES"] = '5'
    # build the model
    seq = Gate_Sequence(args.use_cuda, args)
    # seq = Sequence(args.use_cuda)
    seq.double()
    # criterion = nn.MSELoss(size_average=True)
    criterion = Weighted_MSE(size_average=True)
    if args.use_cuda:
        seq, criterion = seq.cuda(), criterion.cuda()
    # use LBFGS as optimizer since we can load the whole data to train
    # optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    optimizer = torch.optim.Adam(seq.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_schedual = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.2, last_epoch=-1)

    type = sys.getfilesystemencoding()
    sys.stdout = LogWriter('./log/')
    print(seq)
    # args.logwriter = LogWriter('./log/')
    train(model=seq,
          criterion=criterion,
          optimizer=optimizer,
          lr_schedual=lr_schedual,
          dataloader=dataloader,
          args=args)
