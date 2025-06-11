import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from torch.utils.tensorboard import SummaryWriter
import scipy.misc
import time as tm

from utils import *
from model import *
from data import *
from args import Args
import create_graphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_rnn_epoch(epoch, args, rnn, output, data_loader,
                    optimizer_rnn, optimizer_output,
                    scheduler_rnn, scheduler_output, writer):
    rnn.train()
    output.train()
    loss_sum = 0
    for batch_idx, data in enumerate(data_loader):
        optimizer_rnn.zero_grad()
        optimizer_output.zero_grad()

        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']

        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, :y_len_max, :]
        y_unsorted = y_unsorted[:, :y_len_max, :]

        rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0)).to(device)

        y_len, sort_index = torch.sort(y_len_unsorted, 0, descending=True)
        y_len = y_len
        x = torch.index_select(x_unsorted, 0, sort_index).to(device)
        y = torch.index_select(y_unsorted, 0, sort_index).to(device)

        y_reshape = torch.nn.utils.rnn.pack_padded_sequence(y, y_len, batch_first=True).data
        y_reshape = torch.flip(y_reshape, [0])
        y_reshape = y_reshape.unsqueeze(-1)

        output_x = torch.cat((torch.ones(y_reshape.size(0), 1, 1, device=device), y_reshape[:, :-1, :]), dim=1)
        output_y = y_reshape

        output_y_len = []
        output_y_len_bin = np.bincount(np.array(y_len))
        for i in range(len(output_y_len_bin) - 1, 0, -1):
            count_temp = np.sum(output_y_len_bin[i:])
            output_y_len.extend([min(i, y.size(2))] * count_temp)

        h = rnn(x, pack=True, input_len=y_len)
        h = torch.nn.utils.rnn.pack_padded_sequence(h, y_len, batch_first=True).data
        h = torch.flip(h, [0])

        hidden_null = torch.zeros(args.num_layers - 1, h.size(0), h.size(1), device=device)
        output.hidden = torch.cat((h.unsqueeze(0), hidden_null), dim=0)

        y_pred = output(output_x, pack=True, input_len=output_y_len)
        y_pred = F.sigmoid(y_pred)

        y_pred = torch.nn.utils.rnn.pack_padded_sequence(y_pred, output_y_len, batch_first=True)
        y_pred = torch.nn.utils.rnn.pad_packed_sequence(y_pred, batch_first=True)[0]

        output_y = torch.nn.utils.rnn.pack_padded_sequence(output_y, output_y_len, batch_first=True)
        output_y = torch.nn.utils.rnn.pad_packed_sequence(output_y, batch_first=True)[0]

        loss = binary_cross_entropy_weight(y_pred, output_y)
        loss.backward()

        optimizer_output.step()
        optimizer_rnn.step()
        scheduler_output.step()
        scheduler_rnn.step()

        if epoch % args.epochs_log == 0 and batch_idx == 0:
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
                epoch, args.epochs, loss.item(), args.graph_type, args.num_layers, args.hidden_size_rnn))

        writer.add_scalar('loss_' + args.fname, loss.item(), epoch * args.batch_ratio + batch_idx)
        feature_dim = y.size(1) * y.size(2)
        loss_sum += loss.item() * feature_dim
    writer.close()
    return loss_sum / (batch_idx + 1)


def test_rnn_epoch(epoch, args, rnn, output, test_batch_size=16):
    rnn.hidden = rnn.init_hidden(test_batch_size)
    rnn.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node)
    y_pred_long = torch.zeros(test_batch_size, max_num_node, args.max_prev_node).to(device) # discrete prediction
    x_step = torch.ones(test_batch_size,1,args.max_prev_node).to(device)
    for i in range(max_num_node):
        h = rnn(x_step)
        hidden_null = torch.zeros(args.num_layers - 1, h.size(0), h.size(2)).to(device)
        output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                  dim=0)  # num_layers, batch_size, hidden_size
        x_step = torch.zeros(test_batch_size,1,args.max_prev_node).to(device)
        output_x_step = torch.ones(test_batch_size,1,1).to(device)
        for j in range(min(args.max_prev_node,i+1)):
            output_y_pred_step = output(output_x_step)
            output_x_step = sample_sigmoid(output_y_pred_step, sample=True, sample_time=1)
            x_step[:,:,j:j+1] = output_x_step
            output.hidden = output.hidden.data.to(device)
        y_pred_long[:, i:i + 1, :] = x_step
        rnn.hidden = rnn.hidden.data.to(device)
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy())
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)

    return G_pred_list

########### train function for LSTM + VAE
def train(args, dataset_train, rnn, output, writer):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch<=args.epochs:
        time_start = tm.time()

        train_rnn_epoch(epoch, args, rnn, output, dataset_train,
                        optimizer_rnn, optimizer_output,
                        scheduler_rnn, scheduler_output, writer)

        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            for sample_time in range(1,4):
                G_pred = []
                while len(G_pred)<args.test_total_size:
                    G_pred_step = test_rnn_epoch(epoch, args, rnn, output, test_batch_size=args.test_batch_size)
                    G_pred.extend(G_pred_step)
                # save graphs
                fname = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat'
                save_graph_list(G_pred, fname)
                if 'GraphRNN_RNN' in args.note:
                    break
            print('test done, graphs saved')


        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path+args.fname,time_all)