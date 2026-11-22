import argparse
import csv
import os
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

torch.set_num_threads(6)
parser = argparse.ArgumentParser(description='CALFS')
# basic config
parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='RefineTS',
                    help='model name (RefineTS)')
# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--data_adj', type=str, default='adj_tg_ETTh1.csv', help='data adj file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seed', type=int, default=2025, help='random seed')
parser.add_argument('--results_csv', type=str, default='', help='if set, append final test mse/mae as a row to this CSV')
# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--d_core', type=int, default=512, help='dimension of core')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--attention_type', type=str, default="full", help='the attention type of transformer')
parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')

# optimization
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

parser.add_argument('--save_model', action='store_true', default=True)

####Adaptive Channel Normalization; CN 论文按数据集在
#### {0.1, 0.2, 0.5} 中调, 其它模型忽略该参数
parser.add_argument('--temperature', type=float, default=0.2,
                    help='ACN: softmax temperature of the channel-similarity weighting')

###lora
parser.add_argument('--rank', type=int, default=32, help='Lora Rank')
parser.add_argument('--node_dim', type=int, default=64, help='node dim')

####iTransformer
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')

####DUET (defaults follow the official DUET repo's MODEL_HYPER_PARAMS)
parser.add_argument('--num_experts', type=int, default=4, help='DUET: number of linear pattern-extractor experts')
parser.add_argument('--k', type=int, default=1, help='DUET: number of experts activated per sample (top-k gating)')
parser.add_argument('--hidden_size', type=int, default=256, help='DUET: hidden size of the router encoder')
parser.add_argument('--noisy_gating', type=int, default=1, help='DUET: noisy top-k gating')
parser.add_argument('--fc_dropout', type=float, default=0.2, help='DUET: dropout of the linear head')
parser.add_argument('--CI', type=int, default=1, help='DUET: channel-independent expert extraction')

###partial
parser.add_argument('--partial_start_index', type=int, default=0, help='the start index of variates for partial training, '
                                                                           'you can select [partial_start_index, min(enc_in + partial_start_index, N)]')

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.data == 'PEMS':
    args.adj = np.load(args.root_path + args.data_adj)
else:
    args.adj = np.loadtxt(args.root_path + args.data_adj, delimiter=',')

#args.adj = np.loadtxt('./dataset/weather/adj_tg_weather.csv',delimiter=',')
#args.adj = np.loadtxt('./dataset/electricity/adj_tg_electricity.csv',delimiter=',')
#args.adj = np.load('./dataset/PEMS/pems08_adj.npy')
#args.adj = np.loadtxt('./dataset/traffic/adj_tg_traffic.csv',delimiter=',')
#args.adj = np.loadtxt('./dataset/ETT-small/adj_tg_ETTh2.csv',delimiter=',')
#print(args.adj)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
print('Args in experiment:')
print(args)
Exp = Exp_Long_Term_Forecast


def train(args=args):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des)
    exp = Exp(args)  # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    mse, mae, _ = exp.test(setting)
    torch.cuda.empty_cache()

    if args.results_csv:
        write_header = not os.path.exists(args.results_csv)
        os.makedirs(os.path.dirname(args.results_csv), exist_ok=True) if os.path.dirname(args.results_csv) else None
        with open(args.results_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['model', 'data', 'pred_len', 'seed', 'mse', 'mae', 'setting'])
            w.writerow([args.model, args.data, args.pred_len, args.seed, mse, mae, setting])


if args.is_training:
    train(args)
else:
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des)
    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
