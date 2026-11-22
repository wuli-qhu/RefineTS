from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, AverageMeter
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'MAE':
            criterion = nn.L1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = AverageMeter()
        self.model.eval()
        noise_std = getattr(self.args, 'noise_std', 0.0)
        noise_channel = getattr(self.args, 'noise_channel', None)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if noise_std > 0:
                    channel_std = batch_x.std(dim=1, keepdim=True)
                    noise = torch.randn_like(batch_x) * noise_std * channel_std
                    if noise_channel is not None:
                        mask = torch.zeros_like(batch_x)
                        mask[:, :, noise_channel] = 1.0
                        noise = noise * mask
                    batch_x = batch_x + noise

                if 'PEMS' in self.args.data or 'Solar' in self.args.data or 'ECL' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                total_loss.update(loss.item(), batch_x.size(0))
        total_loss = total_loss.avg
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            print(path)
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                aux_loss = getattr(self.model, 'aux_loss', None)
                if aux_loss is not None:
                    loss = loss + aux_loss

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            print(self.args.save_model)
            import shutil
            print("remove", path)
            shutil.rmtree(path)
        return self.model

    def test(self, setting, test=0, noise_std=0.0, noise_channel=None, missing_rate=0.0, spike_rate=0.0,
             spike_magnitude=5.0, leak_channel=None, leak_rate=0.0, leak_contiguous=False):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')))

        mse_loss = nn.MSELoss()
        mae_loss = nn.L1Loss()
        mse = AverageMeter()
        mae = AverageMeter()
        self.model.eval()

        preds = []
        trues = []
        inputx = []

        if noise_std == 0.0 and missing_rate == 0.0 and spike_rate == 0.0 and leak_rate == 0.0:
            folder_path = './test_results/' + setting + '/'
        elif noise_channel is None:
            folder_path = './test_results/' + setting + f'_noise{noise_std}/'
        elif missing_rate == 0.0 and spike_rate == 0.0 and leak_rate == 0.0:
            folder_path = './test_results/' + setting + f'_noise{noise_std}_ch{noise_channel}/'
        elif leak_rate == 0.0:
            folder_path = './test_results/' + setting + \
                f'_noise{noise_std}_ch{noise_channel}_miss{missing_rate}_spike{spike_rate}/'
        else:
            folder_path = './test_results/' + setting + \
                f'_noise{noise_std}_ch{noise_channel}_miss{missing_rate}_spike{spike_rate}' \
                f'_leak{leak_rate}from{leak_channel}/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pre_time = time.time()
        fwd_time = 0.0
        n_batches = 0
        n_samples = 0
        _sync = torch.cuda.synchronize if self.device.type == 'cuda' else (lambda: None)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if noise_std > 0:
                    channel_std = batch_x.std(dim=1, keepdim=True)  # [B, 1, C]
                    noise = torch.randn_like(batch_x) * noise_std * channel_std
                    if noise_channel is not None:
                        mask = torch.zeros_like(batch_x)
                        mask[:, :, noise_channel] = 1.0
                        noise = noise * mask
                    batch_x = batch_x + noise

                if spike_rate > 0 and noise_channel is not None:
                    chan_vals = batch_x[:, :, noise_channel]
                    chan_std = chan_vals.std(dim=1, keepdim=True)  # [B, 1]
                    spike_mask = torch.rand(batch_x.shape[0], batch_x.shape[1], device=batch_x.device) < spike_rate
                    spike_sign = torch.sign(torch.randn(batch_x.shape[0], batch_x.shape[1], device=batch_x.device))
                    spikes = spike_sign * spike_magnitude * chan_std
                    batch_x[:, :, noise_channel] = torch.where(spike_mask, chan_vals + spikes, chan_vals)

                if leak_rate > 0 and noise_channel is not None and leak_channel is not None:
                    if leak_contiguous:
                        seq_len = batch_x.shape[1]
                        block_len = max(1, round(leak_rate * seq_len))
                        starts = torch.randint(0, seq_len - block_len + 1, (batch_x.shape[0],), device=batch_x.device)
                        t_idx = torch.arange(seq_len, device=batch_x.device).unsqueeze(0)
                        leak_mask = (t_idx >= starts.unsqueeze(1)) & (t_idx < (starts + block_len).unsqueeze(1))
                    else:
                        leak_mask = torch.rand(batch_x.shape[0], batch_x.shape[1], device=batch_x.device) < leak_rate
                    batch_x[:, :, noise_channel] = torch.where(
                        leak_mask, batch_x[:, :, leak_channel], batch_x[:, :, noise_channel])

                if missing_rate > 0 and noise_channel is not None:
                    missing_mask = torch.rand(batch_x.shape[0], batch_x.shape[1], device=batch_x.device) < missing_rate
                    batch_x[:, :, noise_channel] = batch_x[:, :, noise_channel].masked_fill(missing_mask, 0.0)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                _sync()
                _fwd_start = time.time()
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                _sync()
                fwd_time += time.time() - _fwd_start
                n_batches += 1
                n_samples += batch_x.size(0)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse.update(mse_loss(outputs, batch_y).item(), batch_x.size(0))
                mae.update(mae_loss(outputs, batch_y).item(), batch_x.size(0))
                
                ##save
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
        total_time = time.time() - pre_time
        print(f"test cost: {total_time}")
        preds = np.concatenate(preds, axis=0)
        trues= np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)
        #preds = np.array(preds)
        #trues = np.array(trues)
        #inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])
       
        print("pred shape", preds.shape)
        print("trues shape", trues.shape)
        print("inputx shape", inputx.shape)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'x.npy', inputx)
        np.save(folder_path + 'true.npy', trues)

        mse_per_channel = np.mean((preds - trues) ** 2, axis=(0, 1))
        mae_per_channel = np.mean(np.abs(preds - trues), axis=(0, 1))
        self.last_mse_per_channel = mse_per_channel
        self.last_mae_per_channel = mae_per_channel

        mse = mse.avg
        mae = mae.avg
        print('noise_std:{}, mse:{}, mae:{}'.format(noise_std, mse, mae))
        if noise_channel is not None:
            others = np.delete(mse_per_channel, noise_channel)
            print('  noise_channel={}: channel_mse={:.6f}, other_channels_mse={:.6f}'.format(
                noise_channel, mse_per_channel[noise_channel], others.mean()))

        return mse, mae, mse_per_channel

    def train_noise(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)

                if self.args.noise_std > 0:
                    channel_std = batch_x.std(dim=1, keepdim=True)
                    noise = torch.randn_like(batch_x) * self.args.noise_std * channel_std
                    noise_channel = getattr(self.args, 'noise_channel', None)
                    if noise_channel is not None:
                        mask = torch.zeros_like(batch_x)
                        mask[:, :, noise_channel] = 1.0
                        noise = noise * mask
                    batch_x = batch_x + noise

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)

                aux_loss = getattr(self.model, 'aux_loss', None)
                if aux_loss is not None:
                    loss = loss + aux_loss

                if (i + 1) % 100 == 0:
                    loss_float = loss.item()
                    train_loss.append(loss_float)
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_float))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()


            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if not self.args.save_model:
            import shutil
            shutil.rmtree(path)
        return self.model

