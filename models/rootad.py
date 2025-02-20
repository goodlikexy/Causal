import utils.utils 
from models.senn import SENNGC
import torch.nn as nn
import torch
from utils.utils import compute_kl_divergence, sliding_window_view_torch, eval_causal_structure, eval_causal_structure_binary
from numpy.lib.stride_tricks import sliding_window_view
import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np


class RootAD(nn.Module):
    """
    RootAD 模型类
    """
    
    def __init__(self, num_vars: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 window_size: int, stride: int = 1, encoder_alpha: float = 0.5, decoder_alpha: float = 0.5,
                 encoder_gamma: float = 0.5, decoder_gamma: float = 0.5,
                 encoder_lambda: float = 0.5, decoder_lambda: float = 0.5,
                 beta: float = 0.5, lr: float = 0.0001, epochs: int = 100,
                 recon_threshold: float = 0.95, data_name: str = 'ld',
                 causal_quantile: float = 0.80, root_cause_threshold_encoder: float = 0.95,
                 root_cause_threshold_decoder: float = 0.95, initial_z_score: float = 3.0,
                 risk=1e-2, initial_level=0.98, num_candidates=100):
        """
        初始化 RootAD 模型
        
        参数：
        num_vars : 输入维度
        hidden_layer_size : 隐藏层维度
        num_hidden_layers : 隐藏层数量
        device : 运行设备 ('cuda' 或 'cpu')
        window_size : 窗口大小
        stride : 步长
        encoder_alpha : 编码器稀疏性参数
        decoder_alpha : 解码器稀疏性参数
        encoder_gamma : 编码器平滑性参数
        decoder_gamma : 解码器平滑性参数
        encoder_lambda : 编码器稀疏性惩罚参数
        decoder_lambda : 解码器稀疏性惩罚参数
        beta : VAE 的 beta 参数
        lr : 学习率
        epochs : 训练轮数
        recon_threshold : 重构阈值
        data_name : 数据集名称
        causal_quantile : 因果结构阈值
        root_cause_threshold_encoder : 编码器根因阈值
        root_cause_threshold_decoder : 解码器根因阈值
        initial_z_score : 初始z分数
        risk : 风险参数
        initial_level : 初始水平
        num_candidates : 候选数量
        """
        super(RootAD, self).__init__()
        self.encoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device)
        self.decoder = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device)
        self.decoder_prev = SENNGC(num_vars, window_size, hidden_layer_size, num_hidden_layers, device)
        self.device = device
        self.num_vars = num_vars
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.window_size = window_size
        self.stride = stride
        self.encoder_alpha = encoder_alpha
        self.decoder_alpha = decoder_alpha
        self.encoder_gamma = encoder_gamma
        self.decoder_gamma = decoder_gamma
        self.encoder_lambda = encoder_lambda
        self.decoder_lambda = decoder_lambda
        self.beta = beta
        self.lr = lr
        self.epochs = epochs
        self.recon_threshold = recon_threshold
        self.root_cause_threshold_encoder = root_cause_threshold_encoder
        self.root_cause_threshold_decoder = root_cause_threshold_decoder
        self.initial_z_score = initial_z_score
        self.mse_loss = nn.MSELoss()
        self.mse_loss_wo_reduction = nn.MSELoss(reduction='none')
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.decoder_prev.to(self.device)
        self.model_name = 'rootad_' + data_name + '_ws_' + str(window_size) + '_stride_' + str(stride) + \
                            '_encoder_alpha_' + str(encoder_alpha) + '_decoder_alpha_' + str(decoder_alpha) + \
                            '_encoder_gamma_' + str(encoder_gamma) + '_decoder_gamma_' + str(decoder_gamma) + \
                            '_encoder_lambda_' + str(encoder_lambda) + '_decoder_lambda_' + str(decoder_lambda) + \
                            '_beta_' + str(beta) + '_lr_' + str(lr) + '_epochs_' + str(epochs) + \
                            '_hidden_layer_size_' + str(hidden_layer_size) + '_num_hidden_layers_' + \
                            str(num_hidden_layers)
        self.causal_quantile = causal_quantile
        self.risk = risk
        self.initial_level = initial_level
        self.num_candidates = num_candidates


    def encoding(self, xs):
        windows = sliding_window_view(xs, (self.window_size+1, self.num_vars))[:, 0, :, :]
        winds = windows[:, :-1, :]
        nexts = windows[:, -1, :]
        winds = torch.tensor(winds).float().to(self.device)
        nexts = torch.tensor(nexts).float().to(self.device)
        preds, coeffs = self.encoder(winds)
        us = preds - nexts
        return us, coeffs, nexts[self.window_size:], winds[:-self.window_size]

    def decoding(self, us, winds, add_u=True):
        u_windows = sliding_window_view_torch(us, self.window_size+1)
        u_winds = u_windows[:, :-1, :]
        u_next = u_windows[:, -1, :]

        preds, coeffs = self.decoder(u_winds)
        prev_preds, prev_coeffs = self.decoder_prev(winds)

        if add_u:
            nexts_hat = preds + u_next + prev_preds
        else:
            nexts_hat = preds + prev_preds
        return nexts_hat, coeffs, prev_coeffs

    def forward(self, x, add_u=True):
        us, encoder_coeffs, nexts, winds = self.encoding(x)
        kl_div = compute_kl_divergence(us, self.device)
        nexts_hat, decoder_coeffs, prev_coeffs = self.decoding(us, winds, add_u=add_u)
        return nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us

    def _training_step(self, x, add_u=True, verbose=False):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)
        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        loss_encoder_coeffs = (1 - self.encoder_alpha) * torch.mean(torch.mean(torch.norm(encoder_coeffs, dim=1, p=2),
                dim=0)) + self.encoder_alpha * torch.mean(torch.mean(torch.norm(encoder_coeffs, dim=1, p=1), dim=0))
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        loss_decoder_coeffs = (1 - self.decoder_alpha) * torch.mean(torch.mean(torch.norm(decoder_coeffs, dim=1, p=2),
                dim=0)) + self.decoder_alpha * torch.mean(torch.mean(torch.norm(decoder_coeffs, dim=1, p=1), dim=0))
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        loss_prev_coeffs = (1 - self.decoder_alpha) * torch.mean(torch.mean(torch.norm(prev_coeffs, dim=1, p=2),
                dim=0)) + self.decoder_alpha * torch.mean(torch.mean(torch.norm(prev_coeffs, dim=1, p=1), dim=0))
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())


        #  Smoothness-inducing penalty term
        loss_encoder_smooth = torch.norm(encoder_coeffs[:, 1:, :, :] - encoder_coeffs[:, :-1, :, :], dim=1).mean()
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        #  Smoothness-inducing penalty term
        loss_decoder_smooth = torch.norm(decoder_coeffs[:, 1:, :, :] - decoder_coeffs[:, :-1, :, :], dim=1).mean()
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        # Smoothness-inducing penalty term
        loss_prev_smooth = torch.norm(prev_coeffs[:, 1:, :, :] - prev_coeffs[:, :-1, :, :], dim=1).mean()
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        # KL divergence term
        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        loss_encoder_recon = self.mse_loss(us, torch.zeros_like(us))

        # Total loss
        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * loss_decoder_coeffs +
                self.decoder_lambda * loss_prev_coeffs +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * loss_decoder_smooth +
                self.decoder_gamma * loss_prev_smooth +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss


    def _training(self, xs):
        if len(xs) == 1:
            xs_train = xs[:, :int(0.8*len(xs[0]))]
            xs_val = xs[:, int(0.8*len(xs[0])):]
        else:
            xs_train = xs[:int(0.8*len(xs))]
            xs_val = xs[int(0.8*len(xs)):]
        # xs_train = xs
        # xs_val = xs
        best_val_loss = np.inf
        count = 0
        for epoch in range(self.epochs):
            count += 1
            epoch_loss = 0
            self.train()
            for x in tqdm(xs_train, desc='Epoch '+str(epoch+1)+'/'+str(self.epochs)):
                self.optimizer.zero_grad()
                loss = self._training_step(x)
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            logging.info('Epoch %s/%s', epoch+1, self.epochs)
            logging.info('Epoch training loss: %s', epoch_loss)
            logging.info('-------------------')
            epoch_val_loss = 0
            self.eval()
            with torch.no_grad():
                for x in xs_val:
                    loss = self._training_step(x, verbose=True)
                    epoch_val_loss += loss.item()
            logging.info('Epoch val loss: %s', epoch_val_loss)
            logging.info('-------------------')
            if epoch_val_loss < best_val_loss:
                count = 0
                print(f'Saving model at epoch', epoch+1)
                print(f'Saving model name: {self.model_name}.pt')
                best_val_loss = epoch_val_loss
                torch.save(self.state_dict(), f'./saved_models/{self.model_name}.pt')
            if count >= 20:
                print('Early stopping')
                break
        self.load_state_dict(torch.load(f'./saved_models/{self.model_name}.pt', map_location=self.device))
        logging.info('Training complete')
        self._get_recon_threshold(xs_val)
        self._get_root_cause_threshold_encoder(xs_val)
        self._get_root_cause_threshold_decoder(xs_val)

    def _testing_step(self, x, label=None, add_u=True):
        nexts_hat, nexts, encoder_coeffs, decoder_coeffs, prev_coeffs, kl_div, us = self.forward(x, add_u=add_u)

        if label is not None:
            preprocessed_label = sliding_window_view(label, (self.window_size+1, self.num_vars))[self.window_size:, 0, :-1, :]
        else:
            preprocessed_label = None

        loss_recon = self.mse_loss(nexts_hat, nexts)
        logging.info('Reconstruction loss: %s', loss_recon.item())

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        loss_encoder_coeffs = (1 - self.encoder_alpha) * torch.mean(torch.mean(torch.norm(encoder_coeffs, dim=1, p=2),
                dim=0)) + self.encoder_alpha * torch.mean(torch.mean(torch.norm(encoder_coeffs, dim=1, p=1), dim=0))
        logging.info('Encoder coeffs loss: %s', loss_encoder_coeffs.item())

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        loss_decoder_coeffs = (1 - self.decoder_alpha) * torch.mean(torch.mean(torch.norm(decoder_coeffs, dim=1, p=2),
                dim=0)) + self.decoder_alpha * torch.mean(torch.mean(torch.norm(decoder_coeffs, dim=1, p=1), dim=0))
        logging.info('Decoder coeffs loss: %s', loss_decoder_coeffs.item())

        # Sparsity-inducing penalty term
        # coeffs.shape:     [T x K x p x p]
        loss_prev_coeffs = (1 - self.decoder_alpha) * torch.mean(torch.mean(torch.norm(prev_coeffs, dim=1, p=2),
                dim=0)) + self.decoder_alpha * torch.mean(torch.mean(torch.norm(prev_coeffs, dim=1, p=1), dim=0))
        logging.info('Prev coeffs loss: %s', loss_prev_coeffs.item())

        #  Smoothness-inducing penalty term
        loss_encoder_smooth = torch.norm(encoder_coeffs[:, 1:, :, :] - encoder_coeffs[:, :-1, :, :], dim=1).mean()
        logging.info('Encoder smooth loss: %s', loss_encoder_smooth.item())

        #  Smoothness-inducing penalty term
        loss_decoder_smooth = torch.norm(decoder_coeffs[:, 1:, :, :] - decoder_coeffs[:, :-1, :, :], dim=1).mean()
        logging.info('Decoder smooth loss: %s', loss_decoder_smooth.item())

        # Smoothness-inducing penalty term
        loss_prev_smooth = torch.norm(prev_coeffs[:, 1:, :, :] - prev_coeffs[:, :-1, :, :], dim=1).mean()
        logging.info('Prev smooth loss: %s', loss_prev_smooth.item())

        # KL divergence term
        loss_kl = kl_div
        logging.info('KL loss: %s', loss_kl.item())

        # Total loss
        loss = (loss_recon +
                self.encoder_lambda * loss_encoder_coeffs +
                self.decoder_lambda * loss_decoder_coeffs +
                self.decoder_lambda * loss_prev_coeffs +
                self.encoder_gamma * loss_encoder_smooth +
                self.decoder_gamma * loss_decoder_smooth +
                self.decoder_gamma * loss_prev_smooth +
                self.beta * loss_kl)
        logging.info('Total loss: %s', loss.item())

        return loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us

    def _get_recon_threshold(self, xs):
        self.eval()
        recon_losses = np.array([])
        with torch.no_grad():
            for x in xs:
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, _, _ = self._testing_step(x, add_u=False)
                recon_losses = np.append(recon_losses, self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy())
        self.recon_threshold_value = np.quantile(recon_losses, self.recon_threshold)
        self.recon_mean = np.mean(recon_losses)
        self.recon_std = np.std(recon_losses)
        np.save('./saved_models/' + self.model_name + '_recon_threshold.npy', self.recon_threshold_value)
        np.save('./saved_models/' + self.model_name + '_recon_mean.npy', self.recon_mean)
        np.save('./saved_models/' + self.model_name + '_recon_std.npy', self.recon_std)


    def _get_root_cause_threshold_encoder(self, xs):
        self.eval()
        us_all = np.array([])
        with torch.no_grad():
            for x in xs:
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, _, us = self._testing_step(x)
                us_all = np.append(us_all, us.cpu().numpy())
        us_all = us_all.reshape(-1, self.num_vars)
        self.lower_encoder = np.quantile(us_all, (1-self.root_cause_threshold_encoder)/2, axis=0)
        self.upper_encoder = np.quantile(us_all, 1-(1-self.root_cause_threshold_encoder)/2, axis=0)
        # self.us_mean_encoder = np.mean(us_all, axis=0)
        self.us_mean_encoder = np.median(us_all, axis=0)
        self.us_std_encoder = np.std(us_all, axis=0)
        np.save('./saved_models/' + self.model_name + '_lower_encoder.npy', self.lower_encoder)
        np.save('./saved_models/' + self.model_name + '_upper_encoder.npy', self.upper_encoder)
        np.save('./saved_models/' + self.model_name + '_us_mean_encoder.npy', self.us_mean_encoder)
        np.save('./saved_models/' + self.model_name + '_us_std_encoder.npy', self.us_std_encoder)

    def _get_root_cause_threshold_decoder(self, xs):
        self.eval()
        us_all = np.array([])
        with torch.no_grad():
            for x in xs:
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, _, us = self._testing_step(x, add_u=False)
                us_all = np.append(us_all, (nexts-nexts_hat).cpu().numpy())
        us_all = us_all.reshape(-1, self.num_vars)
        self.lower_decoder = np.quantile(us_all, (1-self.root_cause_threshold_decoder)/2, axis=0)
        self.upper_decoder = np.quantile(us_all, 1-(1-self.root_cause_threshold_decoder)/2, axis=0)
        self.us_mean_decoder = np.mean(us_all, axis=0)
        self.us_std_decoder = np.std(us_all, axis=0)
        np.save('./saved_models/' + self.model_name + '_lower_decoder.npy', self.lower_decoder)
        np.save('./saved_models/' + self.model_name + '_upper_decoder.npy', self.upper_decoder)
        np.save('./saved_models/' + self.model_name + '_us_mean_decoder.npy', self.us_mean_decoder)
        np.save('./saved_models/' + self.model_name + '_us_std_decoder.npy', self.us_std_decoder)

    def _testing_root_cause(self, xs, labels):
        self.load_state_dict(torch.load(f'./saved_models/{self.model_name}.pt', map_location=self.device))
        self.eval()
        self.recon_threshold_value = np.load('./saved_models/' + self.model_name + '_recon_threshold.npy')
        self.recon_mean = np.load('./saved_models/' + self.model_name + '_recon_mean.npy')
        self.recon_std = np.load('./saved_models/' + self.model_name + '_recon_std.npy')
        self.lower_encoder = np.load('./saved_models/' + self.model_name + '_lower_encoder.npy')
        self.upper_encoder = np.load('./saved_models/' + self.model_name + '_upper_encoder.npy')
        self.us_mean_encoder = np.load('./saved_models/' + self.model_name + '_us_mean_encoder.npy')
        self.us_std_encoder = np.load('./saved_models/' + self.model_name + '_us_std_encoder.npy')
        self.lower_decoder = np.load('./saved_models/' + self.model_name + '_lower_decoder.npy')
        self.upper_decoder = np.load('./saved_models/' + self.model_name + '_upper_decoder.npy')
        self.us_mean_decoder = np.load('./saved_models/' + self.model_name + '_us_mean_decoder.npy')
        self.us_std_decoder = np.load('./saved_models/' + self.model_name + '_us_std_decoder.npy')
        pred_labels = np.array([])
        true_labels = np.array([])
        true_root_cause = np.array([])
        us_all = np.array([])
        recons_all = np.array([])
        us_decoder_all = np.array([])
        us_sample = []
        us_sample_decoder = []
        with torch.no_grad():
            for i in range(len(xs)):
                x = xs[i]
                label = labels[i]
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, preprocessed_label, us = self._testing_step(x, label, add_u=False)
                recon_loss = self.mse_loss_wo_reduction(nexts_hat, nexts).cpu().numpy()
                recons_all = np.append(recons_all, recon_loss)
                pred_labels = np.append(pred_labels, recon_loss > self.recon_threshold_value)
                true_labels = np.append(true_labels, np.max(preprocessed_label, axis=1))
                us = us[self.window_size:]
                us_sample.append(us.cpu().numpy())
                us_all = np.append(us_all, us.cpu().numpy())
                us_decoder_all = np.append(us_decoder_all, (nexts-nexts_hat).cpu().numpy())
                us_sample_decoder.append((nexts-nexts_hat).cpu().numpy())
                true_root_cause = np.append(true_root_cause, label[self.window_size*2:])
        us_all = us_all.reshape(-1, self.num_vars)
        us_decoder_all = us_decoder_all.reshape(-1, self.num_vars)
        print('='*50)
        # print('Initial z-score performance')
        # print('Anomaly detection performance')
        # pred_labels_z_score = (-(recons_all - self.recon_mean) / self.recon_std) > self.initial_z_score
        # pred_labels_z_score = pred_labels_z_score.astype(int)
        # print('Confusion matrix: ', confusion_matrix(true_labels, pred_labels_z_score))
        # print('Classification report: ', classification_report(true_labels, pred_labels_z_score, digits=5))
        # print('AUC: {:.5f}'.format(roc_auc_score(true_labels, pred_labels_z_score)))
        # print('Average precision: {:.5f}'.format(average_precision_score(true_labels, pred_labels_z_score)))
        # print('Root cause detection performance')
        # us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        # pred_root_cause_encoder_z_score = us_all_z_score > self.initial_z_score
        # pred_root_cause_encoder_z_score = pred_root_cause_encoder_z_score.astype(int)
        # us_decoder_all_z_score = ((us_decoder_all - self.us_mean_decoder) / self.us_std_decoder)
        # pred_root_cause_decoder_z_score = us_decoder_all_z_score > self.initial_z_score
        # pred_root_cause_decoder_z_score = pred_root_cause_decoder_z_score.astype(int)
        # pred_root_cause_encoder_z_score = pred_root_cause_encoder_z_score.reshape(-1)
        # pred_root_cause_decoder_z_score = pred_root_cause_decoder_z_score.reshape(-1)
        # print('Encoder Confusion matrix: ', confusion_matrix(true_root_cause, pred_root_cause_encoder_z_score))
        # print('Encoder Classification report: ', classification_report(true_root_cause, pred_root_cause_encoder_z_score, digits=5))
        # print('Encoder AUC: {:.5f}'.format(roc_auc_score(true_root_cause, pred_root_cause_encoder_z_score,)))
        # print('Encoder Average precision: {:.5f}'.format(average_precision_score(true_root_cause, pred_root_cause_encoder_z_score)))
        # print('Decoder Confusion matrix: ', confusion_matrix(true_root_cause, pred_root_cause_decoder_z_score))
        # print('Decoder Classification report: ', classification_report(true_root_cause, pred_root_cause_decoder_z_score, digits=5))
        # print('Decoder AUC: {:.5f}'.format(roc_auc_score(true_root_cause, pred_root_cause_decoder_z_score, )))
        # print('Decoder Average precision: {:.5f}'.format(average_precision_score(true_root_cause, pred_root_cause_decoder_z_score)))

        # print('='*50)
        # print('POT z-score performance')
        # print('Anomaly detection performance')
        # z_scores = (-(recons_all - self.recon_mean) / self.recon_std)
        # recons_pot, recons_initial = utils.utils.pot(z_scores, self.risk, self.initial_level, self.num_candidates)
        # pred_labels_z_score_pot = z_scores > recons_pot
        # pred_labels_z_score_pot = pred_labels_z_score_pot.astype(int)
        # print('POT z-score threshold: {:.5f}'.format(recons_pot))
        # print('POT initial threshold: {:.5f}'.format(recons_initial))
        # print('Confusion matrix: ', confusion_matrix(true_labels, pred_labels_z_score_pot))
        # print('Classification report: ', classification_report(true_labels, pred_labels_z_score_pot, digits=5))
        # print('AUC: {:.5f}'.format(roc_auc_score(true_labels, pred_labels_z_score_pot)))
        # print('Average precision: {:.5f}'.format(average_precision_score(true_labels, pred_labels_z_score_pot)))
        # print('Root cause detection performance')
        us_all_z_score = (-(us_all - self.us_mean_encoder) / self.us_std_encoder)
        us_all_z_score_pot = np.array([])
        us_all_z_score_initial = np.array([])
        for i in range(self.num_vars):
            us_all_z_score_pot = np.append(us_all_z_score_pot, utils.utils.pot(us_all_z_score[:, i],
                                                                               self.risk, self.initial_level, self.num_candidates)[0])
            us_all_z_score_initial = np.append(us_all_z_score_initial, utils.utils.pot(us_all_z_score[:, i],
                                                                                       self.risk, self.initial_level, self.num_candidates)[1])
        pred_root_cause_encoder_z_score_pot = us_all_z_score > us_all_z_score_pot
        pred_root_cause_encoder_z_score_pot = pred_root_cause_encoder_z_score_pot.astype(int)
        us_decoder_all_z_score = ((us_decoder_all - self.us_mean_decoder) / self.us_std_decoder)
        us_decoder_all_z_score_pot = np.array([])
        us_decoder_all_z_score_initial = np.array([])
        for i in range(self.num_vars):
            us_decoder_all_z_score_pot = np.append(us_decoder_all_z_score_pot, utils.utils.pot(us_decoder_all_z_score[:, i],
                                                                                               self.risk, self.initial_level, self.num_candidates)[0])
            us_decoder_all_z_score_initial = np.append(us_decoder_all_z_score_initial, utils.utils.pot(us_decoder_all_z_score[:, i],
                                                                                                       self.risk, self.initial_level, self.num_candidates)[1])
        pred_root_cause_decoder_z_score_pot = us_decoder_all_z_score > us_decoder_all_z_score_pot
        pred_root_cause_decoder_z_score_pot = pred_root_cause_decoder_z_score_pot.astype(int)
        pred_root_cause_encoder_z_score_pot = pred_root_cause_encoder_z_score_pot.reshape(-1)
        pred_root_cause_decoder_z_score_pot = pred_root_cause_decoder_z_score_pot.reshape(-1)
        print('POT RC encoder threshold: {}'.format(us_all_z_score_pot))
        print('POT RC encoder initial threshold: {}'.format(us_all_z_score_initial))
        print('POT RC decoder threshold: {}'.format(us_decoder_all_z_score_pot))
        print('POT RC decoder initial threshold: {}'.format(us_decoder_all_z_score_initial))
        print('Encoder Confusion matrix: ', confusion_matrix(true_root_cause, pred_root_cause_encoder_z_score_pot))
        print('Encoder Classification report: ', classification_report(true_root_cause, pred_root_cause_encoder_z_score_pot, digits=5))
        print('Encoder AUC: {:.5f}'.format(roc_auc_score(true_root_cause, pred_root_cause_encoder_z_score_pot,)))
        print('Encoder Average precision: {:.5f}'.format(average_precision_score(true_root_cause, pred_root_cause_encoder_z_score_pot)))
        print('Decoder Confusion matrix: ', confusion_matrix(true_root_cause, pred_root_cause_decoder_z_score_pot))
        print('Decoder Classification report: ', classification_report(true_root_cause, pred_root_cause_decoder_z_score_pot, digits=5))
        print('Decoder AUC: {:.5f}'.format(roc_auc_score(true_root_cause, pred_root_cause_decoder_z_score_pot, )))
        print('Decoder Average precision: {:.5f}'.format(average_precision_score(true_root_cause, pred_root_cause_decoder_z_score_pot)))

        print('='*50)
        print('POT z-score with top-k performance')
        print('Root cause detection performance')
        k_all = []
        k_at_step_all = []
        for i in range(len(xs)):
            us = us_sample[i]
            z_scores = (-(us - self.us_mean_encoder)/self.us_std_encoder)
            k_lst = utils.utils.topk(z_scores, labels[i][self.window_size*2:], us_all_z_score_pot)
            k_at_step = utils.utils.topk_at_step(z_scores, labels[i][self.window_size*2:])
            k_all.append(k_lst)
            k_at_step_all.append(k_at_step)
        k_all = np.array(k_all).mean(axis=0)
        k_at_step_all = np.array(k_at_step_all).mean(axis=0)
        print('POT RC: {}'.format(k_all))
        print('POT RC average: {}'.format(np.mean(k_all)))
        print('RC at step: {}'.format(k_at_step_all))
        print('RC at step average: {}'.format(np.mean(k_at_step_all)))

        return pred_labels

    def _testing_causal_discover(self, xs, causal_struct_value):
        self.load_state_dict(torch.load(f'./saved_models/{self.model_name}.pt', map_location=self.device))
        self.eval()
        encoder_causal_struct_estimate_lst = np.array([])
        decoder_causal_struct_estimate_lst = np.array([])
        with torch.no_grad():
            for x in xs:
                loss, nexts_hat, nexts, encoder_coeffs, decoder_coeffs, kl_div, _, _ = self._testing_step(x)
                #这里修改了删除了abs绝对值
                encoder_causal_struct_estimate_temp = torch.max(torch.median(encoder_coeffs, dim=0)[0], dim=0).values.cpu().numpy()#torch.median(torch.abs(encoder_coeffs), dim=0)[0]
                decoder_causal_struct_estimate_temp = torch.max(torch.median(decoder_coeffs, dim=0)[0], dim=0).values.cpu().numpy()#torch.median(torch.abs(decoder_coeffs), dim=0)[0]
                encoder_causal_struct_estimate_lst = np.append(encoder_causal_struct_estimate_lst, encoder_causal_struct_estimate_temp)
                decoder_causal_struct_estimate_lst = np.append(decoder_causal_struct_estimate_lst, decoder_causal_struct_estimate_temp)
        encoder_causal_struct_estimate_lst = encoder_causal_struct_estimate_lst.reshape(-1, self.num_vars, self.num_vars)
        decoder_causal_struct_estimate_lst = decoder_causal_struct_estimate_lst.reshape(-1, self.num_vars, self.num_vars)
        # Causal structure estimation performance
        encoder_auroc = np.array([])
        encoder_auprc = np.array([])
        decoder_auroc = np.array([])
        decoder_auprc = np.array([])
        encoder_acc = np.array([])
        decoder_acc = np.array([])
        encoder_ba = np.array([])
        decoder_ba = np.array([])
        encoder_hamming = np.array([])
        decoder_hamming = np.array([])
        encoder_f1 = np.array([])
        decoder_f1 = np.array([])
        for i in range(len(encoder_causal_struct_estimate_lst)):
            encoder_auroc_temp, encoder_auprc_temp = eval_causal_structure(a_true=causal_struct_value, a_pred=encoder_causal_struct_estimate_lst[i])
            decoder_auroc_temp, decoder_auprc_temp = eval_causal_structure(a_true=causal_struct_value, a_pred=decoder_causal_struct_estimate_lst[i])
            encoder_auroc = np.append(encoder_auroc, encoder_auroc_temp)
            encoder_auprc = np.append(encoder_auprc, encoder_auprc_temp)
            decoder_auroc = np.append(decoder_auroc, decoder_auroc_temp)
            decoder_auprc = np.append(decoder_auprc, decoder_auprc_temp)
            encoder_q = np.quantile(a=encoder_causal_struct_estimate_lst[i], q=self.causal_quantile)
            encoder_a_hat_binary = (encoder_causal_struct_estimate_lst[i] >= encoder_q) * 1.0
            decoder_q = np.quantile(a=decoder_causal_struct_estimate_lst[i], q=self.causal_quantile)
            decoder_a_hat_binary = (decoder_causal_struct_estimate_lst[i] >= decoder_q) * 1.0
            encoder_acc_temp, encoder_ba_temp, _, _, encoder_hamming_temp = eval_causal_structure_binary(a_true=causal_struct_value, a_pred=encoder_a_hat_binary)
            decoder_acc_temp, decoder_ba_temp, _, _, decoder_hamming_temp = eval_causal_structure_binary(a_true=causal_struct_value, a_pred=decoder_a_hat_binary)
            encoder_acc = np.append(encoder_acc, encoder_acc_temp)
            encoder_ba = np.append(encoder_ba, encoder_ba_temp)
            encoder_hamming = np.append(encoder_hamming, encoder_hamming_temp)
            decoder_acc = np.append(decoder_acc, decoder_acc_temp)
            decoder_ba = np.append(decoder_ba, decoder_ba_temp)
            decoder_hamming = np.append(decoder_hamming, decoder_hamming_temp)
            encoder_f1 = np.append(encoder_f1, f1_score(causal_struct_value.flatten(), encoder_a_hat_binary.flatten()))
            decoder_f1 = np.append(decoder_f1, f1_score(causal_struct_value.flatten(), decoder_a_hat_binary.flatten()))
        logging.info('Encoder AUROC: %s', np.mean(encoder_auroc))
        logging.info('Encoder AUPRC: %s', np.mean(encoder_auprc))
        logging.info('Decoder AUROC: %s', np.mean(decoder_auroc))
        logging.info('Decoder AUPRC: %s', np.mean(decoder_auprc))
        logging.info('Encoder accuracy: %s', np.mean(encoder_acc))
        logging.info('Encoder balanced accuracy: %s', np.mean(encoder_ba))
        logging.info('Encoder hamming: %s', np.mean(encoder_hamming))
        logging.info('Decoder accuracy: %s', np.mean(decoder_acc))
        logging.info('Decoder balanced accuracy: %s', np.mean(decoder_ba))
        logging.info('Decoder hamming: %s', np.mean(decoder_hamming))

        print('Encoder AUROC: {:.5f} std: {:.5f}'.format(np.mean(encoder_auroc), np.std(encoder_auroc)))
        print('Encoder AUPRC: {:.5f} std: {:.5f}'.format(np.mean(encoder_auprc), np.std(encoder_auprc)))
        print('Decoder AUROC: {:.5f} std: {:.5f}'.format(np.mean(decoder_auroc), np.std(decoder_auroc)))
        print('Decoder AUPRC: {:.5f} std: {:.5f}'.format(np.mean(decoder_auprc), np.std(decoder_auprc)))

        print('Encoder accuracy: {:.5f} std: {:.5f}'.format(np.mean(encoder_acc), np.std(encoder_acc)))
        print('Encoder balanced accuracy: {:.5f} std: {:.5f}'.format(np.mean(encoder_ba), np.std(encoder_ba)))
        print('Encoder hamming: {:.5f} std: {:.5f}'.format(np.mean(encoder_hamming), np.std(encoder_hamming)))
        print('Decoder accuracy: {:.5f} std: {:.5f}'.format(np.mean(decoder_acc), np.std(decoder_acc)))
        print('Decoder balanced accuracy: {:.5f} std: {:.5f}'.format(np.mean(decoder_ba), np.std(decoder_ba)))
        print('Decoder hamming: {:.5f} std: {:.5f}'.format(np.mean(decoder_hamming), np.std(decoder_hamming)))

        print('Encoder F1: {:.5f} std: {:.5f}'.format(np.mean(encoder_f1), np.std(encoder_f1)))
        print('Decoder F1: {:.5f} std: {:.5f}'.format(np.mean(decoder_f1), np.std(decoder_f1)))
        return encoder_causal_struct_estimate_lst, decoder_causal_struct_estimate_lst

    def generate_causal_graph(self, causal_matrix, filename, threshold=0.4, figsize=(10,8), 
                            positive_color='#2ecc71', negative_color='#e74c3c',
                            node_color='#3498db', show_labels=True,
                            title="Causal Graph with Absolute Threshold"):
        """
        生成优化后的一步时延因果图
        
        参数：
        causal_matrix   : list of lists 下三角权重矩阵
        threshold       : float 边创建阈值（绝对值）
        figsize         : tuple 图像尺寸
        positive_color  : str 正向影响边颜色
        negative_color  : str 负向影响边颜色
        node_color      : str 节点颜色（统一使用浅蓝色）
        show_labels     : bool 是否显示权重标签
        title          : str 图像标题
        """
        # 参数校验
        assert all(len(row) == len(causal_matrix) for row in causal_matrix), "必须为方阵"
        assert all(causal_matrix[i][j] == 0 for i in range(len(causal_matrix)) 
                  for j in range(i+1, len(causal_matrix))), "必须为下三角矩阵"

        N = len(causal_matrix)
        G = nx.DiGraph()

        # 找出有效节点（有边相连的节点）
        active_nodes = set()
        for target in range(N):
            for source in range(N):
                weight = causal_matrix[target][source]
                if abs(weight) > threshold:
                    active_nodes.add(source)
                    active_nodes.add(target)

        # 只添加有效节点
        t_minus_1_nodes = [f'X{i}_t-1' for i in active_nodes]
        t_nodes = [f'X{i}_t' for i in active_nodes]
        G.add_nodes_from(t_minus_1_nodes + t_nodes)

        # 添加边
        edges = []
        for target in active_nodes:
            for source in active_nodes:
                weight = causal_matrix[target][source]
                if abs(weight) > threshold:
                    edges.append((
                        f'X{source}_t-1',
                        f'X{target}_t',
                        {'weight': round(weight, 3)}
                    ))
        G.add_edges_from(edges)

        # 优化节点布局
        active_list = sorted(list(active_nodes))
        pos = {
            **{f'X{i}_t-1': (0, len(active_list)-1-active_list.index(i)) 
               for i in active_nodes},
            **{f'X{i}_t': (2, len(active_list)-1-active_list.index(i)) 
               for i in active_nodes}
        }

        # 绘图
        plt.figure(figsize=figsize, dpi=100)
        
        # 绘制节点
        nx.draw_networkx_nodes(
            G, pos,
            node_size=600,  # 增大节点大小以容纳文字
            node_color=[node_color] * (len(active_nodes) * 2),  # 统一使用浅蓝色
            edgecolors='white',  # 添加白色边框
            linewidths=1
        )
        
        # 绘制边
        edge_data = G.edges(data=True)
        edge_colors = []
        for u, v, d in edge_data:
            weight = d['weight']
            edge_colors.append(positive_color if weight > 0 else negative_color)
        
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edge_data,
            edge_color=edge_colors,
            width=1.5,  # 统一线条粗细
            arrowsize=10,  # 减小箭头大小
            alpha=0.7,
            min_source_margin=20,
            min_target_margin=20
        )
        
        # 添加标签
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='normal')  # 减小字体大小，使用普通字重
        if show_labels:
            edge_labels = {(u, v): f'{d["weight"]:+.2f}' for u, v, d in edge_data}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=10,  # 减小边标签字体大小
                label_pos=0.5,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )
        
        plt.title(title + f"\n(Threshold: |weight| > {threshold})", fontsize=12, pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()
        
        return G, plt.gcf()

    def make_lower_triangular(self, matrix):
        """
        将输入矩阵转换为下三角矩阵，保留原本下三角数据，上三角数据置0
        
        参数：
        matrix : numpy.ndarray 或 torch.Tensor 输入矩阵
        
        返回：
        lower_triangular_matrix : 与输入相同类型的下三角矩阵
        """
        # 复制输入矩阵以避免修改原始数据
        if isinstance(matrix, torch.Tensor):
            result = matrix.clone()
            # 将上三角部分（不包括对角线）置为0
            mask = torch.triu(torch.ones_like(result), diagonal=1)
            result = result * (1 - mask)
        else:
            result = np.copy(matrix)
            # 将上三角部分（不包括对角线）置为0
            result[np.triu_indices(result.shape[0], k=1)] = 0
            
        return result
