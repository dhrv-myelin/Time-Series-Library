from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment

# should remove
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score


import torch.multiprocessing
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")
warnings.filterwarnings("ignore")


class Exp_PoA_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_PoA_Detection, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

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
        criterion = nn.BCEWithLogitsLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_anomaly_loss = []
        total_poa_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, next_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device).unsqueeze(-1)
                next_y = next_y.float().to(self.device).unsqueeze(-1)

                anomaly_logits, poa_logits = self.model(batch_x, None, None, None, None)

                loss_anomaly = criterion(anomaly_logits, batch_y)
                loss_poa = criterion(poa_logits, next_y)
                total_anomaly_loss.append(loss_anomaly.item())
                total_poa_loss.append(loss_poa.item())
        total_anomaly_loss = np.average(total_anomaly_loss)
        total_poa_loss = np.average(total_poa_loss)
        self.model.train()
        return total_anomaly_loss, total_poa_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_anomaly_loss = []
            train_poa_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, next_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device).unsqueeze(-1)
                next_y = next_y.float().to(self.device).unsqueeze(-1)

                anomaly_logits, poa_logits = self.model(batch_x, None, None, None, None)

                loss_anomaly = criterion(anomaly_logits, batch_y)
                loss_poa = criterion(poa_logits, next_y)
                loss = loss_anomaly + loss_poa

                train_anomaly_loss.append(loss_anomaly.item())
                train_poa_loss.append(loss_poa.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} (anomaly: {3:.7f}, poa: {4:.7f})".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                            loss_anomaly.item(),
                            loss_poa.item(),
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_anomaly_loss = np.average(train_anomaly_loss)
            train_poa_loss = np.average(train_poa_loss)
            vali_anomaly_loss, vali_poa_loss = self.vali(
                vali_data, vali_loader, criterion
            )
            test_anomaly_loss, test_poa_loss = self.vali(
                test_data, test_loader, criterion
            )

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: [{2:.7f}, {3:.7f}] Vali Loss: [{4:.7f}, {5:.7f}] Test Loss: [{6:.7f}, {7:.7f}]".format(
                    epoch + 1,
                    train_steps,
                    train_anomaly_loss,
                    train_poa_loss,
                    vali_anomaly_loss,
                    vali_poa_loss,
                    test_anomaly_loss,
                    test_poa_loss,
                )
            )
            early_stopping(vali_poa_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        train_data, train_loader = self._get_data(flag="train")
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        attens_anomaly = []
        attens_poa = []
        test_labels = []
        next_labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y, next_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                anomaly_logits, poa_logits = self.model(batch_x, None, None, None, None)

                anomaly_score = torch.sigmoid(anomaly_logits).detach().cpu().numpy()
                poa_score = torch.sigmoid(poa_logits).detach().cpu().numpy()
                attens_anomaly.append(anomaly_score)
                attens_poa.append(poa_score)

        attens_anomaly = np.concatenate(attens_anomaly, axis=0).reshape(-1)
        train_anomaly_scores = np.array(attens_anomaly)
        attens_poa = np.concatenate(attens_poa, axis=0).reshape(-1)
        train_poa_scores = np.array(attens_poa)

        attens_anomaly = []
        attens_poa = []
        test_labels = []
        next_labels = []
        for i, (batch_x, batch_y, next_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device)
            anomaly_logits, poa_logits = self.model(batch_x, None, None, None, None)

            anomaly_score = torch.sigmoid(anomaly_logits).detach().cpu().numpy()
            poa_score = torch.sigmoid(poa_logits).detach().cpu().numpy()
            attens_anomaly.append(anomaly_score)
            attens_poa.append(poa_score)
            test_labels.append(batch_y)
            next_labels.append(next_y)

        attens_anomaly = np.concatenate(attens_anomaly, axis=0).reshape(-1)
        test_anomaly_scores = np.array(attens_anomaly)
        attens_poa = np.concatenate(attens_poa, axis=0).reshape(-1)
        test_poa_scores = np.array(attens_poa)

        combined_anomaly_scores = np.concatenate(
            [train_anomaly_scores, test_anomaly_scores], axis=0
        )
        combined_poa_scores = np.concatenate(
            [train_poa_scores, test_poa_scores], axis=0
        )

        threshold = np.percentile(
            combined_anomaly_scores, 100 - self.args.anomaly_ratio
        )
        print("Threshold (anomaly):", threshold)

        pred_anomaly = (test_anomaly_scores > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        next_labels = np.concatenate(next_labels, axis=0).reshape(-1)
        gt_anomaly = test_labels.astype(int)

        accuracy = accuracy_score(gt_anomaly, pred_anomaly)
        precision, recall, f_score, support = precision_recall_fscore_support(
            gt_anomaly, pred_anomaly, average="binary"
        )
        print("=== Anomaly Detection ===")
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
                accuracy, precision, recall, f_score
            )
        )

        pred_poa = (test_poa_scores > threshold).astype(int)
        gt_poa = next_labels.astype(int)

        accuracy_poa = accuracy_score(gt_poa, pred_poa)
        precision_poa, recall_poa, f_score_poa, support_poa = (
            precision_recall_fscore_support(gt_poa, pred_poa, average="binary")
        )
        print("=== Precursor-of-Anomaly Detection ===")
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}".format(
                accuracy_poa, precision_poa, recall_poa, f_score_poa
            )
        )

        f = open("result_precursor_anomaly_detection.txt", "a")
        f.write(setting + "  \n")
        f.write("=== Anomaly Detection ===\n")
        f.write(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision, recall, f_score
            )
        )
        f.write("\n")
        f.write("=== Precursor-of-Anomaly Detection ===\n")
        f.write(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy_poa, precision_poa, recall_poa, f_score_poa
            )
        )
        f.write("\n")
        f.write("\n")
        f.close()
        return

