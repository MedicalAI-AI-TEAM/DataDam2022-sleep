import os
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score, fbeta_score
from sklearn.preprocessing import label_binarize


class DataDamSolver(object):
    def __init__(self, config):
        self.config = config

        self.model = config["model"]
        self.optimizer = config["optimizer"]
        self.loss_function = config["loss_function"]

    def fit(self):
        for epoch in range(self.config["epochs"]):
            loss = self._training(
                loader=self.config["dataloader"]["train"], epoch=str(epoch)
            )
            self._test(loader=self.config["dataloader"]["valid"], epoch=str(epoch))
        self._test(loader=self.config["dataloader"]["test"], epoch="test")

        torch.save(
            {"config": self.config, "weight": self.model.state_dict()},
            os.path.join(self.config["save_path"], "checkpoint.pth"),
        )

    def feed_forward(self, ecg, y):
        ecg = ecg.cuda()
        y = y.cuda().long()
        prob = self.model(ecg)
        loss = self.loss_function(prob, y)

        return y, prob, loss

    def _training(self, loader, epoch):
        self.model.train()

        for _, (_, x, y) in enumerate(loader):
            y, prob, loss = self.feed_forward(x, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"epoch {epoch} => loss: {loss}")

        return loss

    def _test(self, loader, epoch):
        y_label, y_prob = [], []
        self.model.eval()
        with torch.no_grad():
            for i, (fname, x, y) in enumerate(loader):
                y, prob, loss = self.feed_forward(x, y)
                y_label.append(y.detach().cpu()) if y is not None else None
                y_prob.append(prob.detach().cpu()) if prob is not None else None
                loss = 0 if torch.isnan(loss) else loss.detach().cpu()

        auroc, f_beta = get_performance(y_label, y_prob, 2)
        print(f"epoch {epoch} performance => AUROC: {auroc}, F1_score:{f_beta}")
        return y_label, y_prob


def get_performance(y_true, y_prob, num_classes):

    y_true = np.concatenate(y_true, 0)
    y_prob = np.concatenate(y_prob, 0)
    if num_classes > 2:
        y_true_onehot = label_binarize(y_true, classes=range(num_classes))
        multi_class = "ovo"
    else:
        multi_class = None
        y_true_onehot = y_true
        if num_classes == 2:
            y_prob = y_prob[:, 1:]

    auroc = roc_auc_score(
        y_true_onehot, y_prob, multi_class=multi_class, average="weighted"
    )

    # jstat base
    cutoff = 0
    if num_classes <= 2:
        fpr, tpr, threshold_auroc = roc_curve(y_true, y_prob)
        cutoff = threshold_auroc[np.argmax(tpr - fpr)]
        y_pred = (y_prob > np.array(cutoff)) * 1
    else:
        y_pred = np.argmax(y_prob, -1)

    f_beta = fbeta_score(y_true, y_pred, beta=1, average="weighted")

    return auroc, f_beta
