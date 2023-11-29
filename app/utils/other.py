import torch
import wandb
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd


class DataParallel(torch.nn.DataParallel):
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(DataParallel, self).__init__(module, device_ids, output_device, dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Logger():
    def __init__(self, args,names):
        super(Logger, self).__init__()
        self.args  =args
        self.use_wandb = args.use_wandb
        self.show_step = args.show_step
        self.column_names = names

    def step(self, data, epoch, iteration, loader_len):
        if self.use_wandb:
            if data["Graph"]:
                wandb.log(data["Graph"])
            if data["Conf"]:
                for key in data["Conf"]:
                    data["Conf"][key] = wandb.plot.confusion_matrix(y_true=data["Conf"][key][0], preds=data["Conf"][key][1],
                                                                    class_names=data["Conf"][key][2])
                wandb.log(data["Conf"])
            if data["Table"]:
                for key in data["Table"]:
                    data["Table"][key] = wandb.Table(data = data["Table"][key][0],columns = data["Table"][key][1])
                wandb.log(data["Table"])
        if iteration % 100 == 0:
            print(f'epoch: {epoch}    {iteration} / {loader_len}')
            print(data["Print"])


class FindThreshold():
    def __init__(self, args,device):
        super(FindThreshold, self).__init__()
        self.y = torch.empty(0).to(device)
        self.pred = torch.empty(0).to(device)
        self.df = pd.DataFrame({"target":[]})


    def append_data(self,target,predicted):
        self.y = torch.cat([self.y,target])
        self.pred = torch.cat([self.pred, predicted])


    def Find_Optimal_Cutoff(self):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value

        """
        self.y[self.y>0] = 1
        target = pd.DataFrame(self.y.detach().cpu().numpy())
        predicted = pd.DataFrame(self.pred.detach().cpu().numpy())
        self.df["target"] = target
        self.df["predicted"] = predicted

        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

        return list(roc_t['threshold'])

    def Confusion_Matrix(self, threshold):
        # Find prediction to the dataframe applying threshold
        self.df['pred'] = self.df['predicted'].map(lambda x: 1 if x > threshold[0] else 0)

        # Print confusion Matrix
        conf = confusion_matrix(self.df['target'], self.df['pred'])

        return self.df["target"], self.df['pred'], conf
