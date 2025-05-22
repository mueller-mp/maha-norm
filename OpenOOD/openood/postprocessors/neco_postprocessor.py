from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict



class NecoPostprocessor(BasePostprocessor):
    def __init__(self, config):
        self.config = config
        self.num_classes = num_classes_dict[self.config.dataset.name]
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # estimate mean and variance from training set
            print('\n Estimating mean and variance from training set...')
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict['train'],
                                  desc='Setup: ',
                                  position=0,
                                  leave=True):
                    data, labels = batch['data'].cuda(), batch['label']
                    logits, features = net(data, return_feature=True)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats).cpu().numpy()
            all_labels = torch.cat(all_labels).cpu().numpy()
            all_preds = torch.cat(all_preds).cpu().numpy()
            # sanity check on train acc
            train_acc = np.mean(all_labels==all_preds)#all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            self.ss = StandardScaler()
            complete_vectors_train = self.ss.fit_transform(all_feats)
            print('Fitting PCA...')
            self.pca_estimator = PCA()
            self.pca_estimator.fit(complete_vectors_train)
            cumulative_variance = np.cumsum(self.pca_estimator.explained_variance_ratio_)

            # Find the number of components that explain at least 90% of the variance
            self.neco_dim = np.where(cumulative_variance >= 0.90)[0][0] + 1
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        logits, features = net(data, return_feature=True)
        features = features.cpu().numpy()
        pred = logits.argmax(1)
        cls_reduced_all = self.pca_estimator.transform(features)
        score_maxlogit = logits.cpu().numpy().max(axis=-1)
        cls_reduced = cls_reduced_all[:, :self.neco_dim]
        l=[]
        for i in range(cls_reduced.shape[0]):
            sc_complet = LA.norm(features[i, :])
            sc = LA.norm(cls_reduced[i, :])
            sc_finale = sc / sc_complet
            l.append(sc_finale)
        l = np.array(l)
        conf = torch.from_numpy(l* score_maxlogit)
        return pred, conf
