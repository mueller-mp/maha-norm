from typing import Any
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import sklearn.covariance
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .info import num_classes_dict


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def gmm_fit(embeddings, labels, num_classes,JITTERS):
    with torch.no_grad():
        classwise_mean_features=[]
        classwise_cov_features=[]
        for c in range(num_classes):
            mean_f = torch.mean(embeddings[labels == c], dim=0)
            f = embeddings[labels == c]
            classwise_mean_features.append(mean_f)
            classwise_cov_features.append(centered_cov_torch(f-mean_f))
    classwise_mean_features=torch.stack(classwise_mean_features)
    classwise_cov_features=torch.stack(classwise_cov_features)
    with torch.no_grad():
        jitter_eps=0
        attempts=1
        # while attempts < max_attempts:
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    classwise_cov_features.shape[1], device=classwise_cov_features.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),
                )
                return gmm, jitter_eps        
            except:
                continue

    raise ValueError(f'Matrix is not positive definite even after multiple attempts, jitter={jitter_eps}')

def get_lob_probs(features,gmm,batch_size=256):
    num_batches = (features.shape[0] + batch_size - 1) // batch_size  # To cover all elements in features_val
    print(num_batches)
    log_probs = []
    with torch.no_grad():
        for i in range(num_batches):
            batch = (features[i * batch_size: (i + 1) * batch_size])[:, None, :]
            log_probs_batch = gmm.log_prob(batch)
            log_probs.append(log_probs_batch)
    log_probs = torch.cat(log_probs, dim=0)
    return log_probs

class GMMNPostprocessor(BasePostprocessor):
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
            all_feats = all_feats/np.linalg.norm(all_feats,axis=-1,keepdims=True)
            all_labels = torch.cat(all_labels).cpu().numpy()
            all_preds = torch.cat(all_preds).cpu().numpy()
            # sanity check on train acc
            train_acc = np.mean(all_labels==all_preds)#all_preds.eq(all_labels).float().mean()
            print(f' Train acc: {train_acc:.2%}')

            DOUBLE_INFO = torch.finfo(torch.double)
            JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-15, 0, 1)]+[0.5,1.]

            gmm, jitter_eps = gmm_fit(torch.from_numpy(all_feats),all_labels,self.num_classes,JITTERS)
            print(f'Used jitter={jitter_eps}')
            self.gmm=gmm
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):

        logits, features = net(data, return_feature=True)
        features = features/torch.norm(features,dim=-1,keepdim=True)
        features = features.cpu()
        log_probs=get_lob_probs((features),self.gmm)

        pred = logits.argmax(1)

        conf = log_probs.max(-1)[0].cuda()
        return pred, conf
