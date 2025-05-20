import os
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from scipy.linalg import cholesky
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from numpy.linalg import norm, pinv
from tqdm import tqdm
import torch
import torch.nn.functional as F
from itertools import groupby
import faiss
from sklearn.neighbors import NearestNeighbors
import scipy
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import linalg as LA
import copy 

def get_cholesky(mat, epsilon=1e-10, max_attempts=10):
    attempts = 0
    while attempts < max_attempts:
        try:
            L = cholesky(mat, lower=True)
            return L
        except np.linalg.LinAlgError:
            print(f'Not PD, adding epsilon={epsilon:.1e} to the diagonal')
            mat = mat + epsilon * np.eye(mat.shape[0])
            epsilon *= 10
            attempts += 1
    
    raise ValueError('Matrix is not positive definite even after multiple attempts.')


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# much of the following is built upon code from https://github.com/haoqiwang/vim/blob/master/benchmark.py

def evaluate_MSP(softmax_id_val, softmax_ood):
    """
    Evaluate Maximum Softmax Probability (MSP) for given softmax values of in-distribution (id) and out-of-distribution (ood) data.

    Inputs:
        softmax_id_val: Numpy array of shape (m, n), representing the softmax probabilities of m data points from in-distribution.
        softmax_ood: Numpy array of shape (p, n), representing the softmax probabilities of p data points from out-of-distribution.
        
    Outputs:
        score_id: Numpy array of shape (m,), representing the maximum softmax probability of m data points from in-distribution.
        score_ood: Numpy array of shape (p,), representing the maximum softmax probability of p data points from out-of-distribution.
        """
    score_id = softmax_id_val.max(axis=-1)
    score_ood = softmax_ood.max(axis=-1)
    return score_id, score_ood


def evaluate_MaxLogit(logits_in_distribution, logits_out_of_distribution):
    """Compute the maximum logit value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the maximum logit value for both in- and out-of-distribution data.
    """
    score_in_distribution = logits_in_distribution.max(axis=-1)
    score_out_of_distribution = logits_out_of_distribution.max(axis=-1)
    return score_in_distribution, score_out_of_distribution


def evaluate_nan(features_in_distribution, features_out_of_distribution, use_pos=False):
    """Compute the maximum logit value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the maximum logit value for both in- and out-of-distribution data.
        
    """

    l1_norm_id = np.sum(np.abs(features_in_distribution), axis=1)
    l0_norm_id = np.count_nonzero(features_in_distribution, axis=1)
    n_pos_id = np.sum(features_in_distribution>0,axis=1)

    l1_norm_ood = np.sum(np.abs(features_out_of_distribution), axis=1)
    l0_norm_ood = np.count_nonzero(features_out_of_distribution, axis=1)
    n_pos_ood = np.sum(features_out_of_distribution>0,axis=1)

    score_in_distribution = (l1_norm_id/n_pos_id) if use_pos else (l1_norm_id/l0_norm_id)
    score_out_of_distribution = (l1_norm_ood/n_pos_ood) if use_pos else (l1_norm_ood/l0_norm_ood)
    return score_in_distribution, score_out_of_distribution


def evaluate_Energy(logits_in_distribution, logits_out_of_distribution):
    """Compute the energy value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the energy value for both in- and out-of-distribution data.
    """
    score_in_distribution = logsumexp(logits_in_distribution, axis=1)
    score_out_of_distribution = logsumexp(logits_out_of_distribution, axis=1)
    return score_in_distribution, score_out_of_distribution

def softmax_scaled_cosine(features, w,scale=1):
    out = torch.from_numpy(features).float()
    x_norm = F.normalize(out)
    w_norm = F.normalize(w)
    w_norm_transposed = torch.transpose(w_norm, 0, 1)
    
    # cos_sim = torch.mm(out,w.T)+b#
    cos_sim = torch.mm(x_norm, w_norm_transposed) # cos_theta
    scaled_cosine = cos_sim * scale
    softmax = F.softmax(scaled_cosine, -1)
    return softmax.numpy(), scaled_cosine.numpy()

def evaluate_softmax_scaled_cosine(feature_id_val, feature_ood, w, s=1):
    with torch.no_grad():
        softmax_sc_val, cos_val = softmax_scaled_cosine(feature_id_val,torch.from_numpy(w),scale=s)
        softmax_sc_ninco, cos_ninco = softmax_scaled_cosine(feature_ood,torch.from_numpy(w),scale=s)
    return softmax_sc_val.max(-1),softmax_sc_ninco.max(-1)

def evaluate_ViM(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, u, path):
    """
    This function evaluates the performance of the ViM out-of-distribution detection method.

    Inputs:

        feature_id_train: numpy array of shape (n, d), the training set features for the in-distribution data.
        feature_id_val: numpy array of shape (m, d), the validation set features for the in-distribution data.
        feature_ood: numpy array of shape (p, d), the features for the out-of-distribution data.
        logits_id_train: numpy array of shape (n, k), the logits for the in-distribution training set.
        logits_id_val: numpy array of shape (m, k), the logits for the in-distribution validation set.
        logits_ood: numpy array of shape (p, k), the logits for the out-of-distribution data.
        u: numpy array of shape (d,), the mean feature vector.
        path: string, the path to store intermediate results.

    Outputs:

        score_id: numpy array of shape (m,), the ViM scores for the in-distribution validation set.
        score_ood: numpy array of shape (p,), the ViM scores for the out-of-distribution data.
        """
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else (
        512 if feature_id_val.shape[-1] >= 768 else int(feature_id_val.shape[-1] / 2))
    print(f'{DIM=}')
    print('Reading alpha and NS')
    alpha_path = os.path.join(path, 'alpha.npy')
    NS_path = os.path.join(path, 'NS.npy')
    if os.path.exists(NS_path):
        NS = np.load(NS_path)
    else:
        print('NS not stored, computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        np.save(NS_path, NS)
    if os.path.exists(alpha_path):
        alpha = np.load(alpha_path)
    else:
        print('alpha not stored, computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
        alpha = logits_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        np.save(alpha_path, alpha)
    print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logits_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    energy_ood = logsumexp(logits_ood, axis=-1)
    vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
    score_ood = -vlogit_ood + energy_ood
    return score_id, score_ood


def evaluate_gen(sf_id, sf_ood, gamma=0.1, M=100):
    def generalized_entropy(softmax_id_val, gamma, M):
            probs =  softmax_id_val 
            probs_sorted = np.sort(probs, axis=1)[:,-M:]
            scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
            return -scores 
    score_id = generalized_entropy(sf_id, gamma=gamma, M=M)
    score_ood = generalized_entropy(sf_ood, gamma=gamma, M=M)

    return score_id,score_ood

def evaluate_fdbd(features_train, features_val, features_ninco,logits_val,logits_ninco,w,):
    device='cuda:0'
    
    feat_log_train=torch.from_numpy(features_train).to(device)
    feat_log_val=torch.from_numpy(features_val).to(device)
    feat_log=torch.from_numpy(features_ninco).to(device)
    score_log_val=torch.from_numpy(logits_val).to(device)
    score_log=torch.from_numpy(logits_ninco).to(device)
    w=torch.from_numpy(w)

    train_mean = torch.mean(feat_log_train, dim= 0).to(device)

    denominator_matrix = torch.zeros((1000,1000)).to(device)
    for p in range(1000):
        w_p = w - w[p,:]
        denominator = torch.norm(w_p, dim=1)
        denominator[p] = 1
        denominator_matrix[p, :] = denominator

    #################### fDBD score OOD detection #################

    values, nn_idx = score_log_val.max(1)
    logits_sub = torch.abs(score_log_val - values.repeat(1000, 1).T)
    #pdb.set_trace()
    score_in = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log_val - train_mean , dim = 1)
    score_in = score_in.float().cpu().numpy()

    values, nn_idx = score_log.max(1)
    logits_sub = torch.abs(score_log - values.repeat(1000, 1).T)
    scores_out_test = torch.sum(logits_sub/denominator_matrix[nn_idx], axis=1)/torch.norm(feat_log - train_mean , dim = 1)
    scores_out_test = scores_out_test.float().cpu().numpy()

    return score_in, scores_out_test

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
            if i%50==0:
                print(f"{i}/{num_batches} batches processed.")
    log_probs = torch.cat(log_probs, dim=0)
    return log_probs

def evaluate_gmm(feature_id_train, feature_id_val, feature_ood,train_labels,path,normalize=False):
    DOUBLE_INFO = torch.finfo(torch.double)
    JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-15, 0, 1)]
    normalizer = lambda x: x/np.linalg.norm(x,axis=-1,keepdims=True)

    f_val = normalizer(feature_id_val) if normalize else feature_id_val
    f_ood = normalizer(feature_ood) if normalize else feature_ood

    gmm_path = os.path.join(path, f'gmm{"_norm" if normalize else ""}.pkl')
    if os.path.exists(gmm_path):
        print('Loading GMM')
        with open(gmm_path, 'rb') as file:
            gmm = pickle.load(file)
    else:    
        print('Fitting GMM...')
        f_train = normalizer(feature_id_train) if normalize else feature_id_train
        gmm, jitter_eps = gmm_fit(torch.from_numpy(f_train),train_labels,1000,JITTERS)
        print(f"Done. Used jitter={jitter_eps}")

        gmmPickle = open(gmm_path, 'wb') 
        pickle.dump(gmm, gmmPickle)  
        gmmPickle.close()


    score_id_gmm_path = os.path.join(path, f'scores_id_gmm{"_norm" if normalize else ""}.npy')
    if os.path.exists(score_id_gmm_path):
        print('Loading ID scores')
        score_id = np.load(score_id_gmm_path)
    else:
        log_probs_val=get_lob_probs(torch.from_numpy(f_val),gmm)
        score_id=log_probs_val.max(-1)[0].numpy()
        np.save(score_id_gmm_path, score_id)

    log_probs_ood=get_lob_probs(torch.from_numpy(f_ood),gmm)
    score_ood = log_probs_ood.max(-1)[0].numpy()

    return score_id, score_ood

def evaluate_neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood, path,use_logit=True):
    '''
    Prints the auc/fpr result for NECO method, with adaptive selection of neco_dim to explain 90% of the variance.

            Parameters:
                    feature_id_train (array): An array of training samples features
                    feature_id_val (array): An array of evaluation samples features
                    feature_ood (array): An array of OOD samples features
                    logit_id_val (array): An array of evaluation samples logits
                    logit_ood (array): An array of OOD samples logits
                    model_architecture_type (string): Module architecture used

            Returns:
                    score_id (array): Scores for in-distribution samples
                    score_ood (array): Scores for out-of-distribution samples
    '''
    ss_path = os.path.join(path, 'ss_neco.pkl')
    if os.path.exists(ss_path):
        with open(ss_path, 'rb') as file:
            ss = pickle.load(file)
            complete_vectors_train = ss.transform(feature_id_train)
    else:
        ss = StandardScaler()
        complete_vectors_train = ss.fit_transform(feature_id_train)

        ssPickle = open(ss_path, 'wb') 
        pickle.dump(ss, ssPickle)  
        ssPickle.close()
        
    complete_vectors_test = ss.transform(feature_id_val)
    complete_vectors_ood = ss.transform(feature_ood)

    # Fit PCA to the training data
    pca_path = os.path.join(path, 'pca_neco.pkl')
    if os.path.exists(pca_path):
        print('Loading PCA...')
        with open(pca_path, 'rb') as file:
            pca_estimator = pickle.load(file)
    else:
        print('Fitting PCA...')
        print(complete_vectors_train.shape)
        print(complete_vectors_ood.shape)
        print(complete_vectors_test.shape)
        pca_estimator = PCA()
        pca_estimator.fit(complete_vectors_train)

        pcaPickle = open(pca_path, 'wb') 
        pickle.dump(pca_estimator, pcaPickle)  
        pcaPickle.close()

    # Calculate the cumulative explained variance ratio
    cumulative_variance = np.cumsum(pca_estimator.explained_variance_ratio_)

    # Find the number of components that explain at least 90% of the variance
    neco_dim = np.where(cumulative_variance >= 0.90)[0][0] + 1

    # Perform PCA transformation
    cls_test_reduced_all = pca_estimator.transform(complete_vectors_test)
    cls_ood_reduced_all = pca_estimator.transform(complete_vectors_ood)

    score_id_maxlogit = logit_id_val.max(axis=-1)
    score_ood_maxlogit = logit_ood.max(axis=-1)

    # # If model architecture is one of 'deit' or 'swin', skip PCA
    # if model_architecture_type in ['deit', 'swin']:
    #     complete_vectors_train = feature_id_train
    #     complete_vectors_test = feature_id_val
    #     complete_vectors_ood = feature_ood

    # Select the reduced dimensions for in-distribution and OOD data
    cls_test_reduced = cls_test_reduced_all[:, :neco_dim]
    cls_ood_reduced = cls_ood_reduced_all[:, :neco_dim]

    l_ID = []
    l_OOD = []
    print('Computing scores...')
    # Compute ID scores based on the norm of reduced vectors
    for i in range(cls_test_reduced.shape[0]):
        sc_complet = LA.norm(complete_vectors_test[i, :])
        sc = LA.norm(cls_test_reduced[i, :])
        sc_finale = sc / sc_complet
        l_ID.append(sc_finale)

    # Compute OOD scores based on the norm of reduced vectors
    for i in range(cls_ood_reduced.shape[0]):
        sc_complet = LA.norm(complete_vectors_ood[i, :])
        sc = LA.norm(cls_ood_reduced[i, :])
        sc_finale = sc / sc_complet
        l_OOD.append(sc_finale)

    l_OOD = np.array(l_OOD)
    l_ID = np.array(l_ID)

    # Return the final scores
    score_id = l_ID* score_id_maxlogit if use_logit else l_ID
    score_ood = l_OOD*score_ood_maxlogit if use_logit else l_OOD

    return score_id, score_ood


def evaluate_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
    This function computes Mahalanobis scores for in-distribution and out-of-distribution samples.

    Parameters:
    feature_id_train (numpy array): The in-distribution training samples.
    feature_id_val (numpy array): The in-distribution validation samples.
    feature_ood (numpy array): The out-of-distribution samples.
    train_labels (numpy array): The labels of the in-distribution training samples.
    path (str): The path to save and load the mean and precision matrix.

    Returns:
    tuple: The Mahalanobis scores for in-distribution validation and out-of-distribution samples.

    """
    feature_id_val = feature_id_val/np.linalg.norm(feature_id_val,axis=-1,keepdims=True)
    feature_ood = feature_ood/np.linalg.norm(feature_ood,axis=-1,keepdims=True)
    # load mean and prec
    mean_path = os.path.join(path, 'mean_norm.npy')
    prec_path = os.path.join(path, 'prec_norm.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        feature_id_train = feature_id_train/np.linalg.norm(feature_id_train,axis=-1,keepdims=True)
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    print('go to gpu...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()
    print('Computing scores...')
    score_id_path = os.path.join(path, 'maha_id_scores_norm.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
        print('Loaded ID scores.')
    else:
        score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                              tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)
        print('Computed ID scores.')
    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                           tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood

def evaluate_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
    This function computes Mahalanobis scores for in-distribution and out-of-distribution samples.

    Parameters:
    feature_id_train (numpy array): The in-distribution training samples.
    feature_id_val (numpy array): The in-distribution validation samples.
    feature_ood (numpy array): The out-of-distribution samples.
    train_labels (numpy array): The labels of the in-distribution training samples.
    path (str): The path to save and load the mean and precision matrix.

    Returns:
    tuple: The Mahalanobis scores for in-distribution validation and out-of-distribution samples.

    """
    # load mean and prec
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    print('go to gpu...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()
    print('Computing scores...')
    score_id_path = os.path.join(path, 'maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                              tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)

    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                           tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood

def evaluate_Relative_Mahalanobis_norm(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
      This function computes the relative Mahalanobis scores for in-distribution and out-of-distribution samples.

      Parameters:
      feature_id_train (numpy array): The in-distribution training samples.
      feature_id_val (numpy array): The in-distribution validation samples.
      feature_ood (numpy array): The out-of-distribution samples.
      train_labels (numpy array): The labels of the in-distribution training samples.
      path (str): The path to save and load the mean and precision matrix.

      Returns:
      tuple: The relative Mahalanobis scores for in-distribution validation and out-of-distribution samples.

      Steps:
      - Load class-wise mean and precision from disk if they exist, otherwise compute them from the ID training samples and save to disk.
      - Load global mean and precision from disk if they exist, otherwise compute them from all the ID training samples and save to disk.
      - Compute the relative Mahalanobis scores for ID validation samples and save to disk if they don't exist.
      - Compute the relative Mahalanobis scores for OOD samples and save to disk.

      """
    feature_id_val = feature_id_val/np.linalg.norm(feature_id_val,axis=-1,keepdims=True)
    feature_ood = feature_ood/np.linalg.norm(feature_ood,axis=-1,keepdims=True)
    # load class-wise mean and prec
    mean_path = os.path.join(path, 'mean_norm.npy')
    prec_path = os.path.join(path, 'prec_norm.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        feature_id_train = feature_id_train/np.linalg.norm(feature_id_train,axis=-1,keepdims=True)
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    # print('go to gpu with class-wise...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()

    # load global mean and prec - stay on cpu and use numpy for better precision
    mean_path_global = os.path.join(path, 'mean-global_norm.npy')
    prec_path_global = os.path.join(path, 'prec-global_norm.npy')
    complete = True
    if os.path.exists(mean_path_global):
        mean_global = np.load(mean_path_global)
    else:
        complete = False
    if os.path.exists(prec_path_global):
        prec_global = np.load(prec_path_global)
    else:
        complete = False
    if not complete:
        print('not complete, computing global mean feature...')
        train_means_global = []
        train_feat_centered_global = []

        _m_global = feature_id_train.mean(axis=0)
        train_means_global.append(_m_global)
        train_feat_centered_global.extend(feature_id_train - _m_global)

        print('computing precision matrix...')
        ec_global = EmpiricalCovariance(assume_centered=True)
        ec_global.fit(np.array(train_feat_centered_global).astype(np.float64))

        mean_global = np.array(train_means_global)
        prec_global = (ec_global.precision_)
        np.save(mean_path_global, mean_global)
        np.save(prec_path_global, prec_global)

    print('Computing scores...')
    score_id_path = os.path.join(path, 'rel_maha_id_scores_norm.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id_path_classwise = os.path.join(path, 'maha_id_scores_norm.npy')
        if os.path.exists(score_id_path_classwise):
            score_id_classwise = np.load(score_id_path_classwise)
        else:
            score_id_classwise = -np.array(
                [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                 tqdm(torch.from_numpy(feature_id_val).cuda().double())])
            np.save(score_id_path_classwise, score_id_classwise)
        #
        score_id_global = -np.array(
            [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in
             tqdm((feature_id_val))])  # tqdm(torch.from_numpy(feature_id_val).cuda().float())])

        score_id = score_id_classwise - score_id_global
        np.save(score_id_path, score_id)

    score_ood_classwise = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                                     tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_global = -np.array(
        [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in tqdm(feature_ood)])
    score_ood = score_ood_classwise - score_ood_global
    return score_id, score_ood

def evaluate_Relative_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
      This function computes the relative Mahalanobis scores for in-distribution and out-of-distribution samples.

      Parameters:
      feature_id_train (numpy array): The in-distribution training samples.
      feature_id_val (numpy array): The in-distribution validation samples.
      feature_ood (numpy array): The out-of-distribution samples.
      train_labels (numpy array): The labels of the in-distribution training samples.
      path (str): The path to save and load the mean and precision matrix.

      Returns:
      tuple: The relative Mahalanobis scores for in-distribution validation and out-of-distribution samples.

      Steps:
      - Load class-wise mean and precision from disk if they exist, otherwise compute them from the ID training samples and save to disk.
      - Load global mean and precision from disk if they exist, otherwise compute them from all the ID training samples and save to disk.
      - Compute the relative Mahalanobis scores for ID validation samples and save to disk if they don't exist.
      - Compute the relative Mahalanobis scores for OOD samples and save to disk.

      """
    # load class-wise mean and prec
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    # print('go to gpu with class-wise...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()

    # load global mean and prec - stay on cpu and use numpy for better precision
    mean_path_global = os.path.join(path, 'mean-global.npy')
    prec_path_global = os.path.join(path, 'prec-global.npy')
    complete = True
    if os.path.exists(mean_path_global):
        mean_global = np.load(mean_path_global)
    else:
        complete = False
    if os.path.exists(prec_path_global):
        prec_global = np.load(prec_path_global)
    else:
        complete = False
    if not complete:
        print('not complete, computing global mean feature...')
        train_means_global = []
        train_feat_centered_global = []

        _m_global = feature_id_train.mean(axis=0)
        train_means_global.append(_m_global)
        train_feat_centered_global.extend(feature_id_train - _m_global)

        print('computing precision matrix...')
        ec_global = EmpiricalCovariance(assume_centered=True)
        ec_global.fit(np.array(train_feat_centered_global).astype(np.float64))

        mean_global = np.array(train_means_global)
        prec_global = (ec_global.precision_)
        np.save(mean_path_global, mean_global)
        np.save(prec_path_global, prec_global)

    print('Computing scores...')
    score_id_path = os.path.join(path, 'rel_maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id_path_classwise = os.path.join(path, 'maha_id_scores.npy')
        if os.path.exists(score_id_path_classwise):
            score_id_classwise = np.load(score_id_path_classwise)
        else:
            score_id_classwise = -np.array(
                [(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                 tqdm(torch.from_numpy(feature_id_val).cuda().double())])
            np.save(score_id_path_classwise, score_id_classwise)
        #
        score_id_global = -np.array(
            [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in
             tqdm((feature_id_val))])  # tqdm(torch.from_numpy(feature_id_val).cuda().float())])

        score_id = score_id_classwise - score_id_global
        np.save(score_id_path, score_id)

    score_ood_classwise = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis=-1).min().cpu().item() for f in
                                     tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_global = -np.array(
        [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis=-1)).item() for f in tqdm(feature_ood)])
    score_ood = score_ood_classwise - score_ood_global
    return score_id, score_ood

def evaluate_KL_Matching(softmax_id_train, softmax_id_val, softmax_ood, path):
    """
    Evaluate KL Matching between softmax output of trained classifier and validation/out-of-distribution data.

    Inputs:
    softmax_id_train (ndarray): Softmax output of classifier on training data. Shape: (num_training_samples, num_classes)
    softmax_id_val (ndarray): Softmax output of classifier on validation data. Shape: (num_validation_samples, num_classes)
    softmax_ood (ndarray): Softmax output of classifier on out-of-distribution data. Shape: (num_ood_samples, num_classes)
    path (str): Path to directory where mean_softmax_train.npy and score_id_KL.npy should be stored/loaded from

    Outputs:
    score_id (ndarray): KL Matching score between softmax_id_val and mean_softmax_train. Shape: (num_validation_samples,)
    score_ood (ndarray): KL Matching score between softmax_ood and mean_softmax_train. Shape: (num_ood_samples,)
    """
    mean_softmax_train_path = os.path.join(path, 'mean_softmax_train.npy')
    score_id_KL_path = os.path.join(path, 'score_id_KL.npy')
    if os.path.exists(mean_softmax_train_path):
        mean_softmax_train = np.load(mean_softmax_train_path)
    else:
        print('not complete, computing classwise mean softmax...')
        pred_labels_train = np.argmax(softmax_id_train, axis=-1)
        mean_softmax_train = np.array(
            [softmax_id_train[pred_labels_train == i].mean(axis=0) for i in tqdm(range(1000))])
        np.save(mean_softmax_train_path, mean_softmax_train)
    if os.path.exists(score_id_KL_path):
        score_id = np.load(score_id_KL_path)
    else:
        print('not complete, Computing id score...')
        score_id = -pairwise_distances_argmin_min(softmax_id_val, (mean_softmax_train), metric=kl)[1]
        print('score_id is nan: ', np.isnan(score_id).any())
        np.save(score_id_KL_path, score_id)
    print('Computing OOD score...')
    score_ood = -pairwise_distances_argmin_min(softmax_ood, (mean_softmax_train), metric=kl)[1]
    return score_id, score_ood


def evaluate_Energy_React(feature_id_train, feature_id_val, feature_ood, w, b, path, clip_quantile=0.99):
    """Evaluate Energy React Score

       The function evaluates Energy React Score by computing score_id and score_ood.

       Parameters
       ----------
       feature_id_train: np.ndarray
           Input features of the training set.
       feature_id_val: np.ndarray
           Input features of the validation set.
       feature_ood: np.ndarray
           Input features of the out-of-distribution set.
       w: np.ndarray
           Weight matrix of classifiers last layer
       b: np.ndarray
           Bias vector of classifiers last layer
       path: str
           Path to store intermediate values.
       clip_quantile: float, optional, default 0.99
           Quantile used for clipping the input features.

       Returns
       -------
       score_id: np.ndarray
           Energy Reactivity Score for validation set.
       score_ood: np.ndarray
           Energy Reactivity Score for out-of-distribution set.
       """
    clip_react_path = os.path.join(path, 'clip_react.npy')
    if os.path.exists(clip_react_path):
        clip = np.load(clip_react_path)
    else:
        clip = np.quantile(feature_id_train, clip_quantile)
        np.save(clip_react_path, clip)
    print(f'clip quantile {clip_quantile}, clip {clip:.4f}')
    score_id_energy_react_path = os.path.join(path, 'score_id_energy_react.npy')
    if os.path.exists(score_id_energy_react_path):
        score_id = np.load(score_id_energy_react_path)
    else:
        print('not complete, Computing id score...')
        logit_id_val_clip = np.clip(feature_id_val, a_min=None, a_max=clip) @ w.T + b
        score_id = logsumexp(logit_id_val_clip, axis=-1)
        np.save(score_id_energy_react_path, score_id)
    logit_ood_clip = np.clip(feature_ood, a_min=None, a_max=clip) @ w.T + b
    score_ood = logsumexp(logit_ood_clip, axis=-1)
    return score_id, score_ood


def evaluate_KNN(feature_id_train, feature_id_val, feature_ood, path):
    """
    Evaluate KNN classification for in-distribution (ID) and out-of-distribution (OOD) samples.

    This function computes KNN scores for ID and OOD samples. The KNN scores are computed as the distance to the K nearest neighbour of the ID samples in a preprocessed feature space.

    Args:
    feature_id_train_prepos (numpy.ndarray): Preprocessed features of ID training samples.
    feature_id_val (numpy.ndarray): Features of ID validation samples.
    feature_ood (numpy.ndarray): Features of OOD samples.
    path (str): File path to save intermediate computations.

    Returns:
    Tuple of numpy.ndarray:
    score_id (numpy.ndarray): KNN scores of ID validation samples.
    score_ood (numpy.ndarray): KNN scores of OOD samples.

    """
    normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

    scores_id_path_knn = os.path.join(path, 'scores_id_knn.npy')
    index_path = os.path.join(path, 'trained.index')

    # compute neighbours
    K = 1000
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print('Index not stored, creating index...')
        feature_id_train_prepos = prepos_feat(feature_id_train)
        index = faiss.IndexFlatL2(feature_id_train_prepos.shape[1])
        index.add(feature_id_train_prepos)
        faiss.write_index(index, index_path)

    if os.path.exists(scores_id_path_knn):
        score_id = np.load(scores_id_path_knn)
    else:
        print('Computing id knn scores...')
        ftest = prepos_feat(feature_id_val).astype(np.float32)
        D, _ = index.search(ftest, K, )
        score_id = -D[:, -1]
        np.save(scores_id_path_knn, score_id)

    print('Computing ood knn scores...')
    food = prepos_feat(feature_ood)
    D, _ = index.search(food, K)
    score_ood = -D[:, -1]
    return score_id, score_ood

def evaluate_nnguide(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, path, alpha=0.01, K=1000):
    normalizer = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True) + 1e-10
    scores_id_path_knn = os.path.join(path, 'scores_id_nnguide.npy')
    index_path = os.path.join(path, 'trained_nnguide.index')

    # compute neighbours
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print('Index not stored, creating index...')
        n_bank_samples = int(len(feature_id_train)*alpha)
        idxs=np.random.choice(range(0,len(feature_id_train)),n_bank_samples,replace=False)
        bank_feas = normalizer(feature_id_train[idxs,:])
        bank_confs = logsumexp(logits_id_train[idxs,:],axis=-1)
        bank_guide = bank_feas * bank_confs[:, None]
        index = faiss.IndexFlatIP(bank_guide.shape[-1])
        index.add(bank_guide)
        faiss.write_index(index, index_path)

    if False: #os.path.exists(scores_id_path_knn): 
        score_id = np.load(scores_id_path_knn)
    else:
        print('Computing id knn scores...')
        feas_val_norm = normalizer(feature_id_val)
        energy_val = logsumexp(logits_id_val,axis=-1)
        D, _ = index.search(feas_val_norm, K)
        conf_val = np.array(D.mean(axis=1))
        score_id = conf_val * energy_val
        # np.save(scores_id_path_knn,score_id)

    feas_ood_norm = normalizer(feature_ood)
    energy_ood = logsumexp(logits_ood,axis=-1)
    D, _ = index.search(feas_ood_norm, K)
    conf_ood = np.array(D.mean(axis=1))
    score_ood = conf_ood * energy_ood

    return score_id, score_ood    

def evaluate_she(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, labels_train, path, metric='inner_product'):
        
    activation_log_path = os.path.join(path, 'activation_log.pt')
    if os.path.exists(activation_log_path):
        activation_log = torch.load(activation_log_path)
    else:
        train_preds=torch.max(torch.from_numpy(logits_id_train),-1)[1]
        all_activation_log = torch.from_numpy(feature_id_train)
        activation_log = []
        for i in range(1000):
            mask = torch.logical_and(torch.from_numpy(labels_train) == i, train_preds == i)
            class_correct_activations = all_activation_log[mask]
            activation_log.append(
                class_correct_activations.mean(0, keepdim=True))

        activation_log = torch.cat(activation_log)
        torch.save(activation_log, activation_log_path)

    def distance(penultimate, target, metric='inner_product'):
        if metric == 'inner_product':
            return torch.sum(torch.mul(penultimate, target), dim=1)
        elif metric == 'euclidean':
            return -torch.sqrt(torch.sum((penultimate - target)**2, dim=1))
        elif metric == 'cosine':
            return torch.cosine_similarity(penultimate, target, dim=1)
        else:
            raise ValueError('Unknown metric: {}'.format(metric))
    pred_val=torch.max(torch.from_numpy(logits_id_val),-1)[1]
    score_id = distance(feature_id_val, activation_log[pred_val], metric).numpy()

    pred_ood=torch.max(torch.from_numpy(logits_ood),-1)[1]
    score_ood = distance(feature_ood, activation_log[pred_ood], metric).numpy()

    return score_id, score_ood

def evaluate_scale(feature_id_val, feature_ood,  w, b, percentile=85):
    def scale(x, percentile=85):
        input_ = x.clone()
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2

        return input_ * torch.exp(scale[:, None, None, None])

    def eval_scale(feature,w,b,percentile=65):
        feature=torch.from_numpy(feature.copy())
        feature = scale(feature.view(feature.size(0), -1, 1, 1), percentile)
        feature = feature.view(feature.size(0), -1)
        logits_cls = (feature@w.T) + b
        return torch.logsumexp(logits_cls.data.cpu(), dim=1).numpy()
    
    s_id = eval_scale(feature_id_val,w,b,percentile=percentile)
    s_ood = eval_scale(feature_ood,w,b,percentile=percentile)

    return s_id, s_ood

def evaluate_ash_s(feature_id_val, feature_ood,  w, b, percentile=90):
    def ash_s(x, percentile=percentile):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])
        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        t.zero_().scatter_(dim=1, index=i, src=v)

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1, 2, 3])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None, None, None])

        return x

    def ash(f,w,b):
        with torch.no_grad():
            to_forward = torch.from_numpy(f.copy())
            feature = ash_s(to_forward.view(to_forward.size(0), -1, 1, 1))
            feature = feature.view(feature.size(0), -1)
            output=feature@ w.T + b
            print(output.shape)
            energyconf = torch.logsumexp(output.data.cpu(), dim=-1)
            return energyconf.numpy()
    s_id = ash(feature_id_val,w,b)
    s_ood = ash(feature_ood,w,b)
    return s_id, s_ood

def evaluate_ash_b(feature_id_val, feature_ood,  w, b, percentile=65):
    def ash_b(x, percentile=percentile):
        assert x.dim() == 4
        assert 0 <= percentile <= 100
        b, c, h, w = x.shape

        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1, 2, 3])

        n = x.shape[1:].numel()
        k = n - int(np.round(n * percentile / 100.0))
        t = x.view((b, c * h * w))
        v, i = torch.topk(t, k, dim=1)
        fill = s1 / k
        fill = fill.unsqueeze(dim=1).expand(v.shape)
        t.zero_().scatter_(dim=1, index=i, src=fill)
        return x

    def ash(f,w,b):
        with torch.no_grad():
            to_forward = torch.from_numpy(f.copy())
            feature = ash_b(to_forward.view(to_forward.size(0), -1, 1, 1))
            feature = feature.view(feature.size(0), -1)
            output=feature@ w.T + b
            print(output.shape)
            energyconf = torch.logsumexp(output.data.cpu(), dim=-1)
            return energyconf.numpy()
    s_id = ash(feature_id_val,w,b)
    s_ood = ash(feature_ood,w,b)
    return s_id, s_ood

def evaluate_cosine(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    '''
    Like Cosine for CLIP, but with class-wise mean-features as encoded text:

    This function loads the mean of the in-distribution features, or computes and saves it if not found,
    and computes the cosine similarity scores between the in-distribution and out-of-distribution inputs and the mean.

    Parameters:
    feature_id_train (np.array): In-distribution training features.
    feature_id_val (np.array): In-distribution validation features.
    feature_ood (np.array): Out-of-distribution features.
    train_labels (np.array): Labels for in-distribution training data.
    path (str): Path to save and load mean.

    Returns:
    tuple: In-distribution and out-of-distribution cosine similarity scores.
    '''
    # load mean
    mean_path = os.path.join(path, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        mean = np.array(train_means)
        np.save(mean_path, mean)
    means_n = np.array([m / np.linalg.norm(m) for m in mean])
    features_id_normalized = np.array([m / np.linalg.norm(m) for m in feature_id_val])
    score_id = (features_id_normalized @ means_n.T).max(axis=-1)
    features_ood_normalized = np.array([m / np.linalg.norm(m) for m in feature_ood])
    score_ood = (features_ood_normalized @ means_n.T).max(axis=-1)
    return score_id, score_ood


def evaluate_rcos(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    '''
    Like MCM, but with class-wise mean-features as encoded text:
    This function loads the mean of the in-distribution data, or computes and saves it if not found,
    and computes the softmax of the cosine similarity scores between the in-distribution and out-of-distribution
    inputs and the mean.

    Parameters:
    feature_id_train (np.array): In-distribution training features.
    feature_id_val (np.array): In-distribution validation features.
    feature_ood (np.array): Out-of-distribution features.
    train_labels (np.array): Labels for in-distribution training data.
    path (str): Path to save and load mean.

    Returns:
    tuple: In-distribution and out-of-distribution re-scaled cosine similarity scores.
    '''
    T = 1.
    mean_path = os.path.join(path, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis=0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        mean = np.array(train_means)
        np.save(mean_path, mean)

    # use train means as encoded text pairs
    text_encoded = torch.from_numpy(mean).float()
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    scores_id_path_clip = os.path.join(path, 'mcm_scores_id.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
    else:
        print('Computing ID scores...')

        features_id = torch.from_numpy(feature_id_val).float()
        features_id /= features_id.norm(dim=-1, keepdim=True)

        out_id = features_id @ text_encoded.T
        smax_id = F.softmax(out_id / T, dim=1).data.cpu().numpy()
        score_id = np.max(smax_id, axis=1)

        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')

    features_ood = torch.from_numpy(feature_ood).float()
    features_ood /= features_ood.norm(dim=-1, keepdim=True)

    out_ood = features_ood @ text_encoded.T
    smax_ood = F.softmax(out_ood / T, dim=1).data.cpu().numpy()
    score_ood = np.max(smax_ood, axis=1)
    return score_id, score_ood


def evaluate_cosine_clip(feature_id_val, feature_ood, clip_labels, labels_encoded_clip, path):
    """
       Evaluates cosine similarity scores for in-distribution and out-of-distribution samples and returns the scores,
       along with the in-distribution accuracy for CLIP features.

       Parameters:
       feature_id_val (np.ndarray): In-distribution validation feature tensor.
       feature_ood (np.ndarray): Out-of-distribution feature tensor.
       clip_labels (np.ndarray): Ground truth labels for the in-distribution validation samples.
       labels_encoded_clip (np.ndarray): Encoded ground truth labels for the in-distribution samples.
       path (str): Path to the directory to save the scores.

       Returns:
       tuple:
           score_id (np.ndarray): Cosine similarity scores for the in-distribution samples.
           score_ood (np.ndarray): Cosine similarity scores for the out-of-distribution samples.
           val_acc (float): In-distribution accuracy.
       """
    text_encoded = np.array([m / np.linalg.norm(m) for m in
                             labels_encoded_clip])  # labels_encoded_clip / labels_encoded_clip.norm(dim = -1, keepdim = True)
    scores_id_path_clip = os.path.join(path, 'cosine-clip_scores_id.npy')
    acc_path = os.path.join(path, 'accuracy.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
        val_acc = np.load(acc_path)
    else:
        print('Computing ID scores...')
        x_val_id_encoded = np.array([m / np.linalg.norm(m) for m in feature_id_val])
        # feature_id_val / feature_id_val.norm(dim = -1, keepdim = True)
        similarity_id = (x_val_id_encoded @ text_encoded.T)
        preds = np.argmax(similarity_id, axis=-1)
        val_acc = np.equal(preds, clip_labels).mean()
        np.save(acc_path, val_acc)
        score_id = np.max(similarity_id, axis=-1)
        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')
    x_ood_encoded = np.array([m / np.linalg.norm(m) for m in feature_ood])
    # feature_ood / feature_ood.norm(dim = -1, keepdim = True)

    similarity_ood = (x_ood_encoded @ text_encoded.T)
    score_ood = np.max(similarity_ood, axis=-1)
    return score_id, score_ood, val_acc


def evaluate_mcm_clip(feature_id_val, feature_ood, clip_labels, labels_encoded_clip, path):
    """
    This function computes the MCM score for a given set of ID data (feature_id_val) and OOD data (feature_ood)
    by first normalizing the features and then computing the dot product between the features and the encoded
    text representations (labels_encoded_clip). The resulting scores are then passed through a softmax function
    to obtain the final MCM scores. The ID scores are saved to disk (mcm-clip_scores_id.npy) along with the
    accuracy (accuracy.npy) if they have not already been computed.

    Inputs:

        feature_id_val: numpy array, shape (num_ID_data, num_features)
        The in-distribution data to evaluate the MCM scores for.
        feature_ood: numpy array, shape (num_OOD_data, num_features)
        The out-of-distribution data to evaluate the MCM scores for.
        clip_labels: numpy array, shape (num_ID_data,)
        The labels for the in-distribution data.
        labels_encoded_clip: numpy array, shape (num_texts, num_features)
        The encoded text representations.
        path: str
        The path to save the ID scores and accuracy if they have not already been computed.

    Returns:

        score_id: numpy array, shape (num_ID_data,)
        The MCM scores for the in-distribution data.
        score_ood: numpy array, shape (num_OOD_data,)
        The MCM scores for the out-of-distribution data.
        val_acc: float
        The accuracy of the prediction on the in-distribution validation data.
        """
    T = 1.
    text_encoded = torch.from_numpy(labels_encoded_clip).float()
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    scores_id_path_clip = os.path.join(path, 'mcm-clip_scores_id.npy')
    acc_path = os.path.join(path, 'accuracy.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
        val_acc = np.load(acc_path)
    else:
        print('Computing ID scores...')

        features_id = torch.from_numpy(feature_id_val).float()
        features_id /= features_id.norm(dim=-1, keepdim=True)

        out_id = features_id @ text_encoded.T
        smax_id = F.softmax(out_id / T, dim=1).data.cpu().numpy()
        score_id = np.max(smax_id, axis=1)

        preds = np.argmax(out_id.data.cpu().numpy(), axis=-1)
        val_acc = np.equal(preds, clip_labels).mean()
        np.save(acc_path, val_acc)
        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')

    features_ood = torch.from_numpy(feature_ood).float()
    features_ood /= features_ood.norm(dim=-1, keepdim=True)

    out_ood = features_ood @ text_encoded.T
    smax_ood = F.softmax(out_ood / T, dim=1).data.cpu().numpy()
    score_ood = np.max(smax_ood, axis=1)

    return score_id, score_ood, val_acc
