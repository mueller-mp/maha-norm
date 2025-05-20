import os
import csv
import argparse
import datetime
import timm
import torchvision.datasets as dset
from scipy.special import softmax
from torch.utils.data.dataset import Dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils import extract_features, extract_clip_embeddings, timm_models, fpr_at_tpr, auroc_ood, set_seed
import utils
from resnet50 import ResNetNNGUIDE, ResNetSupCon
import data.paths_config
from detection_methods import *
import datasets
import pandas as pd

os.umask(0)  # all created files and directories have full 777 permissions


class OODScore:
    def __init__(self, path_to_imagenet=data.paths_config.dset_location_dict['ImageNet1K'],
                 path_to_cache='model_outputs/cache'):
        self.path_to_cache = path_to_cache
        self.path_to_imagenet = path_to_imagenet
        self.clip_quantile = 0.99
        self.methods = ['MSP', 'MaxLogit', 'ViM', 'Mahalanobis', 'Energy+React',
                        'Energy', 'KL-Matching', 'knn', 'Relative_Mahalanobis', 'mcm', 'cosine', 
    'Mahalanobis_norm',
    'ash_s',
    'ash_b',
    'scale',
    'ssc',
    'nnguide',
    'nnguide10',
    'Relative_Mahalanobis_norm',
    'neco_l',
    'gmm',
    'nan',
    'fdbd',
    'gen',
    ]
        self.clip_transform = None
        self.val_acc = -99
        self.train_acc = -99

    def setup(self, dataset, model, ood_dataset_paths_prefix=None, clip_model=False):
        """Load and prepare the data."""
        self.dataset = dataset

        # transform from timm cfg
        config = resolve_data_config({}, model=model, use_test_size=True)
        if clip_model:
            test_transform = self.clip_transform
        else:
            test_transform = create_transform(**config)

        available_OOD_datasets = {
            'NINCO': datasets.NINCO,
            'NINCO_OOD_unit_tests': datasets.NINCOOODUnitTests,
            'NINCO_popular_datasets_subsamples': datasets.NINCOPopularDatasetsSubsamples,
        }

        self.dataset_in_train = dset.ImageFolder(os.path.join(self.path_to_imagenet, 'train'), transform=test_transform)
        self.dataset_in_val = dset.ImageFolder(os.path.join(self.path_to_imagenet, 'val'), transform=test_transform)
        if dataset.endswith('.csv'):
            if ood_dataset_paths_prefix == None:
                self.dataset_out = datasets.ImageCSVDataset(image_table_csv=dataset, transform=test_transform, )
            else:
                self.dataset_out = datasets.ImageCSVDataset(image_table_csv=dataset, paths_prefix=ood_dataset_paths_prefix, transform=test_transform)
        else:
            self.dataset_out = available_OOD_datasets[dataset](transform=test_transform)

    @staticmethod
    def check_complete(path, expected_samples, sources=['features', 'labels_true', 'logits']):
        predictions = {}
        if os.path.exists(path):
            for source in sources:
                print('checking ', source)
                names = sorted([f for f in os.listdir(path) if f.startswith(source + '_') and f.endswith('.npy')
                                and f[len(source + '_'):-len('.npy')].isdigit()])
                if len(names) == 0:
                    print('No samples in {}'.format(path))
                    return False
                predictions[source] = np.concatenate([np.load(os.path.join(path, f)) for f in names])
                print('# samples: ',len(predictions[source]))
                if source == 'text_encoded' and len(predictions[source]) == 1000:
                    continue
                else:
                    if len(predictions[source]) != expected_samples:
                        print(f'There should be {expected_samples} samples of {source} in {path},'
                              f'but there are {len(predictions[source])}')
                        return False
        return predictions

    def get_features_and_logits(self, model, train=True, val=True, ood=True, overwrite='no'):
        if train:
            save_path_train = os.path.join(os.path.join(self.path_to_cache, 'cache_train', model.model_name))
            if overwrite in {'no', 'ood', 'notrain'}:
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train))
            else:
                predictions_train = None
            if not predictions_train:
                print('Train features not complete, extracting...')
                extract_features(model, self.dataset_in_train, wo_head=False, savepath=save_path_train)
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train))
            self.train_labels = predictions_train['labels_true']
            self.feature_id_train = predictions_train['features'].astype(np.float64)  # [:,:,0,0]
            self.logits_id_train = predictions_train['logits'].astype(np.float64)
            print('Computing softmax...')
            self.softmax_id_train = softmax(self.logits_id_train, axis=-1)
            predicted_classes_train = np.argmax(self.logits_id_train, axis=-1)
            self.train_acc = np.equal(predicted_classes_train, predictions_train['labels_true']).mean()
            print('Accuracy train: ', self.train_acc)
            print('Done')
        if val:
            save_path_val = os.path.join(os.path.join(self.path_to_cache, 'cache_val', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val))
            else:
                predictions_val = None
            if not predictions_val:
                print('Val features not complete, extracting...')
                extract_features(model, self.dataset_in_val, wo_head=False, savepath=save_path_val)
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val))
            self.feature_id_val = predictions_val['features'].astype(np.float64)
            self.logits_id_val = predictions_val['logits'].astype(np.float64)
            self.labels_id_val = predictions_val['labels_true']
            print('Computing softmax...')
            self.softmax_id_val = softmax(self.logits_id_val, axis=-1)
            self.predicted_classes = np.argmax(self.logits_id_val, axis=-1)
            self.val_acc = np.equal(self.predicted_classes, self.labels_id_val).mean()
            print('Accuracy val: ', self.val_acc)
            print('Done')
        if ood:
            save_path_ood = os.path.join(os.path.join(self.path_to_cache, 'cache_ood', model.model_name, self.dataset_out.__name__))
            if overwrite in {'no'}:
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out))
            else:
                predictions_ood = None
            if not predictions_ood:
                print('OOD features ({}) not complete, extracting...'.format(self.dataset))
                extract_features(model, self.dataset_out, wo_head=False, savepath=save_path_ood)
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out))
            self.feature_ood = predictions_ood['features'].astype(np.float64)
            self.logits_ood = predictions_ood['logits'].astype(np.float64)
            self.labels_ood = predictions_ood['labels_true']
            print('Computing softmax...')
            self.softmax_ood = softmax(self.logits_ood, axis=-1)
            print('Done')

    def get_features_clip(self, model, train=False, val=True, ood=True, overwrite='no',openclip=False):
        if train:
            save_path_train = os.path.join(os.path.join(self.path_to_cache, 'cache_train', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train),
                                                      sources=['features', 'labels_true'])
            else:
                predictions_train = None
            if not predictions_train:
                print('train features not complete, extracting...')
                if openclip:
                    extract_openclip_embeddings(model, self.dataset_in_train, text=None, savepath=save_path_train)
                else:
                    extract_clip_embeddings(model, self.dataset_in_train, text=None, savepath=save_path_train)
                predictions_train = self.check_complete(save_path_train, expected_samples=len(self.dataset_in_train),
                                                      sources=['features', 'labels_true'])
            self.feature_id_train = predictions_train['features'].astype(np.float64)
            self.labels_id_train = predictions_train['labels_true']
            self.clip_labels_true = predictions_train['labels_true']
            print('train done.')           
        if val:
            save_path_val = os.path.join(os.path.join(self.path_to_cache, 'cache_val', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val),
                                                      sources=['features', 'labels_true', 'text_encoded'])
            else:
                predictions_val = None
            if not predictions_val:
                print('Val features not complete, extracting...')
                text_labels = np.load('model_outputs/im_class_clean.npy')
                if openclip:
                    extract_openclip_embeddings(model, self.dataset_in_val, text=text_labels, savepath=save_path_val)
                else:
                    extract_clip_embeddings(model, self.dataset_in_val, text=text_labels, savepath=save_path_val)
                predictions_val = self.check_complete(save_path_val, expected_samples=len(self.dataset_in_val),
                                                      sources=['features', 'labels_true', 'text_encoded'])
            self.feature_id_val = predictions_val['features'].astype(np.float64)
            self.labels_id_val = predictions_val['labels_true']
            self.labels_encoded_clip = predictions_val['text_encoded']
            self.clip_labels_true = predictions_val['labels_true']
            print('Val done.')
        if ood:
            save_path_ood = os.path.join(
                os.path.join(self.path_to_cache, 'cache_ood', model.model_name, self.dataset_out.__name__))
            if overwrite in {'no', }:
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out),
                                                      sources=['features', 'labels_true'])
            else:
                predictions_ood = None
            if not predictions_ood:
                print('OOD features ({}) not complete, extracting...'.format(self.dataset_out.__name__))
                if openclip:
                    print('Using openclip...')
                    extract_openclip_embeddings(model, self.dataset_out, savepath=save_path_ood)
                else:
                    extract_clip_embeddings(model, self.dataset_out, savepath=save_path_ood)
                predictions_ood = self.check_complete(save_path_ood, expected_samples=len(self.dataset_out),
                                                      sources=['features', 'labels_true'])
            self.feature_ood = predictions_ood['features'].astype(np.float64)
            self.labels_ood = predictions_ood['labels_true']
            print('OOD done.')

    def evaluate(self, model, OOD_classes, methods=['MSP']):
        # patly adapted from https://github.com/haoqiwang/vim/blob/master/benchmark.py
        path = os.path.join(self.path_to_cache, 'cache_methods', model.model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        if any(m in methods for m in {'ViM', 'Energy+React','ash_b','ash_s','scale', 'ssc', 'ViM_norm', 'fdbd'}):
            print('Reading w and b')
            if 'maxvit' in model.model_name or 'convnext' in model.model_name or 'tiny_vit' in model.model_name or 'regnety' in model.model_name or 'edgenext' in model.model_name or 'tresnet_v2' in model.model_name or 'swinv2' in model.model_name:
                w = model.head.fc.weight.cpu().clone().detach().numpy()
                b = model.head.fc.bias.cpu().clone().detach().numpy()
            elif ('vit' in model.model_name and 'max' not in model.model_name) or 'deit' in model.model_name or \
                    'swin' in model.model_name or 'xcit' in model.model_name or 'eva02' in model.model_name or 'mixer' in model.model_name:
                w = model.head.weight.cpu().clone().detach().numpy()
                b = model.head.bias.cpu().clone().detach().numpy()
            elif 'BiT' in model.model_name:
                w = model.head.fc.weight.clone().detach().flatten(1).cpu().numpy()  # need to flatten conv filter
                b = model.head.fc.bias.clone().detach().cpu().numpy()
            elif 'efficient' in model.model_name:
                w = model.classifier.weight.cpu().clone().detach().numpy()
                b = model.classifier.bias.cpu().clone().detach().numpy()
            elif 'resnet50' in model.model_name or 'resnet18' in model.model_name or 'resnet101' in model.model_name or 'resnet152' in model.model_name or 'rn50supcon' in model.model_name:
                w = model.fc.weight.cpu().clone().detach().numpy()
                b = model.fc.bias.cpu().clone().detach().numpy()
            else:
                state_dict = model.model.state_dict()
                w = state_dict['fc.weight'].clone().detach().cpu().numpy()
                b = state_dict['fc.bias'].clone().detach().cpu().numpy()
            u = -np.matmul(pinv(w), b)
        methods_results = {}
        for method in methods:
            if method == 'MSP':
                scores_id, scores_ood = evaluate_MSP(self.softmax_id_val, self.softmax_ood)
            elif method == 'MaxLogit':
                scores_id, scores_ood = evaluate_MSP(self.logits_id_val, self.logits_ood)
            elif method=='nan':
                scores_id, scores_ood = evaluate_nan(features_in_distribution=self.feature_id_val, features_out_of_distribution=self.feature_ood, use_pos=False)
            elif method=='nan_pos':
                scores_id, scores_ood = evaluate_nan(features_in_distribution=self.feature_id_val, features_out_of_distribution=self.feature_ood, use_pos=True)
            elif method == 'Energy':
                scores_id, scores_ood = evaluate_Energy(logits_in_distribution=self.logits_id_val,
                                                        logits_out_of_distribution=self.logits_ood)
            elif method=='ssc':
                scores_id, scores_ood = evaluate_softmax_scaled_cosine(feature_id_val=self.feature_id_val, feature_ood=self.feature_ood, w=w, s=1)
            elif method == 'ViM':
                scores_id, scores_ood = evaluate_ViM(feature_id_train=self.feature_id_train,
                                                     feature_id_val=self.feature_id_val,
                                                     feature_ood=self.feature_ood, logits_id_train=self.logits_id_train,
                                                     logits_id_val=self.logits_id_val,
                                                     logits_ood=self.logits_ood, u=u, path=path)
            elif method == 'fdbd':
                scores_id, scores_ood = evaluate_fdbd(features_train=self.feature_id_train, features_val=self.feature_id_val, features_ninco=self.feature_ood,logits_val=self.logits_id_val,logits_ninco=self.logits_ood,w=w) 
            elif method == 'gen':
                scores_id, scores_ood = evaluate_gen(self.softmax_id_val, self.softmax_ood, gamma=0.1, M=100)                 
            elif method == 'Mahalanobis':
                scores_id, scores_ood = evaluate_Mahalanobis(feature_id_train=self.feature_id_train,
                                                             feature_id_val=self.feature_id_val,
                                                             feature_ood=self.feature_ood,
                                                             train_labels=self.train_labels, path=path)
            elif method=='Mahalanobis_norm':
                scores_id, scores_ood = evaluate_Mahalanobis_norm(feature_id_train=self.feature_id_train,
                                                             feature_id_val=self.feature_id_val,
                                                             feature_ood=self.feature_ood,
                                                             train_labels=self.train_labels, path=path)
            elif method == 'Relative_Mahalanobis':
                scores_id, scores_ood = evaluate_Relative_Mahalanobis(feature_id_train=self.feature_id_train,
                                                                      feature_id_val=self.feature_id_val,
                                                                      feature_ood=self.feature_ood,
                                                                      train_labels=self.train_labels, path=path)
            elif method=='Relative_Mahalanobis_norm':
                scores_id, scores_ood = evaluate_Relative_Mahalanobis_norm(feature_id_train=self.feature_id_train,
                                                                      feature_id_val=self.feature_id_val,
                                                                      feature_ood=self.feature_ood,
                                                                      train_labels=self.train_labels, path=path)
            elif method == 'KL-Matching':
                scores_id, scores_ood = evaluate_KL_Matching(softmax_id_train=self.softmax_id_train,
                                                             softmax_id_val=self.softmax_id_val,
                                                             softmax_ood=self.softmax_ood, path=path)
            elif method == 'Energy+React':
                scores_id, scores_ood = evaluate_Energy_React(feature_id_train=self.feature_id_train,
                                                              feature_id_val=self.feature_id_val,
                                                              feature_ood=self.feature_ood, w=w, b=b, path=path)
            elif method == 'knn':
                scores_id, scores_ood = evaluate_KNN(feature_id_train=self.feature_id_train,
                                                     feature_id_val=self.feature_id_val,
                                                     feature_ood=self.feature_ood, path=path)
            elif method == 'mcm':
                scores_id, scores_ood = evaluate_rcos(feature_id_train=self.feature_id_train,
                                                      feature_id_val=self.feature_id_val,
                                                      feature_ood=self.feature_ood, train_labels=self.train_labels,
                                                      path=path)
            elif method == 'cosine':
                scores_id, scores_ood = evaluate_cosine(feature_id_train=self.feature_id_train,
                                                        feature_id_val=self.feature_id_val,
                                                        feature_ood=self.feature_ood, train_labels=self.train_labels,
                                                        path=path)
            elif method == 'cosine-clip':
                scores_id, scores_ood, self.val_acc = evaluate_cosine_clip(feature_id_val=self.feature_id_val,
                                                             feature_ood=self.feature_ood,
                                                             clip_labels=self.clip_labels_true,
                                                             labels_encoded_clip=self.labels_encoded_clip, path=path)
            elif method == 'mcm-clip':
                scores_id, scores_ood, self.val_acc = evaluate_mcm_clip(feature_id_val=self.feature_id_val,
                                                          feature_ood=self.feature_ood,
                                                          clip_labels=self.clip_labels_true,
                                                          labels_encoded_clip=self.labels_encoded_clip, path=path)                      
            elif method=='ash_s':
                scores_id, scores_ood = evaluate_ash_s(feature_id_val=self.feature_id_val,feature_ood=self.feature_ood,w=w,b=b,percentile=90)
            elif method=='ash_b':
                scores_id, scores_ood = evaluate_ash_b(feature_id_val=self.feature_id_val,feature_ood=self.feature_ood,w=w,b=b,percentile=65)
            elif method=='scale':
                scores_id, scores_ood = evaluate_scale(feature_id_val=self.feature_id_val,feature_ood=self.feature_ood,w=w,b=b,percentile=85)
            elif method=='nnguide':
                scores_id, scores_ood = evaluate_nnguide(feature_id_train=self.feature_id_train, feature_id_val=self.feature_id_val, feature_ood=self.feature_ood,logits_id_train=self.logits_id_train,logits_id_val=self.logits_id_val,logits_ood=self.logits_ood,path=path,alpha=0.01,K=1000)
            elif method=='nnguide10':
                scores_id, scores_ood = evaluate_nnguide(feature_id_train=self.feature_id_train, feature_id_val=self.feature_id_val, feature_ood=self.feature_ood,logits_id_train=self.logits_id_train,logits_id_val=self.logits_id_val,logits_ood=self.logits_ood,path=path,alpha=0.01,K=10)
            elif method=='neco_l':
                scores_id, scores_ood = evaluate_neco(self.feature_id_train, self.feature_id_val, self.feature_ood, self.logits_id_val, self.logits_ood, path=path,use_logit=True)
            elif method=='gmm':
                scores_id, scores_ood = evaluate_gmm(self.feature_id_train, self.feature_id_val, self.feature_ood,self.train_labels,path=path,normalize=False)
            else:
                raise NotImplementedError(f'Method {method} not implemented.')
            
            print('s-id finite:',np.isfinite(scores_id).all())
            print('s-ood finite:',np.isfinite(scores_ood).all())
            methods_results[method] = {'scores_id': scores_id,
                                       'scores_ood': scores_ood}

            for c in OOD_classes:
                class_indices = np.where(self.labels_ood == self.dataset_out.class_to_idx[c])
                scores_on_ood_class = scores_ood[class_indices]
                methods_results[method][c] = {'auroc': auroc_ood(scores_id, scores_on_ood_class),
                                              'fpr_at_95': fpr_at_tpr(scores_id, scores_on_ood_class, 0.95)}
            methods_results[method]['samples_mean_auroc'] = auroc_ood(scores_id, scores_ood)
            methods_results[method]['samples_mean_fpr_at_95'] = fpr_at_tpr(scores_id, scores_ood, 0.95)
            methods_results[method]['ood_classes_mean_auroc'] = np.mean(
                np.array([methods_results[method][c]['auroc'] for c in OOD_classes]))
            methods_results[method]['ood_classes_mean_fpr_at_95'] = np.mean(
                np.array([methods_results[method][c]['fpr_at_95'] for c in OOD_classes]))

            print('{} on {} evaluated with {}.\nAuroc: {}\nfpr at 95: {}\naccuracy val: {}\n accuracy train: {}'.format(
            method, self.dataset, model.model_name, methods_results[method]['ood_classes_mean_auroc'],
            methods_results[method]['ood_classes_mean_fpr_at_95'], self.val_acc, self.train_acc))
        # save results
        savepath = os.path.join(self.path_to_cache, 'scores', model.model_name, self.dataset_out.__name__)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        eval_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        np.savez(os.path.join(savepath, f'E{eval_time}.npz'), methods_results=methods_results,
                 id_labels=self.labels_id_val, ood_labels=self.labels_ood, ood_classes=OOD_classes,
                 val_acc=self.val_acc, train_acc=self.train_acc)


methods_train_usage = {
    'MSP': False,
    'MaxLogit': False,
    'Energy': False,
    'mcm-clip': False,
    'ViM': True,
    'Mahalanobis': True,
    'Relative_Mahalanobis': True,
    'KL-Matching': True,
    'Energy+React': True,
    'knn': True,
    'mcm': True,
    'cosine': True,
    'cosine-clip': False,
    'mcm-clip':False,
    'Mahalanobis_norm': True,
    'Relative_Mahalanobis_norm':True,
    'ash_s':False,
    'ash_b':False,
    'scale':False,
    'ssc':False,
    'nnguide':True,
    'nnguide10':True,
    'neco_l':True,
    'gmm':True,
    'nan':True,
    'nan_pos':True,
    'fdbd':True,
    'gen':False,
}

parser = argparse.ArgumentParser(description='OOD Evaluation on NINCO')
parser.add_argument('--path_to_weights', default='model_weights', )
parser.add_argument('--model_name', default='convnext_base_in22ft1k')
parser.add_argument('--dataset', type=str, default='NINCO') #choices=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples' ...csv],
parser.add_argument('--dataset_paths_prefix', type=str) #choices=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples' ...csv],
parser.add_argument('--overwrite_model_outputs', type=str, choices=['no', 'all', 'notrain', 'ood'], default='no')
parser.add_argument('--method', default='MSP')
parser.add_argument('--path_to_imagenet', default=data.paths_config.dset_location_dict['ImageNet1K'])
parser.add_argument('--path_to_cache', default='./cache')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--seed', type=int, default=99)


def main():
    args = parser.parse_args()
    set_seed(args.seed)
    torch.hub.set_dir(args.path_to_weights)
    task = OODScore(path_to_cache=args.path_to_cache, path_to_imagenet=args.path_to_imagenet)
    methods = task.methods if args.method == 'all' else [args.method]
    need_train_outputs = any([methods_train_usage[m] for m in methods])  # raises KeyError if a method is not available
    if args.model_name=='all':
        model_names=list(timm_models.keys())
    else:
        model_names = [args.model_name]
    ood_datasets=['./data/ssb_hard.csv', './data/places365.csv', './data/texture.csv', './data/openimages_o.csv', './data/inaturalist.csv'] if args.dataset=='openood-datasets' else [args.dataset]
    ood_datasets=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples'] if args.dataset=='ninco-datasets' else ood_datasets
    for ood_dataset_name in ood_datasets:
        current_dataset=ood_dataset_name
        for model_name in model_names:
            if model_name in timm_models.keys():
                model = timm.create_model(**timm_models[model_name]['config']).cuda().eval()
                model.model_name = model_name
                model.batch_size = args.batch_size
                model.project_features = False
                print('Created model {}.'.format(model.model_name))
                task.setup(current_dataset, model, ood_dataset_paths_prefix=args.dataset_paths_prefix, clip_model=False)
                print('Task is set up.')
                task.get_features_and_logits(model, ood=True, train=need_train_outputs,
                                            overwrite=args.overwrite_model_outputs)
                OOD_classes = task.dataset_out.classes
                task.evaluate(model, OOD_classes=OOD_classes, methods=methods)
                print(f'# ood classes: {len(OOD_classes)}')
            elif model_name=='rn50supcon':
                model = ResNetSupCon()
                # sd = torch.load('/mnt/qb/hein/mmueller67/vkd/resnet50-supcon.pt')
                print('Loading SD')
                sd = torch.load('/mnt/qb/hein/mmueller67/vkd/resnet50-supcon.pt')
                print('Loaded SD')
                model.load_state_dict(sd['model_state_dict'])
                model = model.cuda().eval()
                model.model_name = model_name
                model.batch_size = args.batch_size
                model.project_features = False
                print('Created model {}.'.format(model.model_name))
                task.setup(current_dataset, model, ood_dataset_paths_prefix=args.dataset_paths_prefix, clip_model=False)
                print('Task is set up.')
                task.get_features_and_logits(model, ood=True, train=need_train_outputs,
                                            overwrite=args.overwrite_model_outputs)
                #if current_dataset.endswith('.csv'):
                OOD_classes = task.dataset_out.classes
                task.evaluate(model, OOD_classes=OOD_classes, methods=methods)
                print(f'# ood classes: {len(OOD_classes)}')
            else:
                raise NotImplementedError(
                    '{} is not implemented. Please add it to the model-dictionary.'.format(model_name))


if __name__ == "__main__":
    with torch.no_grad():
        main()
