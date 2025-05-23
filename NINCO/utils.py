#!/usr/bin/env python
import torch
from tqdm import tqdm
import os
# import clip, open_clip
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from itertools import groupby
import json
import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore
import random
from scipy.linalg import cholesky


timm_models = {
'vit_small_patch14_dinov2.ft_i1k_70eps':{'config':{'model_name':'vit_small_patch14_dinov2','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/timm-ft/20250328-084036-vit_small_patch14_dinov2-518/checkpoint-69.pth.tar','num_classes':1000}},    
'vit_small_patch14_dinov2.ft_i1k_60eps':{'config':{'model_name':'vit_small_patch14_dinov2','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/timm-ft/20250328-084036-vit_small_patch14_dinov2-518/checkpoint-59-perm.pth.tar','num_classes':1000}},    
'vit_small_patch14_dinov2.ft_i1k_30eps':{'config':{'model_name':'vit_small_patch14_dinov2','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/timm-ft/20250325-170305-vit_small_patch14_dinov2-518/checkpoint-29-perm.pth.tar','num_classes':1000}},    
'vit_small_patch14_dinov2.ft_i1k_9eps':{'config':{'model_name':'vit_small_patch14_dinov2','checkpoint_path': '/mnt/SHARED/mmueller67/vkd/timm-ft/20250325-222245-vit_small_patch14_dinov2-518/last.pth.tar','num_classes':1000}},
'vit_small_patch14_dinov2.ft_i1k_13eps':{'config':{'model_name':'vit_small_patch14_dinov2','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/timm-ft/20250325-170305-vit_small_patch14_dinov2-518/checkpoint-13.pth.tar','num_classes':1000}},    
'vit_small_patch14_dinov2.ft_i1k_6eps':{'config':{'model_name':'vit_small_patch14_dinov2','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/timm-ft/20250325-174942-vit_small_patch14_dinov2-518/last.pth.tar','num_classes':1000}},
    'vit_small_patch16_224.dino_i1k6eps':{'config':{'model_name':'vit_small_patch16_224.dino','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/timm-ft/20250326-075337-vit_small_patch16_224_dino-224/last.pth.tar','num_classes':1000}},
        'vit_small_patch16_224.dino':{'config': {'model_name': 'vit_small_patch16_224.dino',
   'pretrained': True,'num_classes':1000}}, 
       'vit_base_patch16_224.dino':{'config': {'model_name': 'vit_base_patch16_224.dino',
   'pretrained': True,'num_classes':1000}}, 
    'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k':{'config': {'model_name': 'vit_base_patch16_clip_224.laion2b_ft_in12k_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k':{'config': {'model_name': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch16_224.orig_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_224.orig_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch16_384.orig_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_384.orig_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_large_patch32_384.orig_in21k_ft_in1k':{'config': {'model_name': 'vit_large_patch32_384.orig_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch16_224.augreg2_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch16_384.augreg2_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_384.augreg2_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch8_224.augreg2_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch8_224.augreg2_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'vit_base_patch8_224.augreg_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch8_224.augreg_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k':{'config': {'model_name': 'swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k':{'config': {'model_name': 'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'swinv2_base_window16_256.ms_in1k':{'config': {'model_name': 'swinv2_base_window16_256.ms_in1k',
   'pretrained': True,'num_classes':1000}},
    'swinv2_small_window16_256.ms_in1k':{'config': {'model_name': 'swinv2_small_window16_256.ms_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet18.tv_in1k':{'config': {'model_name': 'resnet18.tv_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet50.tv_in1k':{'config': {'model_name': 'resnet50.tv_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet101.tv_in1k':{'config': {'model_name': 'resnet101.tv_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet152.tv_in1k':{'config': {'model_name': 'resnet152.tv_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet50.tv2_in1k':{'config': {'model_name': 'resnet50.tv2_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet101.tv2_in1k':{'config': {'model_name': 'resnet101.tv2_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnet152.tv2_in1k':{'config': {'model_name': 'resnet152.tv2_in1k',
   'pretrained': True,'num_classes':1000}},
'deit3_large_patch16_384.fb_in22k_ft_in1k': {'config': {'model_name': 'deit3_large_patch16_384.fb_in22k_ft_in1k', 'pretrained': True, 'num_classes': 1000}},
'deit3_base_patch16_384.fb_in22k_ft_in1k': {'config': {'model_name': 'deit3_base_patch16_384.fb_in22k_ft_in1k', 'pretrained': True, 'num_classes': 1000}},
'deit3_large_patch16_384.fb_in1k': {'config': {'model_name': 'deit3_large_patch16_384.fb_in1k', 'pretrained': True, 'num_classes': 1000}},
'deit3_base_patch16_384.fb_in1k': {'config': {'model_name': 'deit3_base_patch16_384.fb_in1k', 'pretrained': True, 'num_classes': 1000}},
'deit3_small_patch16_384.fb_in22k_ft_in1k': {'config': {'model_name': 'deit3_small_patch16_384.fb_in22k_ft_in1k', 'pretrained': True, 'num_classes': 1000}},
'deit3_small_patch16_384.fb_in1k': {'config': {'model_name': 'deit3_small_patch16_384.fb_in1k', 'pretrained': True, 'num_classes': 1000}},
    'tf_efficientnetv2_l.in21k_ft_in1k':{'config': {'model_name': 'tf_efficientnetv2_l.in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'tf_efficientnetv2_l.in1k':{'config': {'model_name': 'tf_efficientnetv2_l.in1k',
   'pretrained': True,'num_classes':1000}},
    'tf_efficientnetv2_s.in21k_ft_in1k':{'config': {'model_name': 'tf_efficientnetv2_s.in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'tf_efficientnetv2_s.in1k':{'config': {'model_name': 'tf_efficientnetv2_s.in1k',
   'pretrained': True,'num_classes':1000}},
    'tf_efficientnetv2_m.in21k_ft_in1k':{'config': {'model_name': 'tf_efficientnetv2_m.in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'tf_efficientnetv2_m.in1k':{'config': {'model_name': 'tf_efficientnetv2_m.in1k',
   'pretrained': True,'num_classes':1000}},
    'resnetv2_152x2_bit.goog_in21k_ft_in1k':{'config': {'model_name': 'resnetv2_152x2_bit.goog_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'resnetv2_50x1_bit.goog_in21k_ft_in1k':{'config': {'model_name': 'resnetv2_50x1_bit.goog_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'mixer_b16_224.goog_in21k_ft_in1k':{'config': {'model_name': 'mixer_b16_224.goog_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'convnextv2_tiny.fcmae_ft_in22k_in1k_384':{'config': {'model_name': 'convnextv2_tiny.fcmae_ft_in22k_in1k_384',
   'pretrained': True,'num_classes':1000}},
    'convnextv2_base.fcmae_ft_in22k_in1k_384':{'config': {'model_name': 'convnextv2_base.fcmae_ft_in22k_in1k_384',
   'pretrained': True,'num_classes':1000}},
    'convnextv2_large.fcmae_ft_in22k_in1k_384':{'config': {'model_name': 'convnextv2_large.fcmae_ft_in22k_in1k_384',
   'pretrained': True,'num_classes':1000}},

       'convnextv2_tiny.fcmae_ft_in1k':{'config': {'model_name': 'convnextv2_tiny.fcmae_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'convnextv2_base.fcmae_ft_in1k':{'config': {'model_name': 'convnextv2_base.fcmae_ft_in1k',
   'pretrained': True,'num_classes':1000}},
    'convnextv2_large.fcmae_ft_in1k':{'config': {'model_name': 'convnextv2_large.fcmae_ft_in1k',
   'pretrained': True,'num_classes':1000}},

    'convnext_base.fb_in1k':{'config': {'model_name': 'convnext_base.fb_in1k',
   'pretrained': True,'num_classes':1000}},
    'convnext_base.fb_in22k_ft_in1k':{'config': {'model_name': 'convnext_base.fb_in22k_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
        'vit_base_patch16_224.augreg_in1k':{'config': {'model_name': 'vit_base_patch16_224.augreg_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_small_patch16_384.augreg_in21k_ft_in1k':{'config': {'model_name': 'vit_small_patch16_384.augreg_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}},  
    'vit_small_patch16_384.augreg_in1k':{'config': {'model_name': 'vit_small_patch16_384.augreg_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_small_patch16_224.augreg_in1k':{'config': {'model_name': 'vit_small_patch16_224.augreg_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k':{'config': {'model_name': 'mambaout_base_plus_rw.sw_e150_r384_in12k_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_small_patch14_dinov2':{'config': {'model_name': 'vit_small_patch14_dinov2',
   'pretrained': True,'num_classes':1000}}, 
    'vit_base_patch14_dinov2':{'config': {'model_name': 'vit_base_patch14_dinov2',
   'pretrained': True,'num_classes':1000}}, 
    'vit_large_patch14_dinov2':{'config': {'model_name': 'vit_large_patch14_dinov2',
   'pretrained': True,'num_classes':1000}}, 
    'vit_giant_patch14_dinov2':{'config': {'model_name': 'vit_giant_patch14_dinov2',
   'pretrained': True,'num_classes':1000}}, 
    'vit_large_patch16_384.augreg_in21k':{'config': {'model_name': 'vit_large_patch16_384.augreg_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000, 'checkpoint_path':'/mnt/qb/hein/mmueller67/vkd/NINCO/weights/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz'}}, 
        'vit_so400m_patch14_siglip_378.webli_ft_in1k':{'config': {'model_name': 'vit_so400m_patch14_siglip_378.webli_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
   'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k':{'config': {'model_name': 'eva02_base_patch14_448.mim_in22k_ft_in22k_in1k',
   'pretrained': True,'num_classes':1000}}, 
   'eva02_base_patch14_448.mim_in22k_ft_in1k':{'config': {'model_name': 'eva02_base_patch14_448.mim_in22k_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
   'eva02_large_patch14_448.mim_m38m_ft_in1k':{'config': {'model_name': 'eva02_large_patch14_448.mim_m38m_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k':{'config': {'model_name': 'eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k':{'config': {'model_name': 'vit_huge_patch14_clip_336.laion2b_ft_in12k_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k':{'config': {'model_name': 'vit_large_patch14_clip_336.laion2b_ft_in12k_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_base_patch14_clip_336.laion2b_ft_in12k_in1k':{'config': {'model_name': 'vit_base_patch14_clip_336.laion2b_ft_in12k_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_large_patch14_clip_336.openai_ft_in12k_in1k':{'config': {'model_name': 'vit_large_patch14_clip_336.openai_ft_in12k_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_large_patch16_224.augreg_in21k_ft_in1k': {'config': {'model_name': 'vit_large_patch16_224.augreg_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_large_patch16_384.augreg_in21k_ft_in1k': {'config': {'model_name': 'vit_large_patch16_384.augreg_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_small_patch16_224.augreg_in21k_ft_in1k': {'config': {'model_name': 'vit_small_patch16_224.augreg_in21k_ft_in1k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_small_patch16_224.augreg_in21k': {'config': {'model_name': 'vit_small_patch16_224.augreg_in21k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_tiny_patch16_224.in21k': {'config': {'model_name': 'vit_tiny_patch16_224.augreg_in21k',
   'pretrained': True,'num_classes':1000}}, 
    'vit_base_patch16_224.in21k': {'config': {'model_name': 'vit_base_patch16_224.augreg_in21k',
   'pretrained': True,'num_classes':1000}},    
   'vit_vkd_test50eps_adamw_wrongaug':{'config':{'model_name':'vit_base_patch16_224.in21k_ft_in1k','checkpoint_path': '/mnt/qb/hein/mmueller67/vkd/vkd/output/augreg50_ft_AdamWalpha0.0_zeta0.0_xeta1.0_gamma0.0_stu-vit_b_tea-vit_t/latest.pth.tar'}},
    'vit_base_patch16_224_miil.in21k_ft_in1k': {'config': {'model_name': 'vit_base_patch16_224_miil.in21k_ft_in1k',
   'pretrained': True}},
    'vit_base_patch16_224_miil': {'config': {'model_name': 'vit_base_patch16_224_miil',
   'pretrained': True}},
    'tresnet_v2_l.miil_in21k_ft_in1k': {'config': {'model_name': 'tresnet_v2_l.miil_in21k_ft_in1k',
   'pretrained': True}},
   'eva02_small_patch14_336.mim_in22k_ft_in1k':{'config': {'model_name': 'eva02_small_patch14_336.mim_in22k_ft_in1k',
   'pretrained': True}},
    'eva02_tiny_patch14_336.mim_in22k_ft_in1k': {'config': {'model_name': 'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
   'pretrained': True}},
    'edgenext_small.usi_in1k': {'config': {'model_name': 'edgenext_small.usi_in1k',
   'pretrained': True}},
    'convnextv2_femto.fcmae_ft_in1k': {'config': {'model_name': 'convnextv2_femto.fcmae_ft_in1k',
   'pretrained': True}}, 
 'regnety_160': {'config': {'model_name': 'regnety_160',
   'pretrained': True}}, 
 'regnety_160.deit_in1k': {'config': {'model_name': 'regnety_160.deit_in1k',
   'pretrained': True}}, 
     'vit_tiny_patch16_384.augreg_in21k_ft_in1k': {'config': {'model_name': 'vit_tiny_patch16_384.augreg_in21k_ft_in1k',
   'pretrained': True}}, 
     'vit_tiny_patch16_384.augreg_in21k': {'config': {'model_name': 'vit_tiny_patch16_384.augreg_in21k',
   'pretrained': True}}, 
 'vit_tiny_patch16_224.augreg_in21k_ft_in1k': {'config': {'model_name': 'vit_tiny_patch16_224.augreg_in21k_ft_in1k',
   'pretrained': True}}, 
 'deit_tiny_distilled_patch16_224.fb_in1k': {'config': {'model_name': 'deit_tiny_distilled_patch16_224.fb_in1k',
   'pretrained': True}}, 
 'deit_tiny_patch16_224.fb_in1k': {'config': {'model_name': 'deit_tiny_patch16_224.fb_in1k',
   'pretrained': True}}, 
 'tiny_vit_5m_224.dist_in22k_ft_in1k': {'config': {'model_name': 'tiny_vit_5m_224.dist_in22k_ft_in1k',
   'pretrained': True}}, 
 'tiny_vit_5m_224.in1k': {'config': {'model_name': 'tiny_vit_5m_224.in1k',
   'pretrained': True}}, 
    'BiT_m': {'config': {'model_name': 'resnetv2_101x1_bitm', 'pretrained': True}},
    'BiT_s': {'config': {'model_name': 'resnetv2_101x1_bitm',
                         'checkpoint_path': './model_weights/checkpoints/BiT-S-R101x1.npz'}},
    'vit_base_patch16_384.augreg_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_384.augreg_in21k_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_224.augreg_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_224.augreg_in21k_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_224.augreg2_in21k_ft_in1k':{'config': {'model_name': 'vit_base_patch16_224.augreg2_in21k_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_224_21kpre': {'config': {'model_name': 'vit_base_patch16_224', 'pretrained': True}},
    'vit_base_patch16_384_21kpre': {'config': {'model_name': 'vit_base_patch16_384', 'pretrained': True}},
    'convnext_base_in22ft1k': {'config': {'model_name': 'convnext_base_in22ft1k', 'pretrained': True}},
    'convnext_base': {'config': {'model_name': 'convnext_base', 'pretrained': True}},
    'convnext_tiny-22k': {'config': {'model_name': 'convnext_tiny_384_in22ft1k', 'pretrained': True}},
    'deit3_base_patch16_224': {'config': {'model_name': 'deit3_base_patch16_224', 'pretrained': True}},
    'deit3_base_patch16_224_in21ft1k': {
        'config': {'model_name': 'deit3_base_patch16_224_in21ft1k', 'pretrained': True}},
    'tf_efficientnetv2_m': {'config': {'model_name': 'tf_efficientnetv2_m', 'pretrained': True}},
    'tf_efficientnetv2_m_in21ft1k': {'config': {'model_name': 'tf_efficientnetv2_m_in21ft1k', 'pretrained': True}},
    'swinv2-22k': {'config': {'model_name': 'swinv2_base_window12to16_192to256_22kft1k', 'pretrained': True}},
    'swinv2-1k': {'config': {'model_name': 'swinv2_base_window16_256', 'pretrained': True}},
    'deit3-384-22k': {'config': {'model_name': 'deit3_base_patch16_384_in21ft1k', 'pretrained': True}},
    'deit3-384-1k': {'config': {'model_name': 'deit3_base_patch16_384', 'pretrained': True}},
    'tf_efficientnet_b7_ns': {'config': {'model_name': 'tf_efficientnet_b7_ns', 'pretrained': True}},
    'tf_efficientnet_b7': {'config': {'model_name': 'tf_efficientnet_b7', 'pretrained': True}},
    'resnet50.tv2_in1k': {'config': {'model_name': 'resnet50.tv2_in1k', 'pretrained': True}},
    'efficientnet_b0': {'config': {'model_name': 'efficientnet_b0', 'pretrained': True}},
    'vit_base_patch16_384_laion2b_in12k_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', 'pretrained': True}},
    'vit_base_patch16_384_laion2b_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.laion2b_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_384_openai_in12k_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.openai_ft_in12k_in1k', 'pretrained': True}},
    'vit_base_patch16_384_openai_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.openai_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_384': {'config': {'model_name': 'vit_base_patch16_384.augreg_in1k', 'pretrained': True},
                             'batch_size': 128, 'server': 'curie'},
    'xcit_medium_24_p16_224_dist': {'config': {'model_name': 'xcit_medium_24_p16_224_dist', 'pretrained': True}},
    'xcit_medium_24_p16_224': {'config': {'model_name': 'xcit_medium_24_p16_224', 'pretrained': True}},
}



imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

NINCO_popular_datasets_subsamples_class_names = [
    'Places',
    'iNaturalist_OOD_Plants',
    'Species',
    'Imagenet_O',
    'OpenImage_O',
    'Textures'
]

models_clip = {'clip-ViT-B16': 'ViT-B/16', 'clip-ViT-L14-336': 'ViT-L/14@336px'}


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def extract_features(model, dataset, savepath, wo_head=False):
    torch.backends.cudnn.benchmark = True
    # save
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    model.eval()

    # slice dataloaders in train
    slice_length = 50000
    n_slices = -((-len(dataset)) // slice_length)
    index_slices = {}
    slice_datasets = {}

    for i in range(n_slices):
        # check if current iterate is already saved in directoy (for train
        files = os.listdir(savepath)
        complete = n_slices>1 # True for train, else False
        for feat in ['logits_{}.npy'.format(i), 'features_{}.npy'.format(i), 'labels_true_{}.npy'.format(i)]:
            if feat not in files:
                complete = False

        # if not, save create dataloader and save features
        # if True: #not complete:
        # complete = False
        if not complete:
            print('Extracting features set ', i)
            index_slices[i] = range(i * slice_length, min((i + 1) * slice_length, len(dataset)))
            slice_datasets[i] = torch.utils.data.Subset(dataset, index_slices[i])
            slice_datasets[i].__name__ = f'slice{index_slices[i].start}_to_{index_slices[i].stop}'
            slice_datasets[i].classes = dataset.classes
            dataloader = torch.utils.data.DataLoader(slice_datasets[i], batch_size=model.batch_size)

            features = []
            logits_ = []
            labels_true = []
            with torch.no_grad():
                for (x, label) in tqdm(dataloader):
                    labels_true.append(label)
                    x = x.cuda()
                    feat_batch_preact = model.forward_features(x)
                    if 'vkd' in model.model_name:
                        x, x_dist, x_patch = feat_batch_preact
                        if 'clstkn' in model.model_name: # use class token embedding as feature
                            feat_batch=x#.copy().detach()
                        elif 'disttkn' in model.model_name: # use distillation token embeddng as feature
                            feat_batch=x_dist#.copy().detach()
                        else: # use pooled patch, which is used for feature distillation
                            feat_batch=x_patch.mean(1)
                        x = model.head(x)
                        x_dist = model.head_dist(x_dist)
                        # during inference, return the average of both classifier predictions
                        logits = (x + x_dist) / 2
                    else:
                        feat_batch = feat_batch_preact[:, 0] if wo_head else model.forward_head(feat_batch_preact,
                                                                                            pre_logits=True)
                        logits = model.forward_head(feat_batch_preact)

                    # project features if required
                    feat_batch = model.projector(feat_batch) if model.project_features else feat_batch

                    feat_batch = feat_batch.cpu().numpy()
                    logits = logits.cpu().numpy()
                    features.append(feat_batch)
                    logits_.append(logits)

            # save
            labels_true = torch.cat(labels_true).numpy()
            features = np.concatenate(features, axis=0)
            logits_ = np.concatenate(logits_, axis=0)
            predictions_dict = {'logits_{}'.format(i): logits_, 'features_{}'.format(i): features,
                                'labels_true_{}'.format(i): labels_true}

            for name, data in predictions_dict.items():
                np.save(savepath + '/' + name,
                        data)



def extract_openclip_embeddings(model, dataset, savepath, text=None):
    print(savepath)
    print('Len dataset: ',len(dataset))
    torch.backends.cudnn.benchmark = True
    # save
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    model.eval()

    # slice dataloaders
    slice_length = 50000
    n_slices = -((-len(dataset)) // slice_length)
    index_slices = {}
    slice_datasets = {}

    for i in range(n_slices):
        print(i)
        # check if current iterate is already saved in directoy (for train
        files = os.listdir(savepath)
        complete = n_slices>1 # True for train, else False
        for feat in ['features_{}.npy'.format(i), 'labels_true_{}.npy'.format(i)]:
            if feat not in files:
                complete = False

        # if not, save create dataloader and save features
        # if True: #not complete:
        complete = False
        if not complete:
            print('Extracting features set ', i)
            index_slices[i] = range(i * slice_length, min((i + 1) * slice_length, len(dataset)))
            slice_datasets[i] = torch.utils.data.Subset(dataset, index_slices[i])
            slice_datasets[i].__name__ = f'slice{index_slices[i].start}_to_{index_slices[i].stop}'
            slice_datasets[i].classes = dataset.classes
            dataloader = torch.utils.data.DataLoader(slice_datasets[i], batch_size=model.batch_size)

            features = []
            labels_true = []
            with torch.no_grad():
                for (x, label) in tqdm(dataloader):
                    labels_true.append(label)
                    x = x.cuda()

                    feat_batch = model.encode_image(x)

                    feat_batch = feat_batch.cpu().numpy()
                    features.append(feat_batch)
                    x=x.cpu()

            # save
            labels_true = torch.cat(labels_true).numpy()
            features = np.concatenate(features, axis=0)
            predictions_dict = {'features_{}'.format(i): features,
                                'labels_true_{}'.format(i): labels_true}

            for name, data in predictions_dict.items():
                np.save(savepath + '/' + name,
                        data)

    if text is not None:
        print('Extracting normal text...')
        # normal 'a photo of ...'
        tokenizer = open_clip.get_tokenizer(models_openclip[model.model_name]['basemodel'])
        text_inputs = torch.cat([tokenizer(f"a photo of a {c}") for c in text]).cuda()
        with torch.no_grad():
            text_encoded = model.encode_text(text_inputs).cpu().numpy()
        np.save(savepath + '/' + 'text_encoded_0',
                text_encoded)
        
        # cupl prompts
        print('Extracting Cupl prompts...')
        file_path='./model_outputs/cupl_prompts.json'
        # prompts = json.loads('./model_outputs/cupl_prompts.json')
        with open(file_path, 'r') as file:
            prompts = json.load(file)
        encoded_texts = []
        with torch.no_grad():
            for classname in imagenet_classes:
                encoded_class_text = model.encode_text(torch.cat([tokenizer(prompt) for prompt in prompts[classname]]).cuda())
                encoded_class_text/=encoded_class_text.norm(dim=-1,keepdim=True)
                encoded_class_text=encoded_class_text.mean(0) # encode prompts per class and compute average encoding
                encoded_class_text/=encoded_class_text.norm() # normalize
                encoded_texts.append(encoded_class_text.unsqueeze(0))
        zeroshot_weights = torch.cat(encoded_texts,dim=0).cpu().numpy()
        np.save(savepath + '/' + 'cupl_text_encoded_0',
                zeroshot_weights)
        # without mean reduction
        print('Extracting Cupl prompts without mean...')
        encoded_texts = []
        with torch.no_grad():
            for classname in imagenet_classes:
                encoded_class_text = model.encode_text(torch.cat([tokenizer(prompt) for prompt in prompts[classname]]).cuda())
                encoded_class_text/=encoded_class_text.norm(dim=-1,keepdim=True)
                encoded_texts.append(encoded_class_text.unsqueeze(0))
        zeroshot_weights = torch.cat(encoded_texts,dim=0).cpu().numpy()
        np.save(savepath + '/' + 'cuplnoreduction_text_encoded_0',
                zeroshot_weights)
        
        # template from openclip
        print('Extracting openclip templates...')
        with torch.no_grad():
            zeroshot_weights = []
            for classname in imagenet_classes:
                texts = [template.format(classname) for template in imagenet_templates] #format with class
                texts = tokenizer(texts).cuda() #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda().T.cpu().numpy()
        np.save(savepath + '/' + 'template_text_encoded_0',
                zeroshot_weights)
        # without mean reduction
        print('Extracting openclip templates without mean...')
        with torch.no_grad():
            zeroshot_weights = []
            for classname in imagenet_classes:
                texts = [template.format(classname) for template in imagenet_templates] #format with class
                texts = tokenizer(texts).cuda() #tokenize
                class_embeddings = model.encode_text(texts) #embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                zeroshot_weights.append(class_embeddings)
            zeroshot_weights=torch.stack(zeroshot_weights, dim=1).cuda().T.cpu().numpy()
        np.save(savepath + '/' + 'templatenoreduction_text_encoded_0',
                zeroshot_weights)


def extract_clip_embeddings(model, dataset, savepath, text=None, k=250):
    torch.backends.cudnn.benchmark = True
    # save
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size)
    features = []
    labels_true = []
    model.eval()
    with torch.no_grad():
        for i, (x, label) in tqdm(enumerate(dataloader)):
            labels_true.append(label)
            x = x.cuda()
            feat_batch = model.encode_image(x)

            feat_batch = feat_batch.cpu().numpy()
            features.append(feat_batch)
            if (i + 1) % k == 0:  # save every k batches
                labels_true = torch.cat(labels_true).numpy()
                features = np.concatenate(features, axis=0)
                predictions_dict = {'features_{}'.format(i): features,
                                    'labels_true_{}'.format(i): labels_true}

                for name, data in predictions_dict.items():
                    np.save(savepath + '/' + name,
                            data)  # with open(savepath + '/' + name, 'wb') as f:  #     pickle.dump(features, f)
                features = []
                labels_true = []

    # save remaining
    labels_true = torch.cat(labels_true).numpy()
    features = np.concatenate(features, axis=0)
    predictions_dict = {'features_{}'.format(0): features, 'labels_true_{}'.format(0): labels_true}
    # text embeddings of labels
    if text is not None:
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text]).cuda()
        with torch.no_grad():
            text_encoded = model.encode_text(text_inputs)
        predictions_dict['text_encoded_0'] = text_encoded.cpu().numpy()

    #

    for name, data in predictions_dict.items():
        np.save(savepath + '/' + name,
                data)


def auroc_ood(values_in: np.ndarray, values_out: np.ndarray) -> float:
    """
    Implementation of Area-under-Curve metric for out-of-distribution detection.
    The higher the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.

    Returns:
        Area-under-curve score.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    y_true = len(values_in) * [1] + len(values_out) * [0]
    y_score = np.nan_to_num(np.concatenate([values_in, values_out]).flatten())
    return roc_auc_score(y_true, y_score)


def fpr_at_tpr(values_in: np.ndarray, values_out: np.ndarray, tpr: float) -> float:
    """
    Calculates the FPR at a particular TRP for out-of-distribution detection.
    The lower the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.
        tpr: (1 - true positive rate), for which probability threshold is calculated for
            in-domain data.

    Returns:
        False positive rate on out-of-domain data at (1 - tpr) threshold.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    t = np.quantile(values_in, (1 - tpr))
    fpr = (values_out >= t).mean()
    return fpr


def set_seed(seed=42):
    # Set the seed for the random module
    random.seed(seed)
    
    # Set the seed for numpy
    np.random.seed(seed)
    
    # Set the seed for torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups

    # Ensure deterministic behavior in some operations, which can slightly impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



