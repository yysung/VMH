# -*- coding: utf-8 -*-
"""
Created on Wed May  4 21:10:11 2022

@author: YY
"""


import os
import argparse
import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mmpt.utils import load_config, set_seed
from mmpt.evaluators import Evaluator
from mmpt.evaluators import predictor as predictor_path
from mmpt.tasks import Task
from mmpt import processors
from mmpt.datasets import MMDataset
from copy import deepcopy
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score, roc_auc_score, average_precision_score


setting = 'videoclip_V'

npys = os.listdir('data/vmh/feat')
for f in npys:
    if f.endswith(' .npy'):
        f_ = f.replace(' .npy', '.npy')
        os.rename(f'data/vmh/feat{f}', f'data/vmh/feat{f_}')
        
        #print(f)

def get_dataloader(config, df, train=True):
    meta_processor_cls = getattr(processors, config.dataset.meta_processor)
    video_processor_cls = getattr(processors, config.dataset.video_processor)
    text_processor_cls = getattr(processors, 'TextProcessor')
    aligner_cls = getattr(processors, config.dataset.aligner)

    meta_processor = meta_processor_cls(config.dataset)
    video_processor = video_processor_cls(config.dataset)
    text_processor = text_processor_cls(config.dataset)
    aligner = aligner_cls(config.dataset)

    test_data = MMDataset(
        meta_processor,
        video_processor,
        text_processor,
        aligner,
    )
    #print("test_len", len(test_data))
    #output = test_data[0]
    #test_data.print_example(output)

    if train:
        dataloader = DataLoader(
            test_data,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            collate_fn=test_data.collater,
        )
    else:
        dataloader = DataLoader(
            test_data,
            batch_size=16,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=test_data.collater,
        )
    return dataloader

#parser = argparse.ArgumentParser()
#parser.add_argument("--taskconfig", type=str, default='projects/retri/videoclip/test_youcook_videoclip.yaml')
args = argparse.Namespace
#args.taskconfig = 'projects/retri/videoclip/test_youcook_videoclip.yaml'
args.taskconfig = 'projects/retri/videoclip/vmh_videoclip.yaml'
config = load_config(args)
mmtask = Task.config_task(config)
mmtask.build_model()

dat_ = pd.read_csv('./2247_features_UPDATE.csv',index_col=0)

#headlines
dat_['headline'] = [''.join(i for i in h["headline"] if i not in "\/:*?<>|\"\'") for _, h in dat_.iterrows()]
print('headlines:', len((dat_['headline']).unique()))
#transcripts
dat_['transcripts'] = dat_['transcripts'].replace(np.nan, '')
#rationales
dat_['rationales'] = dat_['rationales'].replace(np.nan, '')
#videos
video_files = os.listdir('data/vmh/feat')
video_files = [f for f in video_files if f.endswith('.npy')]
headlines_with_video = [f.replace('.npy', '') for f in video_files]
dat_ = dat_[dat_['headline'].isin(headlines_with_video)]

import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.encoder = mmtask.load_checkpoint(config.fairseq.common_eval.path)
        self.linear = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.ReLU(),
            nn.Linear(768, 768//2),
            nn.ReLU(),
            nn.Linear(768//2, 2)
            )
        
    def forward(self, batch):
        encoder_out = self.encoder(**batch)
                
        concat_feature = torch.cat([
            encoder_out['pooled_video'],
            encoder_out['pooled_text']],
            axis=1)
        
        #concat_feature = encoder_out['pooled_video'] + encoder_out['pooled_text']
        
        return self.linear(concat_feature)
        

from sklearn.model_selection import train_test_split
import datetime
print(datetime.datetime.now())

cm_stack, auc_stack, acc_stack, f1_stack, w_f1_stack, rec_stack, pre_stack, avg_pre_stack = [], [], [], [], [], [], [], []

test_size = int(len(dat_)*.15)
for random_seed in np.arange(5):
    np.random.seed(random_seed)
    idx = np.random.choice(np.arange(len(dat_)), len(dat_), replace=False)

    tr_val_idx, te_idx = train_test_split(np.arange(len(idx)), test_size=test_size, stratify=dat_['majority_answer'].values)
    tr_idx, val_idx = train_test_split(tr_val_idx, test_size=test_size, stratify=dat_['majority_answer'].values[tr_val_idx])
    

    train_headlines = [dat_['headline'].iloc[i] for i in tr_idx]
    val_headlines = [dat_['headline'].iloc[i] for i in val_idx]
    test_headlines = [dat_['headline'].iloc[i] for i in te_idx]

    print('train:',len(train_headlines), 'val:', len(val_headlines), 'test:', len(test_headlines))

    with open(f'data/vmh/vmh_train_{setting}.lst', 'w', encoding='utf-8') as f:
        for head in train_headlines:
            f.write(head + '\n')
            
    with open(f'data/vmh/vmh_val_{setting}.lst', 'w', encoding='utf-8') as f:
        for head in val_headlines:
            f.write(head + '\n')
            
    with open(f'data/vmh/vmh_test_{setting}.lst', 'w', encoding='utf-8') as f:
        for head in test_headlines:
            f.write(head + '\n')

    config['dataset']['train_path'] = f'data/vmh/vmh_train_{setting}.lst'
    config['dataset']['split'] = 'train'
    train_dataloader = get_dataloader(config, dat_)

    config['dataset']['val_path'] = f'data/vmh/vmh_val_{setting}.lst'
    config['dataset']['split'] = 'valid'
    val_dataloader = get_dataloader(config, dat_, train=False)

    config['dataset']['test_path'] = f'data/vmh/vmh_test_{setting}.lst'
    config['dataset']['split'] = 'test'
    test_dataloader = get_dataloader(config, dat_, train=False)
    
    clf = Classifier(config).to('cuda')
    clf.train()
    
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(params=clf.parameters(), lr=2e-5, weight_decay=1e-3)
    
    batch = next(iter(train_dataloader))
    
    best_model_weights = deepcopy(clf.state_dict())
    best_val_loss = np.inf
    best_val_f1 = 0.
    best_val_epoch = 0.
    for i in range(10):
        train_loss = 0.
        clf.train()
        for batch in train_dataloader:
            opt.zero_grad()
            labels = [dat_.iloc[np.where(dat_['headline'] == txt)[0]]['majority_answer'].values[0] for txt in batch['video_id']]
            labels = torch.tensor([0 if label == 'representative' else 1 for label in labels], dtype=torch.long)
            labels = labels.to('cuda')
                        
            for key in ['caps', 'cmasks', 'vfeats', 'vmasks']:
                batch[key] = batch[key].to('cuda')

            for row in batch['caps']:
                row[:] = 0
                
            for row in batch['cmasks']:
                row[:] = True
            
            batch_logit = clf(batch)
            loss = loss_fn(batch_logit, labels)
            
            loss.backward()
            opt.step()
            
            train_loss += loss.item() / len(train_dataloader)

        clf.eval()
        val_loss = 0.
        val_label, val_pred = [], []
        for batch in val_dataloader:
            labels = [dat_.iloc[np.where(dat_['headline'] == txt)[0]]['majority_answer'].values[0] for txt in batch['video_id']]
            labels = torch.tensor([0 if label == 'representative' else 1 for label in labels], dtype=torch.long)
            labels = labels.to('cuda')
            
            for key in ['caps', 'cmasks', 'vfeats', 'vmasks']:
                batch[key] = torch.squeeze(batch[key].to('cuda'), dim=1)
                
            for row in batch['caps']:
                row[:] = 0
                
            for row in batch['cmasks']:
                row[:] = True
                
            batch_logit = clf(batch)
    
            loss = loss_fn(batch_logit, labels)
            labels = labels.detach().cpu().numpy()
            batch_pred = batch_logit.detach().cpu().numpy().argmax(1)
            
            val_loss += loss.item() / len(val_dataloader)
            val_label.append(labels)
            val_pred.append(batch_pred)
        
        val_f1 = f1_score(np.concatenate(val_label), np.concatenate(val_pred))
        print(f'Epoch {i:03d} | train_loss: {train_loss:.3f} / val_loss: {val_loss:.3f} / val_f1: {val_f1:.3f}')
        if best_val_f1 < val_f1:
            best_model_weights = deepcopy(clf.state_dict())
            best_val_f1 = val_f1
            best_val_epoch = i
        
    print(f'Best val epoch {best_val_epoch:03d} val f1 {best_val_f1:.3f}')
    clf.load_state_dict(best_model_weights)
    clf.eval()
    
    test_pred = []
    test_prob = []
    test_true = []
    test_info = []
    test_headline = []

    test_h = []
    for batch in test_dataloader:
        rows = [dat_.iloc[np.where(dat_['headline'] == txt)[0][0]] for txt in batch['video_id']]
        labels = [dat_.iloc[np.where(dat_['headline'] == txt)[0]]['majority_answer'].values[0] for txt in batch['video_id']]
        labels = torch.tensor([0 if label == 'representative' else 1 for label in labels], dtype=torch.long)
        for key in ['caps', 'cmasks', 'vfeats', 'vmasks']:
            batch[key] = batch[key].to('cuda')
    
        test_true.append(labels.cpu().numpy())
        test_h.append(batch['video_id'])
        test_info += rows
        
        batch_logit = clf(batch)
        batch_pred = batch_logit.detach().cpu().numpy().argmax(1)
        batch_prob = torch.softmax(batch_logit, 1)[:,1].detach().cpu().numpy()
        
        test_pred.append(batch_pred)
        test_prob.append(batch_prob)
    
    test_prob = np.concatenate(test_prob)    
    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)
    test_headline.append(test_h)
    
    test_concat = pd.concat([pd.DataFrame(sum(test_headline[0],[])),
                          pd.DataFrame(test_prob),
                          pd.DataFrame(test_pred),
                          pd.DataFrame(test_true)],axis=1) 

    test_concat.columns = ['headline','prob','pred','true']
    test_concat.to_csv(f'yield/{setting}.csv')

    print(datetime.datetime.now())
    cm_stack.append(confusion_matrix(test_true, test_pred))
    print('confusion_matrix \n', cm_stack[-1])
    
    auc_stack.append(roc_auc_score(test_true, test_prob))
    print('roc_auc_score', auc_stack[-1])
    
    acc_stack.append(accuracy_score(test_true, test_pred))
    print('accuracy', acc_stack[-1])
    
    f1_stack.append(f1_score(test_true, test_pred))
    print('f1', f1_stack[-1])
    
    w_f1_stack.append(f1_score(test_true, test_pred,average='weighted'))
    print('w_f1', w_f1_stack[-1])

    rec_stack.append(recall_score(test_true, test_pred))
    print('recall', rec_stack[-1])
    pre_stack.append(precision_score(test_true, test_pred))
    print('precision', pre_stack[-1])
    avg_pre_stack.append(average_precision_score(test_true, test_pred))
    print('auprc', avg_pre_stack[-1])

    print('='*20, 'final', '='*20)
    print('average confusion matrix \n', np.mean(cm_stack, axis=0))
    print('average auroc', np.mean(auc_stack), np.std(auc_stack))
    print('average acc', np.mean(acc_stack), np.std(acc_stack))
    print('average f1', np.mean(f1_stack), np.std(f1_stack))
    print('average weighted f1', np.mean(w_f1_stack), np.std(w_f1_stack))
    print('average recall', np.mean(rec_stack), np.std(rec_stack))
    print('average precision', np.mean(pre_stack), np.std(pre_stack))
    print('average auprc', np.mean(avg_pre_stack), np.std(avg_pre_stack))
'''
    test_pred = []
    test_prob = []
    test_true = []
    test_info = []
    for batch in test_dataloader:
        rows = [dat_.iloc[np.where(dat_['headline'] == txt)[0][0]] for txt in batch['video_id']]
        labels = [dat_.iloc[np.where(dat_['headline'] == txt)[0]]['majority_answer'].values[0] for txt in batch['video_id']]
        labels = torch.tensor([0 if label == 'leading' else 1 for label in labels], dtype=torch.long)
        for key in ['caps', 'cmasks', 'vfeats', 'vmasks']:
            batch[key] = btch[key].to('cuda')
    
        test_true.append(labels)
        test_info += rows
        
        batch_logit = clf(batch)
        batch_pred = batch_logit.detach().cpu().numpy().argmax(1)
        batch_prob = torch.softmax(batch_logit, 1)[:,1].detach().cpu().numpy()
        
        test_pred.append(batch_pred)
        test_prob.append(batch_prob)
    
    test_prob = np.concatenate(test_prob)    
    test_pred = np.concatenate(test_pred)
    test_true = np.concatenate(test_true)
    
    print(datetime.datetime.now())
    cm_stack.append(confusion_matrix(test_true, test_pred))
    print('confusion_matrix \n', cm_stack[-1])
    
    auc_stack.append(roc_auc_score(test_true, test_prob))
    print('roc_auc_score', auc_stack[-1])
    
    acc_stack.append(accuracy_score(test_true, test_pred))
    print('accuracy', acc_stack[-1])
    
    f1_stack.append(f1_score(test_true, test_pred))
    print('f1', f1_stack[-1])
    
    rec_stack.append(recall_score(test_true, test_pred))
    print('recall', rec_stack[-1])
    pre_stack.append(precision_score(test_true, test_pred))
    print('precision', pre_stack[-1])
    
print('='*20, 'final', '='*20)
print('average confusion matrix \n', np.mean(cm_stack, axis=0))
print('average auroc', np.mean(auc_stack), np.std(auc_stack))
print('average acc', np.mean(acc_stack), np.std(acc_stack))
print('average f1', np.mean(f1_stack), np.std(f1_stack))
print('average recall', np.mean(rec_stack), np.std(rec_stack))
print('average precision', np.mean(pre_stack), np.std(pre_stack))

'''
