# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 22:57:04 2022

@author: HR
"""

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
#data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

import os
import numpy as np
import pandas as pd
#os.chdir('d:/videoclip/fairseq/examples/MMPT/data/feat/vhm_s3d')
#files = [file for file in os.listdir() if file.endswith('.npy')]


dat_ = pd.read_csv('./2247_features_UPDATE.csv',index_col=0)

#headline
dat_['headline'] = [''.join(i for i in h["headline"] if i not in "\/:*?<>|\"\'") for _, h in dat_.iterrows()]
print('headlines:', len((dat_['headline']).unique()))
#transcripts
dat_['transcripts'] = dat_['transcripts'].replace(np.nan, '')
#rationales
dat_['rationales'] = dat_['rationales'].replace(np.nan, '')

from sklearn.model_selection import train_test_split

cm_stack, auc_stack, acc_stack, f1_stack, w_f1_stack, rec_stack, pre_stack = [], [], [], [], [], [], []

test_size = int(len(dat_)*.15)
for random_seed in np.arange(5):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)


    np.random.seed(random_seed)
    idx = np.random.choice(np.arange(len(dat_)), len(dat_), replace=False)
    
    tr_val_idx, te_idx = train_test_split(np.arange(len(idx)), test_size=test_size, stratify=dat_['majority_answer'].values)
    tr_idx, val_idx = train_test_split(tr_val_idx, test_size=test_size, stratify=dat_['majority_answer'].values[tr_val_idx])
    
    n_train = len(tr_idx)
    n_val = len(val_idx)
    n_test = len(te_idx)
    
    train_dat = [
        dict(zip(['text', 'label'],
            (dat_.iloc[i]['headline'] + ' [SEP] ' + dat_.iloc[i]['transcripts'] + ' [SEP] ' + dat_.iloc[i]['rationales'] if dat_.iloc[i]['transcripts'] != '' else dat_.iloc[i]['headline'],
             dat_.iloc[i]['majority_answer']))) for i in tr_idx]
    
    val_dat = [
        dict(zip(['text', 'label'],
            (dat_.iloc[i]['headline'] + ' [SEP] ' + dat_.iloc[i]['transcripts'] + ' [SEP] ' + dat_.iloc[i]['rationales'] if dat_.iloc[i]['transcripts'] != '' else dat_.iloc[i]['headline'],
             dat_.iloc[i]['majority_answer']))) for i in val_idx]
    
    test_dat = [
        dict(zip(['text', 'label'],
            (dat_.iloc[i]['headline'], dat_.iloc[i]['majority_answer']))) for i in te_idx]
    
    print(len(train_dat), len(val_dat), len(test_dat))
    
    for k in train_dat:
        k['label'] = 0 if k['label'] == 'representative' else 1
        k.update(tokenizer(k["text"], truncation=True, padding=True))
        del k['text']
        
    for k in val_dat:
        k['label'] = 0 if k['label'] == 'representative' else 1
        k.update(tokenizer(k["text"], truncation=True, padding=True))
        del k['text']
        
    for k in test_dat:
        k['label'] = 0 if k['label'] == 'representative' else 1
        k.update(tokenizer(k["text"], truncation=True, padding=True))
        del k['text']
        
        
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dat,
        eval_dataset=val_dat,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    trainer.train()
    
    model.eval()
    pred_probs = trainer.predict(test_dat)[0][:,1]
    pred_labels = trainer.predict(test_dat)[0].argmax(1)
    true_labels = np.array([t['label'] for t in test_dat])
    
    from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    cm_stack.append(confusion_matrix(true_labels, pred_labels))
    print('confusion_matrix \n', cm_stack[-1])
    
    auc_stack.append(roc_auc_score(true_labels, pred_probs))
    print('roc_auc_score', auc_stack[-1])
    
    acc_stack.append(accuracy_score(true_labels, pred_labels))
    print('accuracy', acc_stack[-1])
    
    f1_stack.append(f1_score(true_labels, pred_labels))
    print('f1', f1_stack[-1])
    
    w_f1_stack.append(f1_score(true_labels, pred_labels, average='weighted'))
    print('f1', w_f1_stack[-1])
    
    rec_stack.append(recall_score(true_labels, pred_labels))
    print('recall', rec_stack[-1])
    pre_stack.append(precision_score(true_labels, pred_labels))
    print('precision', pre_stack[-1])

print('average confusion matrix \n', np.mean(cm_stack, axis=0))
print('average auroc', np.mean(auc_stack), np.std(auc_stack))
print('average acc', np.mean(acc_stack), np.std(acc_stack))
print('average f1', np.mean(f1_stack), np.std(f1_stack))
print('average weighted f1', np.mean(w_f1_stack), np.std(w_f1_stack))
print('average recall', np.mean(rec_stack), np.std(rec_stack))
print('average precision', np.mean(pre_stack), np.std(pre_stack))

