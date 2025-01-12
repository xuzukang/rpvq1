
import math
import os
import numpy as np
import torch
import torch.nn as nn
from contextlib import contextmanager
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import torch,datasets,random,transformers,os
from typing import Any, Dict,List, Tuple
import json

class SimpleDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



def group_texts(block_size, examples):
    
    
    concatenated_examples = {}

    
    for d in examples:
        
        for key in d.keys():
            
            if key not in concatenated_examples:
                concatenated_examples[key] = []
            
            concatenated_examples[key].extend(d[key])
    total_length = len(concatenated_examples["input_ids"])
    
    
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    result = {
        k: [
            t[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_auth_token=hf_token)

    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
        
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
    
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader
def get_llmqat_new(name, nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    cnt = 0
    dataset = []
    #qwen2-7b
    # with open("/data01/home/xuzk/workspace/hmllm_generation_analyse/gen_data/gen.chunk.03.jsonl", encoding='utf-8') as file:
    with open(name, encoding='utf-8') as file:
        for line in file:
            if cnt>= nsamples:
                break
            dataset.append(json.loads(line))
            cnt +=1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    # for batch in tqdm(dataloader):
        # texts = batch["inputs_pretokenized"]
        # queries = [build_prompt(query) for query in texts]
        # inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=seqlen,padding=True).to('cuda')
        # cnt+=1
        # if cnt>= nsamples:
        #     break
    return dataloader
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    model_type = model.split("/")[-1]
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_f = f'{cache_dir}/{name}_{model_type}_{"test" if eval_mode else "train"}_{nsamples}_{seqlen}_{seed}.cache'
    if os.path.exists(cache_f):
        loader = torch.load(cache_f)
        print(f"load loader from {cache_f}")
    else:
        if 'wikitext2' in name:
            loader = get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
        if 'ptb' in name:
            loader = get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
        if 'c4' in name:
            loader = get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
        if "gen_data" in name:
            cache_f = "cache/"+os.path.basename(name[:-6])+".cache"
            loader = get_llmqat_new(name,nsamples, seed, seqlen, model, hf_token, eval_mode)
        torch.save(loader, cache_f)
    return loader
def get_train_eval_dataset(args, tokenizer):
    from datasets import Dataset
    cache_dir = "./cache/" + args.model_name.split("/")[-1]+"/"+"_".join(["tokenized", args.train_dataset])
    if os.path.exists(cache_dir):
        print (cache_dir)
        tokenized_datasets = datasets.load_from_disk(cache_dir)
    else:
        if args.train_dataset == "wikitext2":
            train_dataset = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        elif args.train_dataset =="ceval":
            train_dataset = datasets.arrow_dataset.Dataset.from_json('/data01/home/chenzx/project/hm_spin_smooth/ceval_test_calib_datas.jsonl')
        elif args.train_dataset =="ceval_sum":
            train_dataset = datasets.arrow_dataset.Dataset.from_json('/data01/home/chenzx/project/hm_spin_smooth/test_sum.jsonl')  
        else:
            train_dataset = datasets.arrow_dataset.Dataset.from_json('/data01/home/chenzx/project/hm_spin_smooth/data/ceval_gsm8k_humaneval_sum.jsonl') 
            # train_dataset_ceval = datasets.arrow_dataset.Dataset.from_json('/data01/home/chenzx/project/hm_spin_smooth/data/ceval_sum.jsonl') 
            # train_dataset_humaneval = datasets.arrow_dataset.Dataset.from_json('/data01/home/chenzx/project/hm_spin_smooth/data/HumanEval.jsonl') 
            # train_dataset_gsm8k = datasets.arrow_dataset.Dataset.from_json('/data01/home/chenzx/project/hm_spin_smooth/data/gsm8k_train.jsonl') 
        def tokenize_function(examples):
            return tokenizer(examples["text"])
            # elif examples.has_key('prompt'):
            #     return tokenizer(examples["prompt"])
            # elif examples.has_key('question'):
            #     return tokenizer('question')
            # queries = [build_prompt(query) for query in texts]
            # inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=2048,padding=True)
            # return  F.pad(inputs['input_ids'],(0,2048-inputs['input_ids'].shape[1])) 

        tokenized_datasets = train_dataset.map(
            tokenize_function,
            batched=True,
        )
        grouped_datasets = group_texts(args.ctx_size, tokenized_datasets)
        tokenized_datasets = Dataset.from_dict(grouped_datasets)
        tokenized_datasets.save_to_disk(cache_dir)
    test_loader = get_loaders(
        args.eval_dataset, seed=args.seed, model=args.model_name, seqlen=2048, eval_mode=True
    )
    nsample = test_loader["input_ids"].numel() // 2048
    input_ids = test_loader["input_ids"].reshape(-1)[: nsample * 2048]
    eval_dataset = Dataset.from_dict(dict(input_ids=input_ids.split(2048, dim=-1)))

    def f(examples):
        examples["labels"] = examples["input_ids"]
        return examples

    eval_dataset = eval_dataset.map(f)
    return tokenized_datasets, eval_dataset


def wrap_tokenizer(tokenizer, x, ctx_size, truncate=True):
    return tokenizer(x,
                     return_tensors='pt',
                     truncation=truncate,
                     padding=True,
                     max_length=ctx_size)

def split_data(X, Y, args, split_len=None):
    if split_len is None:
        split = int(len(X) - args.ft_valid_size)
    else:
        split = split_len
    print(f'using {split} training seqs, {len(X) - split} validation seqs')
    train_ds = SimpleDataset(X[:split], Y[:split])
    valid_ds = SimpleDataset(X[split:], Y[split:])
    train_dl = DataLoader(train_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=True)
    valid_dl = DataLoader(valid_ds,
                          batch_size=args.ft_bs,
                          pin_memory=True,
                          shuffle=False)
    return train_dl, valid_dl
