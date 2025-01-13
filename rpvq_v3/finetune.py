import math
import os
from typing import List, Tuple, Any, Dict
import datetime
import copy
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import multiprocessing as mp
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
import time
from operator import attrgetter
from datasets import Dataset
from .loss import *
from .data_utils import *
from .ops import QuantizedLinear
from torch.utils.tensorboard import SummaryWriter

def to_device(model,device):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            module.to(device)
    model.to(device)

def test_layers_output_ppl(model, inds, outputs, dev):
    with torch.no_grad():
        outputs = outputs.to(dev)
        inds = inds.to(dev)
        seqlen = outputs.shape[1]
        nsamples = outputs.shape[0]
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)
        model.lm_head = model.lm_head.to(dev)
        nlls = []
        for i in range(nsamples):
            if model.model.norm is not None:
                hidden_states = model.model.norm(outputs[i:i+1])
            lm_logits = model.lm_head(hidden_states)
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inds[:, (i * seqlen):((i + 1) * seqlen)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).long())
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples*seqlen))
    print("ppl: ",ppl.item()) 

def layerwise_finetune_original(model, quantizers, dataloader, testloader, args):
    if True: #args set
        dtype_ = torch.float32
        nproc = torch.cuda.device_count()
        model.model.cpu()
        global global_writer
        global_writer = SummaryWriter(f'{args.output_dir}/runs/finetune_{args.finetune_type}')
    if True: #data set
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        train_dataset,eval_dataset = dataloader, testloader
        # train_dataset,eval_dataset = get_train_eval_dataset(args,tokenizer)
        # devset = torch.tensor(train_dataset['input_ids'][:args.devset_size*2048])
        devset = torch.tensor(eval_dataset['input_ids'][0,:args.devset_size*2048].reshape(args.devset_size,2048),dtype=torch.int32)
        orig_emb_cache = [model.model.embed_tokens(devset)]

    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                        dtype=orig_emb_cache[0].dtype,
                        device=orig_emb_cache[0].device))

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        orig_emb_cache[0][:args.batch_size], 0)

    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(model.model.layers)):
        print(f'layer {i} gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device][0].join()
            model.model.layers[proc_list[cur_device][1]] = None
            torch.cuda.empty_cache()
            if cur_device == 0:
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            proc_list[cur_device + 1][0].join()
        torch.cuda.empty_cache()
        st = time.time()
        position_ids = position_ids.to(cur_device)
        attention_mask = attention_mask.to(cur_device)
        to_device(model.model.layers[i],cur_device)
        for j in range(args.devset_size // args.batch_size):
            torch.cuda.empty_cache()
            with torch.no_grad():
                orig_emb_cache[cur_device + 1][args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.model.layers[i](
                        orig_emb_cache[cur_device][args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False)[0].cpu().detach()
        model.model.layers[i].cpu()
        position_ids = position_ids.cpu()
        attention_mask = attention_mask.cpu()
        torch.cuda.empty_cache()
        print('computed original embedding for layer {} in {}s'.format(i, time.time() - st))
        if nproc==1:
            if i not in args.finetune_blocks:continue
            quantize_finetune_decoder_layer(model.model.layers[i],
                                            i,
                                            args,
                                            cur_device,
                                            quantizers[i],
                                            orig_emb_cache[cur_device].detach(),
                                            orig_emb_cache[cur_device + 1].detach(),)
        else:
            proc_list[cur_device] = (mp.Process(target=quantize_finetune_decoder_layer,
                                                args=(model.model.layers[i],
                                                    i,
                                                    args,
                                                    cur_device,
                                                    quantizers[i],
                                                    orig_emb_cache[cur_device].detach(),
                                                    orig_emb_cache[cur_device + 1].detach(),
                                                    )), i)
            proc_list[cur_device][0].start()
            cur_device = (cur_device + 1) % nproc

    test_layers_output_ppl(model, devset.reshape(1, -1), orig_emb_cache[cur_device + 1], cur_device)
    
    for p in proc_list:
        if p is not None:
            p[0].join()

def layerwise_finetune(model,quantizers, dataloader, testloader, args):  
    if True: #args set
        dtype_ = torch.float32
        nproc = torch.cuda.device_count()
        model.model.cpu()
        global global_writer 
        global_writer = SummaryWriter(f'{args.output_dir}/runs/finetune_{args.finetune_type}')
        cur_device = 0
    if True: #model set
        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(cur_device)
        # fix for llama-3.1
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to(cur_device)
    if True: #data set
        # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        # tokenizer.pad_token = tokenizer.eos_token
        train_dataset,eval_dataset = dataloader, testloader
        # train_dataset,eval_dataset = get_train_eval_dataset(args,tokenizer)
        # devset = torch.tensor(train_dataset['input_ids'][:args.devset_size*2048])
        devset = torch.tensor(eval_dataset['input_ids'][0,:args.devset_size*2048].reshape(args.devset_size,2048),dtype=torch.int32)
        orig_emb_cache = [model.model.embed_tokens(devset.to(cur_device)).detach().cpu()]

        for _ in range(nproc):
            orig_emb_cache.append(torch.zeros(
                                orig_emb_cache[0].shape,
                                dtype=orig_emb_cache[0].dtype,
                                device=orig_emb_cache[0].device))
        cache = {}
        cache['i'] = 0
        cache["attention_mask"] = None
        cache["position_ids"] = None 
    if True: #catch first origin block data 
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                orig_emb_cache[cur_device][args.batch_size*cache["i"] : args.batch_size*(cache["i"]+1)] = inp.data.cpu()
                cache['i'] += 1
                cache["attention_mask"]= kwargs['attention_mask'][-1:]
                cache["position_ids"] = kwargs['position_ids']
                raise ValueError
        
        layers[0] = Catcher(layers[0])
        for j in range(math.ceil(args.devset_size / args.batch_size)):
            batch = devset[j*args.batch_size:(j+1)*args.batch_size].to(cur_device)
            try:
                model(input_ids=batch)
            except ValueError:
                pass
            
        layers[0] = layers[0].module
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        torch.cuda.empty_cache()
    
    proc_list = [None for _ in range(nproc)]
    for i in range(len(layers)):
        print(f'layer {i} gpu {cur_device}')
        if nproc == 1:
            if i>0:#第一层不用将输出复制给输入，第二层开始需要每次将上一层输出赋值给这一层输入
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        else:
            if proc_list[cur_device] is not None:
                proc_list[cur_device][0].join()
                layers[proc_list[cur_device][1]] = None
                torch.cuda.empty_cache()
                if cur_device == 0:
                    orig_emb_cache[0].copy_(orig_emb_cache[-1])
            if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
                proc_list[cur_device + 1][0].join()
        
        torch.cuda.empty_cache()
        st = time.time()
        position_ids = cache['position_ids'].to(cur_device)
        attention_mask = cache['attention_mask'].to(cur_device)
        to_device(layers[i],cur_device)
        
        for j in range(math.ceil(args.devset_size / args.batch_size)):
            torch.cuda.empty_cache()
            with torch.no_grad():
                orig_emb_cache[cur_device + 1][args.batch_size * j : args.batch_size * (j + 1)] = \
                    layers[i](
                        orig_emb_cache[cur_device][args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False)[0].cpu().detach()
        to_device(layers[i],"cpu")
        position_ids = position_ids.cpu()
        attention_mask = attention_mask.cpu()
        torch.cuda.empty_cache()
        print('computed original embedding for layer {} in {}s'.format(i, time.time() - st))
        
        if nproc==1:
            if i not in args.finetune_blocks:continue
            quantize_finetune_decoder_layer(layers[i],
                                            i,
                                            args,
                                            cur_device,
                                            quantizers[i],
                                            orig_emb_cache[cur_device].detach(),
                                            orig_emb_cache[cur_device + 1].detach(),)
        else:
            proc_list[cur_device] = (mp.Process(target=quantize_finetune_decoder_layer,
                                                args=(model.model.layers[i],
                                                    i,
                                                    args,
                                                    cur_device,
                                                    quantizers[i],
                                                    orig_emb_cache[cur_device].detach(),
                                                    orig_emb_cache[cur_device + 1].detach(),
                                                    )), i)
            proc_list[cur_device][0].start()
            cur_device = (cur_device + 1) % nproc

    # test_layers_output_ppl(model, devset.reshape(1, -1), orig_emb_cache[cur_device + 1], cur_device)
    
    for p in proc_list:
        if p is not None:
            p[0].join()

    if True: #recover model set
        model.config.use_cache = use_cache
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to("cpu")
    

def quantize_finetune_decoder_layer(mixed_layer, idx, args,device, quantizer_layers, pre_orig_emb, orig_emb):
    torch.manual_seed(idx)
    torch.set_num_threads(64)
    torch.set_grad_enabled(False)

    dtype_ =  torch.float32
    orig_dtype = None
    for p in mixed_layer.parameters():
        orig_dtype = p.dtype
        break
    mixed_layer = mixed_layer.float()

    with torch.enable_grad():
        if "layer" in args.finetune_type:
            train_dl, valid_dl = split_data(pre_orig_emb, orig_emb, args, split_len=pre_orig_emb.shape[0])
            finetune_decoder_layer_codebook(mixed_layer,quantizer_layers, device, 
                                        train_dl, valid_dl, orig_dtype, args,idx)
        elif "block" in args.finetune_type:
            train_dl, valid_dl = split_data(pre_orig_emb, orig_emb, args)
            finetune_decoder_block(mixed_layer,quantizer_layers, device,
                                   train_dl, valid_dl, orig_dtype, args,idx)
        else:
            raise ValueError(f"Unsupported finetune_type: {args.finetune_type}")
        mixed_layer = mixed_layer.to(orig_dtype)
        torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

def finetune_decoder_layer_codebook(block, quantizers, device, train_dl, valid_dl, orig_dtype, args, block_idx):
    linear_input = []
    linear_output = []
    def hook_fn(module, input, output):
        linear_input.append(input[0].detach().clone().cpu())  # 获取输入并复制
        linear_output.append(output.detach().clone().cpu())  # 获取输出并复制

    linear_ids=['self_attn.v_proj','self_attn.q_proj', 'self_attn.k_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']
    source = next(iter(train_dl))[0]
    position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
    
    to_device(block,device)

    for linear_id in linear_ids:
        torch.cuda.empty_cache()
        linear = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1])
        quantizer = quantizers[f"{block_idx}.{linear_id}"]
        quantizelinear = QuantizedLinear(linear,quantizer,quant_flag=False,finetune=args.finetune_type).to(device)
        setattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1],quantizelinear)
        hook_handle = quantizelinear.register_forward_hook(hook_fn)
        #根据hook搜集当前层linear的输入输出
        for bidx, (source, targets) in enumerate(train_dl):
            with torch.no_grad():
                output = block(source.to(device),position_ids=position_ids)[0]
        hook_handle.remove()
        inputs = torch.cat(linear_input, dim=0)
        linear_input = []
        outputs = torch.cat(linear_output, dim=0)
        linear_output = []
        linear_train_dl, linear_valid_dl = split_data(inputs, outputs, args)
        torch.cuda.empty_cache()
        #搜集当前linear的codebook
        paramters = {}
        if "lowrank" in args.finetune_type:
            paramters[linear_id+".U"] = quantizelinear.quantizer.U
            # paramters[linear_id+".S"] = quantizelinear.quantizer.S
            paramters[linear_id+".Vt"] = quantizelinear.quantizer.Vt
        if "codebook" in args.finetune_type:
            for key1,val1 in quantizelinear.quantizer.centroids.items():
                if isinstance(val1,dict):
                    for key2,val2 in val1.items():
                        quantizelinear.quantizer.centroids[key1][key2] = nn.Parameter(val2.data,requires_grad=True)
                        paramters[f"{linear_id}.{key1}.{key2}"] = (quantizelinear.quantizer.centroids[key1][key2])
                elif isinstance(val1,torch.Tensor):
                    quantizelinear.quantizer.centroids[key1] = nn.Parameter(val1.data,requires_grad=True)
                    paramters[f"{linear_id}.{key1}"] = quantizelinear.quantizer.centroids[key1]
        
        #获取验证集的当前loss
        quantizelinear.quant_flag = True
        best_loss = calculate_linear_mse_loss(quantizelinear, linear_train_dl, device)
        global_writer.add_scalar(f'Loss/test/block_{block_idx}', best_loss, 0)
        bast_codebook = copy.copy(quantizelinear.quantizer.centroids)
        print(f'block {block_idx} layer {linear_id} initial loss {best_loss}')
        worse_ct = 0
        
        #配置优化器
        optim = torch.optim.Adam(paramters.values(), lr=args.ft_lr)
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16)) #
        
        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(linear_train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=orig_dtype,enabled=True):
                    output = quantizelinear(source.to(device))
                    loss = nn.MSELoss()(output, targets)
                    global_writer.add_scalar(f'Loss/train/block_{block_idx}', loss, epoch*len(linear_train_dl)+bidx)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = calculate_linear_mse_loss(quantizelinear, linear_valid_dl, device)
                global_writer.add_scalar(f'Loss/test/block_{block_idx}', test_loss, epoch)
                if test_loss < best_loss:
                    print(f'block {block_idx} layer {linear_id} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                    best_loss = test_loss
                    bast_codebook = copy.copy(quantizelinear.quantizer.centroids)
                    torch.cuda.empty_cache()
                    worse_ct = 0
                else:
                    print(f'block {block_idx} layer {linear_id} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break
        
        quantizelinear.quantizer.centroids = bast_codebook
    del optim, train_dl, valid_dl

    to_device(block,"cpu")
    torch.cuda.empty_cache()

def finetune_decoder_block_lowrank(block, quantizers, device, train_dl, valid_dl, orig_dtype,args,block_idx):
    linear_ids=['self_attn.v_proj','self_attn.q_proj', 'self_attn.k_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']
    for linear_id in linear_ids:
        linear = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1])
        if isinstance(linear,nn.Linear):
            quantizer = quantizers[linear_id]
            quantizelinear = QuantizedLinear(linear,quantizer,quant_flag=False,finetune="low_rank")
            quantizelinear._smooth_lowrank()
            setattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1],quantizelinear)
        else:
            linear.quant_flag = False
            linear.finetune = "low_rank"
            linear._initialize_quantizer("low_rank")
            linear._smooth_lowrank()
              
    with use_tf32():
        torch.cuda.empty_cache()
        block = block.to(device)
        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        # manifest tensor parallel attributes in layer
        output = block(source.to(device),position_ids=position_ids)[0]
        best_svd = {k:copy.copy(getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer) for k in linear_ids}
        torch.cuda.empty_cache()
        paramters = {}
        for linear_id in linear_ids:
            paramters[linear_id+".U"] = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer.U
            # paramters[linear_id+".S"] = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer.S
            paramters[linear_id+".Vt"] = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer.Vt

        optim = torch.optim.Adam(paramters.values(), lr=args.ft_lr)
        best_loss = calculate_block_mse_loss(block, valid_dl, device)
        global_writer.add_scalar(f'Loss/test/block_{block_idx}', best_loss, 0)
        print(f'{block_idx} block initial valid loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0
        for linear_id in linear_ids:
            getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quant_flag = True
            
        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=orig_dtype,enabled=True):
                    output = block(source.to(device),position_ids=position_ids)[0]
                    loss = nn.MSELoss()(output, targets)
                    global_writer.add_scalar(f'Loss/train/block_{block_idx}', loss, epoch*len(train_dl)+bidx)
                    
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = calculate_block_mse_loss(block, valid_dl, device)
                global_writer.add_scalar(f'Loss/test/block_{block_idx}', loss, epoch)
                if test_loss < best_loss:
                    print(f'block {block_idx} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                    best_loss = test_loss
                    best_svd = {k:copy.copy(getattr(getattr(block,k.split('.')[0]),k.split('.')[1]).quantizer) for k in linear_ids}
                    torch.cuda.empty_cache()
                    worse_ct = 0
                else:
                    print(f'block {block_idx} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    to_device(block,"cpu")
    for linear_id in linear_ids:
        getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer = best_svd[linear_id] 
        
    torch.cuda.empty_cache()

def finetune_decoder_block(block, quantizers, device, train_dl, valid_dl, orig_dtype,args,block_idx):
    linear_ids=['self_attn.v_proj','self_attn.q_proj', 'self_attn.k_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']
    for linear_id in linear_ids:
        linear = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1])
        if isinstance(linear,nn.Linear):
            quantizer = quantizers[linear_id]
            quantizelinear = QuantizedLinear(linear,quantizer,quant_flag=True,finetune=args.finetune_type)
            quantizelinear._smooth_lowrank()
            setattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1],quantizelinear)
        else:
            linear.quant_flag = True
            linear.finetune = args.finetune_type
            linear._initialize_quantizer(args.finetune_type)
            linear._smooth_lowrank()
              
    with use_tf32():
        torch.cuda.empty_cache()
        to_device(block,device)
        source = next(iter(train_dl))[0]
        position_ids = torch.arange(source.shape[1], device=device).unsqueeze(0)
        attention_mask = _prepare_4d_causal_attention_mask(
            None, (args.batch_size, args.ctx_size),source, 0).to(device)
        # manifest tensor parallel attributes in layer
        output = block(source.to(device),
                       position_ids=position_ids,
                       attention_mask=attention_mask,
                       use_cache=False,
                       output_attentions=False)[0]
        best_svd = {k:copy.copy(getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer) for k in linear_ids}
        torch.cuda.empty_cache()
        
        #配置需要训练参数
        paramters = {}
        for linear_id in linear_ids:
            quantizelinear = getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1])
            if "lowrank" in args.finetune_type:
                paramters[linear_id+".U"] = quantizelinear.quantizer.U
                # paramters[linear_id+".S"] = quantizelinear.quantizer.S
                paramters[linear_id+".Vt"] = quantizelinear.quantizer.Vt
            if "codebook" in args.finetune_type:
                for key1,val1 in quantizelinear.quantizer.centroids.items():
                    if isinstance(val1,dict):
                        for key2,val2 in val1.items():
                            quantizelinear.quantizer.centroids[key1][key2] = nn.Parameter(val2.data,requires_grad=True)
                            paramters[f"{linear_id}.{key1}.{key2}"] = (quantizelinear.quantizer.centroids[key1][key2])
                    elif isinstance(val1,torch.Tensor):
                        quantizelinear.quantizer.centroids[key1] = nn.Parameter(val1.data,requires_grad=True)
                        paramters[f"{linear_id}.{key1}"] = quantizelinear.quantizer.centroids[key1]

        optim = torch.optim.Adam(paramters.values(), lr=args.ft_lr)
        with torch.no_grad():
            best_loss = calculate_block_mse_loss(block, valid_dl, device)
        global_writer.add_scalar(f'Loss/test/block_{block_idx}', best_loss, 0)
        
        print(f'{block_idx} block initial valid loss {best_loss}')
        scaler = torch.cuda.amp.GradScaler(enabled=(orig_dtype==torch.float16))
        worse_ct = 0
        for linear_id in linear_ids:
            getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quant_flag = True
            
        for epoch in range(args.ft_epochs):
            for bidx, (source, targets) in enumerate(train_dl):
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type='cuda',dtype=orig_dtype,enabled=True):
                    output = block(source.to(device),position_ids=position_ids)[0]
                    loss = nn.MSELoss()(output, targets)
                    global_writer.add_scalar(f'Loss/train/block_{block_idx}', loss, epoch*len(train_dl)+bidx)
                    
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = calculate_block_mse_loss(block, valid_dl, device)
                global_writer.add_scalar(f'Loss/test/block_{block_idx}', loss, epoch+1)
                
                if test_loss < best_loss:
                    print(f'block {block_idx} @ epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                    best_loss = test_loss
                    best_svd = {k:copy.copy(getattr(getattr(block,k.split('.')[0]),k.split('.')[1]).quantizer) for k in linear_ids}
                    torch.cuda.empty_cache()
                    worse_ct = 0
                else:
                    print(f'block {block_idx} @ epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                    worse_ct += 1
                    if worse_ct >= args.ft_early_stop:
                        break

    del optim, train_dl, valid_dl

    to_device(block,"cpu")
    for linear_id in linear_ids:
        getattr(getattr(block,linear_id.split('.')[0]),linear_id.split('.')[1]).quantizer = best_svd[linear_id] 
        
    torch.cuda.empty_cache()


def infer(args, end_dev, n_layers, in_q, out_q):
    with torch.no_grad():
        fake_dev_map = {
            'model.embed_tokens': 0,
            'model.rotary_emb': 0,
            'model.norm': end_dev - 1,
            'lm_head': end_dev - 1
        }
        per_dev = math.ceil(n_layers / end_dev)
        for i in range(n_layers):
            fake_dev_map[f'model.layers.{i}'] = (i + 1) // per_dev

        model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                     torch_dtype='auto',
                                                     device_map=fake_dev_map,
                                                     low_cpu_mem_usage=True)
        while True:
            data = in_q.get()
            if data is None:
                return
            out_q.put(
                model(data.to(0))['logits'][:, :-1].contiguous().softmax(dim=-1).cpu())

def e2e_finetune(quant_model, start_dev, devset, orig_dtype, args):

    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=infer,args=(args, start_dev, len(quant_model.model.layers), in_q,out_q))
    p.start()

    train_dl, valid_dl = split_data(devset, devset, args)

    optim = torch.optim.Adam(quant_model.parameters(), lr=args.ft_lr)

    best_loss = calculate_ce_loss_model(quant_model, valid_dl, start_dev,in_q, out_q)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_sd = copy.deepcopy(quant_model.state_dict())
    print(f'initial loss {best_loss}')
    worse_ct = 0
    for epoch in range(args.ft_epochs):
        for bidx, (source, _) in enumerate(train_dl):
            in_q.put(source)
            with torch.autocast(device_type='cuda',dtype=orig_dtype,enabled=True):
                output = quant_model(source.to(start_dev))['logits'][:, :-1].contiguous()
                target = out_q.get().to(output.device)
                target = target.view(-1, target.shape[-1])
                loss = nn.CrossEntropyLoss()(output.view(-1, output.shape[-1]),target)
                global_writer.add_scalar(f'Loss/train/e2e', loss, epoch*len(train_dl)+bidx)
                
            scaler.scale(loss).backward()
            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            test_loss = calculate_ce_loss_model(quant_model, valid_dl,start_dev, in_q, out_q)
            global_writer.add_scalar(f'Loss/test/e2e', loss, epoch)
            if test_loss < best_loss:
                print(f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                best_loss = test_loss
                best_sd = copy.deepcopy(quant_model.state_dict())
                worse_ct = 0
            else:
                print(f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    in_q.put(None)
    p.join()
    with torch.no_grad():
        quant_model.load_state_dict(best_sd)



