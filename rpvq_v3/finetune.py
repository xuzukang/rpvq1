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
from threading import Thread
import queue
from queue import Queue
import torch.utils.checkpoint as checkpoint

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

def finetune(model,quantizers, dataloader, testloader, args):  
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
        devset = torch.tensor(eval_dataset['input_ids'][0,:args.devset_size*args.ctx_size].reshape(args.devset_size,args.ctx_size),dtype=torch.int32)
        orig_emb_cache = [model.model.embed_tokens(devset.to(cur_device)).detach().cpu()]

        for _ in range(nproc):
            orig_emb_cache.append(torch.zeros(
                                orig_emb_cache[0].shape,
                                dtype=orig_emb_cache[0].dtype,
                                device=orig_emb_cache[0].device))
        cache = {"i": 0, "attention_mask": None, "position_ids": None}
        
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
        if i>0:
            orig_emb_cache[0].copy_(orig_emb_cache[-1])
        
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
        
        
        if i not in args.finetune_blocks:continue
        finetune_decoder(layers[i],
                        i,
                        args,
                        cur_device,
                        quantizers[i],
                        orig_emb_cache[cur_device].detach(),
                        orig_emb_cache[cur_device + 1].detach(),)

    if True: #recover model set
        model.config.use_cache = use_cache
        if hasattr(model.model, 'rotary_emb'):
            model.model.rotary_emb = model.model.rotary_emb.to("cpu")
    
def finetune_decoder(mixed_layer, idx, args,device, quantizer_layers, pre_orig_emb, orig_emb):
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
            finetune_decoder_layer(mixed_layer,quantizer_layers, device, 
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

def finetune_decoder_layer(block, quantizers, device, train_dl, valid_dl, orig_dtype, args, block_idx):
    linear_input = []
    linear_output = []
    def hook_fn(module, input, output):
        linear_input.append(input[0].detach().clone().cpu())  # 获取输入并复制
        linear_output.append(output.detach().clone().cpu())   # 获取输出并复制

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
        best_loss = calculate_linear_loss(quantizelinear, linear_train_dl, device, args.ft_loss_type)
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
                    if args.ft_loss_type == "mse":
                        loss = nn.MSELoss()(output, targets)
                    elif args.ft_loss_type == "kl":
                        loss = nn.KLDivLoss(reduction='batchmean')(
                            F.log_softmax(output, dim=-1),
                            F.softmax(targets, dim=-1))
                    else:
                        raise ValueError(f"Unsupported loss_type: {args.loss_type}")
                    
                    loss = nn.MSELoss()(output, targets)
                    global_writer.add_scalar(f'Loss/train/block_{block_idx}', loss, epoch*len(linear_train_dl)+bidx)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = calculate_linear_loss(quantizelinear, linear_valid_dl, device, args.ft_loss_type)
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
            best_loss = calculate_block_loss(block, valid_dl, device, loss_type=args.ft_loss_type)
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
                    if args.ft_loss_type == "mse":
                        loss = nn.MSELoss()(output, targets)
                    elif args.ft_loss_type == "kl":
                        loss = nn.KLDivLoss(reduction='batchmean')(
                            F.log_softmax(output, dim=-1),
                            F.softmax(targets, dim=-1))
                    else:
                        raise ValueError(f"Unknown loss type: {args.ft_loss_type}")
                    global_writer.add_scalar(f'Loss/train/block_{block_idx}', loss, epoch*len(train_dl)+bidx)
                scaler.scale(loss).backward()
                if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
            if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
                test_loss = calculate_block_loss(block, valid_dl, device, args.ft_loss_type)
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

            
def infer_(model, in_q: Queue, out_q: Queue, device: str):
    """
    推理函数，逐个 block 处理模型并动态管理 GPU 内存。

    参数：
    - model: 要进行推理的模型实例。
    - in_q: 输入队列，包含输入数据（如 input_ids）。
    - out_q: 输出队列，用于存放推理结果（概率分布）。
    - device: 目标设备（如 'cuda:0'）。
    """
    cache = {"attention_mask": None, "position_ids": None}
    cache  # 使 Catcher 可以访问并修改 cache
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            orig_emb_cache[cur_device][args.batch_size*cache["i"] : args.batch_size*(cache["i"]+1)] = inp.data.cpu()
            cache["attention_mask"] = kwargs['attention_mask'][-1:]
            cache["position_ids"] = kwargs['position_ids']
            raise ValueError

    
    model.eval()  # 设置模型为评估模式


    # 插入 Catcher 到第一个 block
    first_block = model.model.layers[0]
    catcher = Catcher(first_block)
    model.model.layers[0] = catcher

    with torch.no_grad():  # 禁用梯度计算
        while True:
            data = in_q.get()  # 从输入队列获取数据
            if data is None:
                break  # 如果收到 None，退出循环
            input_ids = data.to(device)  # 将输入数据移动到 GPU

            # 创建 attention mask 和 position_ids（根据模型需求调整）
            attention_mask = torch.ones_like(input_ids).to(device)
            position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids)

            # 获取嵌入层的输出
            embed_tokens = model.model.embed_tokens
            embed_tokens.to(device)  # 将嵌入层移动到 GPU
            hidden_states = embed_tokens(input_ids)  # [batch_size, seq_len, hidden_size]
            embed_tokens.to('cpu')  # 处理完毕后移动回 CPU
            torch.cuda.empty_cache()

            # 如果模型有嵌入层归一化，应用它
            if hasattr(model.model, 'embed_layer_norm'):
                embed_layer_norm = model.model.embed_layer_norm
                embed_layer_norm.to(device)
                hidden_states = embed_layer_norm(hidden_states)
                embed_layer_norm.to('cpu')
                torch.cuda.empty_cache()

            # 运行第一个 block（Catcher 会捕获 attention_mask 和 position_ids）
            catcher.to(device)
            try:
                hidden_states = catcher(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]
            except ValueError:
                pass  # Catcher 中断前向传递以捕获数据
            catcher.to('cpu')
            hidden_states = hidden_states.to('cpu')  # 将隐藏状态移动回 CPU
            torch.cuda.empty_cache()

            # 确保 cache 已被填充
            if cache["attention_mask"] is None or cache["position_ids"] is None:
                raise ValueError("Catcher failed to capture attention_mask or position_ids")

            # 逐个 block 处理（从第二个 block 开始）
            for block in model.model.layers[1:]:
                block.to(device)  # 将当前 block 移动到 GPU
                hidden_states = block(hidden_states.to(device), 
                                      attention_mask=cache["attention_mask"].to(device), 
                                      position_ids=cache["position_ids"].to(device))[0]
                block.to('cpu')  # 处理完毕后移动回 CPU
                hidden_states = hidden_states.to('cpu')  # 将隐藏状态移动回 CPU
                torch.cuda.empty_cache()

            # 如果模型有最终层归一化，应用它
            if hasattr(model.model, 'norm'):
                norm = model.model.norm
                norm.to(device)
                hidden_states = norm(hidden_states.to(device))
                norm.to('cpu')
                hidden_states = hidden_states.to('cpu')
                torch.cuda.empty_cache()

            # 计算 logits
            lm_head = model.lm_head
            lm_head.to(device)
            hidden_states = hidden_states.to(device)  # 将隐藏状态移动到 GPU
            logits = lm_head(hidden_states)  # [batch_size, seq_len, vocab_size]
            lm_head.to('cpu')  # 处理完毕后移动回 CPU
            hidden_states = hidden_states.to('cpu')  # 隐藏状态移动回 CPU
            torch.cuda.empty_cache()

            # 计算概率分布
            probabilities = logits.softmax(dim=-1).contiguous().cpu()  # 转移到 CPU
            out_q.put(probabilities)  # 将结果放入输出队列
            torch.cuda.empty_cache()  # 清理 GPU 缓存

    # 恢复第一个 block
    model.model.layers[0] = catcher.module  
            
def checkpoint_forward(module, *inputs):
    """
    用于包装模型的前向传递，允许checkpointing。
    """
    return (module, *inputs)

def infer(model, in_q, out_q, device):
    """
    推理函数，使用传入的模型实例。同步版本，去掉了多线程。
    """
    model.eval()
    while True:
        data = in_q.get()
        if data is None:
            break
        data = data.to(device)
        logits = model(data)['logits']
        probabilities = logits.softmax(dim=-1).contiguous().cpu()
        out_q.put(probabilities)
        torch.cuda.empty_cache()  # 每次推理后清理缓存

def e2e_finetune(model, dataloader, testloader, args):
    # model = model.half()
    main_device = 'cuda:0'

    to_device(model, main_device)
    torch.cuda.empty_cache()

    orig_dtype = model.lm_head.weight.dtype
    global_writer = SummaryWriter(f'{args.output_dir}/runs/finetune_{args.finetune_type}')

    # 初始化模型
    for block in model.model.layers:
        linear_ids = ['self_attn.v_proj', 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']
        for linear_id in linear_ids:
            proj_module, proj_attr = linear_id.split('.')
            linear = getattr(getattr(block, proj_module), proj_attr)
            if isinstance(linear,QuantizedLinear):
                linear.quant_flag = True
                linear.finetune = args.finetune_type
                linear._initialize_quantizer(args.finetune_type)
                # linear._smooth_lowrank()
                linear.remove_codebook_lowrank()
    
    parameters = {}

    for name, param in model.named_parameters():
        if "norm" in name or "head" in name:
            param.requires_grad = True
            parameters[name] = param
        else:
            param.requires_grad = False

    train_dataset, eval_dataset = dataloader, testloader
    devset = torch.zeros([1,args.ctx_size]).to(torch.int32)
    devset_test = eval_dataset['input_ids'][0, :args.devset_size * args.ctx_size].reshape(args.devset_size, args.ctx_size).to(torch.int32)
    for i in train_dataset:
        devset = torch.cat([devset, i[0].to(torch.int32)], dim=0)
    devset = torch.cat([devset, devset_test], dim=0)[1:]

    optim = torch.optim.Adam(parameters.values(), lr=args.ft_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    train_dl, valid_dl = split_data(devset[:, :-1], devset[:, 1:], args)
    best_loss = calculate_ce_loss_model(model, valid_dl, main_device)
    global_writer.add_scalar(f'Loss/test', best_loss, 0)
    print(f'initial loss {best_loss}')
    best_sd = copy.deepcopy(model.cpu())
    torch.cuda.empty_cache()
    to_device(model, main_device)

    worse_ct = 0
    for epoch in range(args.ft_epochs):
        model.train()
        for bidx, (source, target) in enumerate(train_dl):
            source = source.cuda()  # 确保输入数据是GPU上的张量
       
            with torch.cuda.amp.autocast(dtype=orig_dtype):
                logits = model(source)['logits'].contiguous()
                loss = nn.CrossEntropyLoss()(logits.view(-1, logits.shape[-1]), target.view(-1).cuda().long())
            global_writer.add_scalar(f'Loss/train', loss.item(), epoch * len(train_dl) + bidx)

            # 使用 scaler 进行反向传播
            # loss.backward()
            scaler.scale(loss).backward()

            if bidx % args.ft_update_freq == args.ft_update_freq - 1 or bidx == len(train_dl) - 1:
                # optim.step()
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                torch.cuda.empty_cache()

        if epoch % args.ft_valid_freq == (args.ft_valid_freq - 1):
            model.eval()
            test_loss = calculate_ce_loss_model(model, valid_dl, main_device)
            global_writer.add_scalar(f'Loss/test', test_loss, epoch + 1)
            if test_loss < best_loss:
                print(f'epoch {epoch} new loss {test_loss} old loss {best_loss} BETTER')
                best_loss = test_loss
                best_sd = copy.deepcopy(model.cpu())
                torch.cuda.empty_cache()
                to_device(model, main_device)
                worse_ct = 0
            else:
                print(f'epoch {epoch} new loss {test_loss} old loss {best_loss} WORSE')
                worse_ct += 1
                if worse_ct >= args.ft_early_stop:
                    break

    # 使用最佳模型
    model = to_device(best_sd, main_device)
    torch.cuda.empty_cache()