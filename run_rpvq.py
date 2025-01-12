# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r".*pyarrow.*size changed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="cudf.utils._numba")


import json
import os.path as osp
import time
from dataclasses import dataclass, field
from typing import Optional
import copy
from sentence_transformers import SentenceTransformer
import torch
from torch.multiprocessing import set_start_method
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from vptq.models.llama import eval_llama, get_llama, quant_llama
from vptq.models.mistral import get_mistral
from vptq.models.qwen import eval_qwen, get_qwen, quant_qwen
from vptq.models.nvembed import get_nvembed, quant_nvembed
from vptq.quantizer import QuantizationArguments
from vptq.utils.data import get_data_loader
from vptq.utils.pack import pack_model

import os,sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDNN_DETERMINISTIC"] = "1"
from datetime import datetime
from typing import List, Tuple
from rpvq_v3.finetune import to_device

class Logger(object):
    def __init__(self, folder="logs"):
        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # 创建日志文件夹（如果不存在）
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        # 定义日志文件名
        filename = os.path.join(folder, f"log_{current_time}.txt")
        
        # 打开日志文件
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()
        

@dataclass
class VPTQArguments:
    model_name: str = field(default="meta-llama/Llama-2-7b-hf")
    seq_len: Optional[int] = field(default=None)
    quant_step: int = field(default=1)
    percdamp: float = field(default=0.01)
    blocksize: int = field(default=128)
    output_dir: str = field(default="outputs")
    seed: int = field(default=0)
    eval: bool = field(default=False)
    new_eval: bool = field(default=False)
    save_model: bool = field(default=False)
    save_packed_model: bool = field(default=False)
    disable_actorder: bool = field(default=False)
    hessian_path: Optional[str] = field(default=None)
    inv_hessian_path: Optional[str] = field(default=None)
    num_gpus: int = field(default=1)
    gpu_ids: List[int] = field(default_factory=lambda: [-1,-1])
    eval_nsamples: int = field(default=128)
    save_qlinear: bool = field(default=False)
    vq_type: str = field(default="vptq")
    train_dataset: str = field(default="wikitext2")
    eval_dataset: str = field(default="wikitext2")
    batch_size: int = field(default=16)
    devset_size: int = field(default=384)
    ctx_size: int = field(default=2048)
    ft_valid_size: int = field(default=128)
    ft_bs: int = field(default=4)
    ft_lr: float = field(default=1e-5)
    ft_epochs: int = field(default=5)
    ft_update_freq: int = field(default=2)
    ft_valid_freq: int = field(default=1)
    ft_valid_size: int = field(default=128)
    ft_early_stop: int = field(default=2)
    finetune_type : str = field(default="codebook")
    finetune_blocks: List[int] = field(default_factory=lambda: list(range(32)))


if __name__ == "__main__":
    parser = HfArgumentParser((VPTQArguments, QuantizationArguments))
    args, quant_args = parser.parse_args_into_dataclasses()
    for key, value in vars(args).items():
            print(f"{key}: {value}")
    for key, value in vars(quant_args).items():
            print(f"{key}: {value}")

    if args.gpu_ids is not None:
        args.num_gpus = len(args.gpu_ids)
    
    folder = args.output_dir+'/print_info'
    sys.stdout = Logger(folder=folder)

    # set output folder based on time
    args.output_dir = osp.join(args.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))

    set_start_method("spawn")

    set_seed(args.seed)

    if "llama" in args.model_name.lower():
        model = get_llama(args.model_name)
    elif "qwen" in args.model_name.lower():
        model = get_qwen(args.model_name)
    elif "mistral" in args.model_name.lower():
        model = get_mistral(args.model_name)
    elif "nv-embed" in args.model_name.lower():
        model = get_nvembed(args.model_name)
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # set sequence length
    if args.seq_len or model.seqlen is None:
        model.seqlen = args.seq_len
    print(f"model sequence length: {model.seqlen}")

    model.eval()
    
    # # eval wikitext2 ppl of original model
    # dataloader, testloader = get_data_loader("wikitext2", seed=args.seed, model=args.model_name, seqlen=model.seqlen)
    # ppl = eval_llama(model, testloader, "cuda")
    
    if args.vq_type == "rpvq_v3":
        tick = time.time()
        print(f'exp time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        if "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
            model, quantizers = quant_llama(model, args, quant_args)
        elif "qwen" in args.model_name.lower():
            model, quantizers = quant_qwen(model, args, quant_args)
        elif "nv-embed" in args.model_name.lower():
            model, quantizers = quant_nvembed(model, args, quant_args)
        else:
            raise ValueError(f"Unsupported model: {args.model_name}")
        
        torch.save(quantizers,args.output_dir+"quantizers.pth")
    elif args.vq_type == "finetune":
        quantizers = torch.load('outputs/llama3-8b-hf/quantizers_fn_codebook.pth')
        from rpvq_v3.ops import QuantizedLinear
        for i in range(len(model.model.layers)):
            for linear_id in ['self_attn.v_proj','self_attn.q_proj', 'self_attn.k_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']:
                torch.cuda.empty_cache()
                linear = getattr(getattr(model.model.layers[i],linear_id.split('.')[0]),linear_id.split('.')[1])
                quantizer = quantizers[i][f"{i}.{linear_id}"]
                quantizelinear = QuantizedLinear(linear,quantizer,quant_flag=True,finetune="")
                setattr(getattr(model.model.layers[i],linear_id.split('.')[0]),linear_id.split('.')[1],quantizelinear)
        torch.cuda.empty_cache()
        
        dataloader, testloader = get_data_loader(
            "wikitext2", seed=args.seed, model=args.model_name, seqlen=model.seqlen
        )
        # ppl = eval_llama(model, testloader, "cuda")
            
        # 将quantizers转到cpu上去，减少显存
        for block_id,block_layers in quantizers.items():
            for layer_id,layer in block_layers.items():
                layer.S = layer.S.to('cpu')
                layer.U = layer.U.to('cpu')
                layer.Vt = layer.Vt.to('cpu')
                layer.perm = layer.perm.to('cpu')
                layer.weight_bias = layer.weight_bias.to('cpu')
                layer.weight_scale = layer.weight_scale.to('cpu')
                for key1, codebooks in layer.centroids.items():
                    for key2, val2 in codebooks.items():
                        layer.centroids[key1][key2] = val2.to('cpu')
                for key1, indices in layer.indices.items():
                    for key2, val2 in indices.items():
                        layer.indices[key1][key2] = val2.to('cpu')
        torch.cuda.empty_cache()
        
        model = model.to('cpu')
        
        from rpvq_v3.finetune import layerwise_finetune,e2e_finetune
        if "codebook" in args.finetune_type or "lowrank" in args.finetune_type:
            layerwise_finetune(model, quantizers, dataloader, testloader, args)
        elif args.finetune_type=="e2e":
            e2e_finetune(model, dataloader, testloader, args)
        else:
            raise ValueError(f"Unsupported finetune_type: {args.finetune_type}")
        
        for mm in range(32):
            to_device(model.model.layers[mm],"cuda")
            for linear_id in ['self_attn.v_proj','self_attn.q_proj', 'self_attn.k_proj','self_attn.o_proj','mlp.up_proj','mlp.gate_proj','mlp.down_proj']:
                linear = getattr(getattr(model.model.layers[mm],linear_id.split('.')[0]),linear_id.split('.')[1])
                linear.quant_flag = True
                linear.finetune=""
    else:
        raise ValueError(f"Unsupported vq_type: {quant_args.vq_type}")
    
    model.eval()
    if args.eval:
        datasets = ["wikitext2", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]

    seqlens = [2048, 8192, 4096]

    # store results
    results = {}

    for seqlen in seqlens:
        model.seqlen = seqlen
        for dataset in datasets:
            dataloader, testloader = get_data_loader(
                dataset, seed=args.seed, model=args.model_name, seqlen=model.seqlen)
            print(dataset)
            if "llama" in args.model_name.lower() or "mistral" in args.model_name.lower():
                ppl = eval_llama(model, testloader, "cuda")
            elif "qwen" in args.model_name.lower():
                ppl = eval_qwen(model, testloader, "cuda")
            else:
                raise ValueError(f"Unsupported model: {args.model_name}")
            print("ppl: ",ppl)
            # torch.save(model,args.output_dir+"/"+args.model_name+dataset+f"{ppl}.pt")

            if f"ctx_{seqlen}" not in results:
                results[f"ctx_{seqlen}"] = {}
            results[f"ctx_{seqlen}"][dataset] = ppl

        with open(osp.join(args.output_dir, "ppl_results.json"), "w") as f:
            json.dump(results, f, indent=2)
