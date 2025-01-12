# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import time

import torch
from rpvq_v2.rpvq import RPVQ_V2
from rpvq_v2.quantizer import NPVectorQuantizer_RPVQV2
from vptq.utils.hessian import load_hessian, load_inv_hessian
from vptq.utils.layer_utils import find_layers, replace_layer
from copy import copy,deepcopy

def layer_quantizer(args, quant_args, layer, layer_idx, logger, dev, dtype):

    qlinear_args = {}
    operators = find_layers(layer)
    opeartor_names = [list(operators.keys())]
    # with torch.no_grad():
    for names in opeartor_names:
        # subset: (op name, op) pairs
        subset = {n: operators[n] for n in names}
        # 'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        # 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj'
        logger.info(subset.keys())

        for name in subset:
            # load Hessian
            name2hessian = {
                'self_attn.v_proj': 'qkv',
                'self_attn.q_proj': 'qkv',
                'self_attn.k_proj': 'qkv',
                'self_attn.o_proj': 'o',
                'mlp.up_proj': 'up',
                'mlp.gate_proj': 'up',
                'mlp.down_proj': 'down'
            }

            layer_name = f'{layer_idx}_{name2hessian[name]}.pt'
            hessian_path = f'{args.hessian_path}/{layer_name}'
            hessian, mu = load_hessian(hessian_path, logger)

            # init data
            linear = subset[name].to(dev)
            hessian.to('cpu')

            # load inv_hessian from files to reduce memory usage
            if args.inv_hessian_path is not None:
                inv_hessian_path = f'{args.inv_hessian_path}/{layer_name}'
                inv_hessian, perm, zero_idx = load_inv_hessian(inv_hessian_path, logger)
                inv_hessian.to('cpu')
                perm.to('cpu')
                zero_idx.to('cpu')
            else:
                inv_hessian = None
                perm = None
                zero_idx = None

            layer_name = f'{layer_idx}.{name}'

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            logger.info(f'----Quantizing llama ...---- {current_time} {layer_name}')

            # init quantizer
            quantizer = NPVectorQuantizer_RPVQV2(
                layer_name=layer_name,
                logger=logger,
                vector_lens=deepcopy(quant_args.vector_lens),
                num_centroids=deepcopy(quant_args.num_centroids),
                num_res_centroids=deepcopy(quant_args.num_res_centroids),
                num_res_layers=deepcopy(quant_args.num_res_layers),
                npercent=quant_args.npercent,
                group_size=quant_args.group_size,
                group_num=quant_args.group_num,
                # enable_transpose=True,
                kmeans_mode='hessian',
                iter=quant_args.kiter,
                tol=quant_args.ktol,
                debug=True,
                codebook_bitwidth=quant_args.codebook_bitwidth,
                # enable_load_checkpoint=args.enable_load_checkpoint,
                # enable_load_checkpoint=args.enable_load_checkpoint,
                # load_checkpoint_path=args.load_checkpoint_path,
            )

            # init vptq algo
            _vptq = RPVQ_V2(
                linear,
                hessian=hessian,
                inv_hessian=inv_hessian,
                perm=perm,
                quantizer=quantizer,
                zero_idx=zero_idx,
                logger=logger,
                collect_act=False,
                layer_name=layer_name,
                enable_perm='hessian',
                enable_norm=quant_args.enable_norm,
                norm_dim=0,
                debug=True,
                vq_type=args.vq_type,
            )

            _vptq.fast_vector_quant_rpvq()
            linear.weight.data = _vptq.qweight
            torch.cuda.empty_cache()
            continue

    return layer, qlinear_args
