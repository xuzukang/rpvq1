# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import time
import math
import torch
import torch.nn as nn
import transformers

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from rpvq_v3.quantizer import NPVectorQuantizer_RPVQV3

class RPVQ_V3:
    """
    没有residual quantization, 权重分组，前一组量化好后，才生成下一组的中心点
    """

    def __init__(
        self,
        layer,
        quantizer: NPVectorQuantizer_RPVQV3,
        hessian,
        inv_hessian,
        perm,
        zero_idx,
        logger,
        collect_act=False,
        layer_name='',
        block_size=128,
        step=1,
        percdamp=.01,
        group_size=-1,
        group_num=-1,
        enable_perm=None,
        enable_norm=False,
        norm_dim=0,
        debug=False,
        vq_type="RPVQ_V3",
        args=None,
    ):
        self.args = args
        self.vq_type = vq_type

        # set quantizer
        self.quantizer = quantizer

        # vptq parameter
        self.block_size = block_size
        self.step = step
        self.percdamp = percdamp

        self.group_size = group_size
        self.group_num = group_num

        # set device
        self.dev = layer.weight.device
        # layer name
        self.layer_name = layer_name

        # save weight
        # self.weight = self.layer.weight.data.to(self.dev)
        # self.qweight = torch.zeros_like(self.weight)
        # self.hessian = hessian.to(self.dev)

        # preprocess
        self.layer = layer.to('cpu')
        self.weight = self.layer.weight.data.to('cpu')
        self.qweight = torch.zeros_like(self.weight).to('cpu')
        self.hessian = hessian.to('cpu')

        if inv_hessian is not None:
            self.inv_hessian = inv_hessian.to('cpu')
        else:
            self.inv_hessian = None
        if perm is not None:
            self.perm = perm.to('cpu')
        else:
            self.perm = None
        if zero_idx is not None:
            self.zero_idx = zero_idx.to('cpu')
        else:
            self.zero_idx = None

        if isinstance(self.layer, nn.Conv2d):
            self.weight = self.weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            self.weight = self.weight.t()

        # out_features
        self.rows = self.weight.shape[0]
        # in_features
        self.columns = self.weight.shape[1]

        # nsamples
        self.nsamples = 0

        # hessian matrix
        # self.hessian = torch.zeros(
        #     (self.columns, self.columns), device=self.dev)

        # collect activation
        self.collect_act = collect_act
        self.act = None

        # permute
        self.quantizer.enable_perm = enable_perm
        # self.quantizer.perm = None

        # weight norm
        self.enable_norm = enable_norm
        self.norm_dim = norm_dim
        # self.quantizer.weight_scale = None
        # self.quantizer.weight_bias = None

        # debug flag
        self.debug = debug
        self.logger = logger

    # vptq algorithm, we do not permute weight and hessian here
    def rpvq(self, weight, hessian, enable_residual=False, inv_hessian=None, kmeans_weight=None):
        self.quantizer.columns = self.columns
        self.quantizer.block_size = self.block_size
        if not self.args.gptq:
            self.step = 1 if self.quantizer.enable_transpose else self.step
            qweight = torch.zeros_like(weight)
            qerror = torch.zeros_like(weight)
            codebook_ids = -1
            for i in range(0, self.columns, self.block_size):
                j = min(i + self.block_size, self.columns)
                size = j - i
                block_weight = weight[:, i:j].clone()
                block_qweight = torch.zeros_like(block_weight)
                block_error = torch.zeros_like(block_qweight)
                for k in range(0, size, self.step):
                    tile_weight = block_weight[:, k:k + self.step]
                    if (i + k)//self.group_size > codebook_ids:
                        codebook_ids = (i + k)//self.group_size
                        self.quantizer.init_centroids_indices(data=weight.clone(),
                                                            weights=kmeans_weight,
                                                            codebook_ids=codebook_ids)
                    tile_qweight = self.quantizer.quantize_vector(tile_weight, i + k)
                    tile_qweight = tile_qweight.reshape(-1, self.step)
                    block_qweight[:, k:k + self.step] = tile_qweight
                qweight[:, i:j] = block_qweight
                qerror[:, i:j] = (block_weight - block_qweight).clone()
            return qweight, qerror
        else:
            # force set step=1 if transposed
            # weight_ori=weight.clone()
            self.step = 1 if self.quantizer.enable_transpose else self.step

            # error = torch.zeros_like(weight)
            qweight = torch.zeros_like(weight)

            # gptq error
            qerror = torch.zeros_like(weight)

            if inv_hessian is None:
                damp = self.percdamp * torch.mean(torch.diag(hessian))
                diag = torch.arange(self.columns, device=self.dev)
                hessian[diag, diag] += damp

                # inverse Hessian
                hessian = torch.linalg.cholesky(hessian)
                hessian = torch.cholesky_inverse(hessian)

                # follow gptq methods
                # upper=True: whether to return an upper triangular matrix
                # compute all information needed from H-1 upfront

                hessian = torch.linalg.cholesky(hessian, upper=True)
                inv_hessian = hessian
            else:
                inv_hessian = inv_hessian
                assert inv_hessian is not None

            codebook_ids = -1

            # select weight[i, j] to quantize
            for i in range(0, self.columns, self.block_size):
                j = min(i + self.block_size, self.columns)
                size = j - i

                block_weight = weight[:, i:j].clone()
                block_qweight = torch.zeros_like(block_weight)
                block_error = torch.zeros_like(block_qweight)

                # block of Hessian
                block_inv_hessian = inv_hessian[i:j, i:j]
                
                for k in range(0, size, self.step):
                    tile_weight = block_weight[:, k:k + self.step]
                    tile_inv_hessian = block_inv_hessian[k:k + self.step, k:k + self.step]
                    
                    if (i + k)//self.group_size > codebook_ids:
                        codebook_ids = (i + k)//self.group_size
                        self.quantizer.init_centroids_indices(data=weight.clone(),
                                                            weights=kmeans_weight,
                                                            codebook_ids=codebook_ids)

                    tile_qweight = self.quantizer.quantize_vector(tile_weight, i + k)
                    tile_qweight = tile_qweight.reshape(-1, self.step)

                    tile_inv_hessian = torch.cholesky_inverse(torch.linalg.cholesky(tile_inv_hessian))

                    # update quantized block qweight
                    block_qweight[:, k:k + self.step] = tile_qweight

                    # (drow,step)*(step,step)=(drow,step)
                    tile_error = (tile_weight - tile_qweight).matmul(tile_inv_hessian)


                    block_weight[:, k + self.step:] -= tile_error.matmul(block_inv_hessian[k:k + self.step, k + self.step:])
                    block_error[:, k:k + self.step] = tile_error

                qweight[:, i:j] = block_qweight
                # Losses[:, i1:i2] = Losses1 / 2

                # copy gptq error from Err1
                qerror[:, i:j] = (block_weight - block_qweight).clone()

                # update remaining full-preicision weight
                weight[:, j:] -= block_error.matmul(inv_hessian[i:j, j:])

            # qweight = self.quantizer.low_rank_approx(weight_ori, qweight)
            return qweight, qerror

    def get_error(self, weight, qweight, hessian):

        def _matrix_multiply_with_blocks(A, B, hessian, block_size=64, dev='cuda'):
            m_dim = A.shape[0]
            k_dim = A.shape[1]
            n_dim = B.shape[1]
            if m_dim >= 16384 and k_dim >= 16384:
                # if m_dim >= 16 and k_dim >= 16:
                result = torch.zeros((m_dim, n_dim), device=dev, dtype=A.dtype)
                for i in range(0, m_dim, block_size):
                    i_end = min(i + block_size, m_dim)
                    for j in range(0, n_dim, block_size):
                        j_end = min(j + block_size, n_dim)
                        result[i:i_end, j:j_end] += A[i:i_end, :].to(dev) @ B[:, j:j_end].to(dev)
                        result[i:i_end, j:j_end] = result[i:i_end, j:j_end] * hessian[i:i_end, j:j_end]
            else:
                result = A.to(dev) @ B.to(dev) * hessian
            result = result.to(dev)
            return result

        # weight_mean = torch.mean(weight.T @ weight * hessian)
        # error_mean = torch.mean(error.T @ error * hessian)
        weight = weight.to(qweight.device)
        hessian = hessian.to(qweight.device)
        wTw_hessian = _matrix_multiply_with_blocks(weight.T, weight, hessian, block_size=512, dev=qweight.device)
        weight_mean = torch.mean(wTw_hessian.to(qweight.device))
        # weight_mean = torch.mean(wTw * hessian)
        del wTw_hessian
        torch.cuda.empty_cache()
        error = qweight - weight
        eTe_hessian = _matrix_multiply_with_blocks(error.T, error, hessian, block_size=512, dev=qweight.device)
        error_mean = torch.mean(eTe_hessian.to(qweight.device))
        del eTe_hessian
        torch.cuda.empty_cache()
        error_norm = error_mean / weight_mean

        return error_mean, weight_mean, error_norm

    def fast_vector_quant_rpvq(self, init_centroids=True):
        self.init_centroids = init_centroids

        # step 0: preprocess weight and hessian
        weight = self.weight.clone().float().to(self.dev)
        hessian = self.hessian.clone().to(self.dev)
        inv_hessian = self.inv_hessian.clone().to('cpu')

        if self.enable_norm:
            self.quantizer.init_norm(weight)

            if self.norm_dim == 0:
                weight = (weight - self.quantizer.weight_bias) / \
                    self.quantizer.weight_scale
            else:
                weight = (weight - self.quantizer.weight_bias.unsqueeze(self.norm_dim)) / \
                    self.quantizer.weight_scale.unsqueeze(self.norm_dim)

        if isinstance(self.layer, nn.Conv2d):
            weight = weight.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            weight = weight.t()

        # set group num and size
        self.outlier_size, self.group_size, self.group_num = \
            self.quantizer.get_group_setting(weight)

        # print(f'group_size: {self.group_size}, group_num: {self.group_num}')

        # if not self.quantizer.ready():
        #     self.quantizer.find_params(W, weight=True)
        # del self.H

        if self.quantizer.kmeans_mode == 'hessian':
            kmeans_weight = torch.diag(hessian).clone().unsqueeze(0).repeat(weight.shape[0], 1)
        else:
            kmeans_weight = None

        # set dead diagonal after kmeans_weight
        if self.zero_idx is None:
            self.zero_idx = torch.diag(hessian) == 0
        hessian[self.zero_idx, self.zero_idx] = 1
        weight[:, self.zero_idx] = 0

        if self.debug:
            self.logger.info(
                f'kmeans_mode: {self.quantizer.kmeans_mode}, '
                f'enable_perm: {self.quantizer.enable_perm}'
            )

        # permute weight and hessian
        if self.quantizer.enable_perm is not None:
            # if self.quantizer.enable_perm == 'hessian':
            #     self.quantizer.perm = torch.argsort(torch.diag(hessian), descending=True)
            # init perm in quantizer
            self.quantizer.init_perm(hessian, self.perm)
            # reorder weight and H
            weight = weight[:, self.quantizer.perm]
            hessian = hessian[self.quantizer.perm][:, self.quantizer.perm]

            # reorder kmeans_weight
            if self.quantizer.kmeans_mode in ['hessian'] \
                    and kmeans_weight is not None:
                kmeans_weight = kmeans_weight[:, self.quantizer.perm]
            else:
                kmeans_weight = None
        else:
            self.quantizer.perm = torch.arange(weight.shape[1])

        # save gpu memory
        weight = weight.to('cpu')
        hessian = hessian.to('cpu')
        inv_hessian = inv_hessian.to('cpu')


        _weight = weight.clone().to(self.dev)
        _hessian = hessian.clone().to(self.dev)
        _inv_hessian = inv_hessian.clone().to(self.dev)
        tick = time.time() if self.debug else None

        # first round vptq
        if self.quantizer.low_rank > 0:
            self.quantizer.low_rank = math.ceil(self.quantizer.low_rank*_weight.shape[-1]/4096)
            if self.args.low_rank_hessian:
                weight_svd = self.quantizer.low_rank_approx(_weight, hessian)
            else:
                weight_svd = self.quantizer.low_rank_approx(_weight)
            residua, qerror = self.rpvq(_weight-weight_svd, _hessian, 
                                    inv_hessian=_inv_hessian, 
                                    kmeans_weight=kmeans_weight)
            qweight = weight_svd + residua
        else:
            qweight, qerror = self.rpvq(_weight, _hessian, 
                                    inv_hessian=_inv_hessian, 
                                    kmeans_weight=kmeans_weight)

        torch.cuda.synchronize()
        del _weight
        del _hessian
        del _inv_hessian
        torch.cuda.empty_cache()

        if self.debug:
            error_sum, sum, norm_error = self.get_error(weight, qweight, hessian)
            self.logger.info(f'{self.layer_name} 1st error time: {time.time() - tick}')
            self.logger.info(
                f'{self.layer_name} proxy error after VPTQ: {error_sum.item()}, '
                f'{sum.item()}, {norm_error.item()}'
            )
            # self.logger.info(f'qerror^2: {torch.mean(qerror ** 2).item()}')

        # self.quantizer.save(qweight)

        if self.quantizer.enable_perm:
            inv_perm = torch.argsort(self.quantizer.perm)
            qweight = qweight[:, inv_perm]
            # self.quantizer.perm = self.quantizer.perm.cpu().numpy()

        if isinstance(self.layer, transformers.Conv1D):
            qweight = qweight.t()

        # reshape back to original shape
        qweight = qweight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if self.enable_norm:
            if self.norm_dim == 0:
                qweight = qweight * self.quantizer.weight_scale + self.quantizer.weight_bias
            elif self.norm_dim == 1:
                qweight = qweight * \
                    self.quantizer.weight_scale.unsqueeze(self.norm_dim) \
                    + self.quantizer.weight_bias.unsqueeze(self.norm_dim)

        self.qweight = qweight

        # post process
        self.layer = self.layer.to(self.dev)
        self.weight = self.weight.to(self.dev)
        self.qweight = self.qweight.to(self.dev)
        torch.cuda.empty_cache()


