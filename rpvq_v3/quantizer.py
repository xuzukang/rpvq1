# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import cuml
import numpy as np
import torch
import torch.nn as nn
from vptq.utils.reshape import reshape

@dataclass
class QuantizationArguments:
    vector_lens: List[int] = field(default_factory=lambda: [-1, 1])
    num_centroids: List[int] = field(default_factory=lambda: [-1, -1])
    num_res_centroids: List[int] = field(default_factory=lambda: [-1, -1])
    num_res_layers: List[int] = field(default_factory=lambda: [1, 1])
    npercent: float = field(default=0)
    group_num: int = field(default=1)
    group_size: int = field(default=-1)
    kiter: int = field(default=100)
    ktol: float = field(default=1e-5)
    kseed: int = field(default=0)
    kmeans_mode: str = field(default=None)
    kmeans_alpha: float = field(default=0)
    enable_norm: bool = field(default=False)
    norm_dim: int = field(default=0)
    enable_perm: bool = field(default=False)
    codebook_bitwidth: int = field(default=-1)

class NPVectorQuantizer_RPVQV3():
    def __init__(
        self,
        layer_name,
        logger,
        # vector quantization parameters
        vector_lens: Tuple[int, int],
        num_centroids: Tuple[int, int],
        num_res_centroids: Tuple[int, int],
        num_res_layers: Tuple[int, int],
        npercent: int,
        group_size: int,
        group_num: int,
        # kmeans parameters
        kmeans_mode: str = '',
        kmeans_seed: int = 0,
        # enable_transpose: bool = False,
        iter: int = 100,
        tol: float = 1e-5,
        # norm
        enable_norm: bool = True,
        norm_dim: int = 0,
        enable_perm: bool = True,
        debug: bool = False,
        # loaded_weights: dict = None,
        codebook_bitwidth = None,
        low_rank = 0,
    ):
        super(NPVectorQuantizer_RPVQV3, self).__init__()
        assert isinstance(num_centroids, (list, tuple))
        self.codebook_bitwidth = codebook_bitwidth
        self.low_rank = low_rank
        self.U = None
        self.S = None
        self.Vt = None
            
        # self.scale = None
        self.enable_perm = enable_perm
        self.enable_transpose = True

        self.vector_lens = vector_lens
        self.num_centroids = num_centroids
        self.num_res_centroids = num_res_centroids
        self.num_res_layers = num_res_layers
        self.npercent = npercent

        self.group_size = group_size
        self.group_num = group_num
        assert not ((self.group_size != -1) and (self.group_num != -1)), 'Can not set both group_size and group_num'
        self.iter = iter
        self.tol = tol
        self.kmeans_seed = kmeans_seed
        self.layer_name = layer_name

        # vector_len
        self.outlier_vector_len = self.vector_lens[0]
        self.vector_len = self.vector_lens[1]

        if self.outlier_vector_len > 0:
            self.enable_outlier = True
        else:
            self.enable_outlier = False

        # check kmeans_mode
        if kmeans_mode not in ['hessian', '']:
            raise ValueError(f'Not supported kmeans mode:{kmeans_mode}')

        self.kmeans_mode = kmeans_mode
        # self.kmeans_alpha = kmeans_alpha

        self.enable_norm = enable_norm
        self.norm_dim = norm_dim

        # centroids and indices
        self.centroids, self.indices = {}, {}
        # residual centroids and indices
        self.res_centroids, self.res_indices = {}, {}
        self.vector_norm = None

        # load checkpoint
        # self.loaded_weights = loaded_weights

        # reshape
        self.reshaper = {}
        self.res_reshaper = {}

        self.perm = None
        self.weight_scale = None
        self.weight_bias = None

        # debug
        self.debug = debug
        self.logger = logger

        # prefix layer name
        self.prefix_layer_name = 'model.layers.'

    def init_norm(self, weight):
        self.weight_scale = torch.std(weight, dim=self.norm_dim)
        self.weight_bias = torch.mean(weight, dim=self.norm_dim)

        if self.debug:
            self.logger.info(
                f'enabling norm dim {self.norm_dim}, '
                f'layer_name:{self.layer_name}, '
                f'scale:{self.weight_scale.shape}, '
                f'bias:{self.weight_bias.shape}'
            )

    # init permutation
    def init_perm(self, hessian, perm=None):
        if perm is not None:
            self.perm = perm
        else:
            self.perm = torch.argsort(torch.diag(hessian), descending=True)

    def get_centroid_cidx(self, index):
        if index < self.outlier_size:
            # if this index belongs to the first N%,
            # it should be quantized with the first codebook
            cidx = 0
        elif self.group_size != -1:
            # if use Product Quantization, find the corresponding codebook
            cidx = (index - self.outlier_size) // self.group_size + 1
        else:
            cidx = 1
        return cidx

    def get_group_setting(self, data):
        if self.enable_transpose:
            initial_outlier_size = int(math.ceil((self.npercent / 100) * data.shape[1]))
            remaining_columns = data.shape[1] - initial_outlier_size
            _pad = remaining_columns % self.group_num
            outlier_size = initial_outlier_size
            self.outlier_size = outlier_size
            if self.group_num != -1:
                if _pad != 0:
                    group_size = (data.shape[1] - self.outlier_size) // self.group_num + _pad
                else:
                    group_size = (data.shape[1] - self.outlier_size) // self.group_num
                self.group_size = group_size
            else:
                assert True, 'only support transpose mode'
        else:
            assert True, 'only support transpose mode'

        return self.outlier_size, self.group_size, self.group_num

    def get_index_list(self, data):
        if self.group_size == -1 and self.group_num == -1:
            index_list = [[0, self.outlier_size], [self.outlier_size, None]]  # N% and (100-N)%
            return index_list

        # if setting group_size or num_goup, update self.group_size and self.group_num
        if self.group_size != -1:
            self.group_num = math.ceil((data.shape[1] - self.outlier_size) / self.group_size)
        elif self.group_num != -1:
            group_size = math.ceil((data.shape[1] - self.outlier_size) / self.group_num)
            if self.enable_transpose:
                self.group_size = group_size
            else:
                # group size should be multiple of vector_len
                self.group_size = math.ceil(group_size / self.vector_len) * self.vector_len
            assert self.group_num == math.ceil((data.shape[1] - self.outlier_size) / group_size)

        index_list = [[0, self.outlier_size]] + \
            [[self.outlier_size + t * self.group_size, self.outlier_size + (t+1) * self.group_size]
             for t in range(self.group_num)]

        if self.debug:
            self.logger.info(f'group_size: {self.group_size} '
                             f'number of groups: {self.group_num}')
        return index_list

    # k-means and quantize
    def init_centroids_indices(self, data, weights=None,codebook_ids=-1):
        self.logger.info(
            f'data shape: {data.shape}, '
            f'weights shape: {weights.shape if weights is not None else None}'
        )
        quantized_data = []
        quantized_data_group = []

        for idx, (i, j) in enumerate(self.get_index_list(data)):
            if idx != codebook_ids+1: #因为有outlier在，所以要+1
                continue
            
            
            self.centroids[idx] = {}
            
            if True:#数据处理
                if len(self.num_centroids)>2:
                    num_centroids = self.num_centroids[idx]
                else:
                    num_centroids = self.num_centroids[0] if idx == 0 else self.num_centroids[1]
                vector_len = self.outlier_vector_len if idx == 0 else self.vector_len
                train_data = data[:, i:j].clone()
                train_weights = weights[:, i:j].clone() if weights is not None else None
                
                self.reshaper[idx] = reshape(vector_len=vector_len, enable_transpose=self.enable_transpose)

                sub_vectors, padded_shape = self.reshaper[idx].matrix2vectors(train_data)
                vector_weights, _ = self.reshaper[idx].matrix2vectors(train_weights)
                
                vector_weights = vector_weights.mean(dim=1) if vector_weights is not None else None
            
            if self.num_res_layers[idx] == 0:
                sub_vectors = torch.zeros_like(sub_vectors)
                pq_lens = vector_len
                quantized_data_group.append(sub_vectors)
            else:
                sub_vectors = sub_vectors.to(torch.float32)
                pq_lens =  torch.ceil(torch.tensor(vector_len / self.num_res_layers[idx])).int().item()
                for num_res_layer in range(self.num_res_layers[idx]):
                    _kmeans = cuml.cluster.KMeans(
                        n_clusters=num_centroids, tol=self.tol, init='random', 
                        max_iter=self.iter, random_state=0, n_init=1)
                    # convert to numpy and float32 to avoid error
                    pq_sub_vectors = sub_vectors.to(torch.float32).cpu().numpy()[:,num_res_layer*pq_lens:(num_res_layer+1)*pq_lens]
                    _kmeans.fit(pq_sub_vectors, sample_weight=vector_weights)
                    self.logger.info(f'cuml kmeans {_kmeans.n_iter_} iterations, error {_kmeans.inertia_}')

                    self.centroids[idx][num_res_layer] = torch.from_numpy(_kmeans.cluster_centers_).to(device=data.device)
                    quant_data = self.centroids[idx][num_res_layer][_kmeans.labels_]
                    quant_data.to(device=data.device)
                    quantized_data_group.append(quant_data)
            
            quantized_data_group=torch.cat(quantized_data_group,dim=-1)         
            quantized_data.append(quantized_data_group)
                
        quantized_data = torch.hstack(quantized_data)
        self.logger.info(f'quantized_data shape: {quantized_data.shape}')

        return quantized_data

    def quantize_vector(self, data, index):
        '''
        input data shape: if not transposed [-1,vector_len] else [nrows,1]
        index: The index of the first column of the input data in the entire weight matrix
        '''
        # Input check
        if self.enable_transpose:
            assert data.shape[1] == 1, 'only support quantize one column each time'
            data = data.T  #[1, 4096]

        cidx = self.get_centroid_cidx(index)
        vector_len = self.outlier_vector_len if cidx == 0 else self.vector_len
        if self.centroids[cidx] is None:
            # keep original data for further quantization
            quantized_data = data
            self.indices[cidx] = None
        else:
            # matrix to vectors
            data, is_padded, pad_cols = self.reshaper[cidx].add_padding(data)

            if is_padded:
                assert is_padded == self.reshaper[cidx].is_padded \
                    and pad_cols == self.reshaper[cidx].pad_cols, \
                    f'cidx {cidx} index {index} pad_cols {pad_cols}' \
                    f'self.pad_cols {self.reshaper[cidx].pad_cols}' \
                    f'Error maybe caused by incorrect block_size settings'

            padded_shape = data.shape
            data = data.reshape(-1, vector_len)
            
            quantized_data = []
            if self.num_res_layers[cidx] == 0:
                quantized_data = torch.zeros_like(data)
            else:
                pq_lens =  torch.ceil(torch.tensor(vector_len / self.num_res_layers[cidx])).int().item()
                for num_res_layer in range(self.num_res_layers[cidx]):
                    dist = torch.cdist(data.float()[:,num_res_layer*pq_lens:(1+num_res_layer)*pq_lens ], 
                                    self.centroids[cidx][num_res_layer].float())
                    indices = dist.argmin(dim=-1)
                    quantized_data.append(self.centroids[cidx][num_res_layer][indices])
                    if cidx not in self.indices or self.indices[cidx] is None:
                        self.indices[cidx] = {}
                        self.indices[cidx][num_res_layer] = indices.unsqueeze(1).to(device=data.device)
                    elif num_res_layer not in self.indices[cidx]:
                        self.indices[cidx][num_res_layer] = indices.unsqueeze(1).to(device=data.device)
                    else:
                        self.indices[cidx][num_res_layer] = \
                            torch.hstack([self.indices[cidx][num_res_layer], indices.unsqueeze(1).to(device=data.device)])
                
                quantized_data = torch.cat(quantized_data,dim=-1)
            quantized_data = quantized_data.reshape(padded_shape)
            quantized_data = self.reshaper[cidx].remove_padding(quantized_data)

        if self.enable_transpose:
            quantized_data = quantized_data.T
        return quantized_data
    
    def vq_dequant(self, data, index):
        if self.enable_transpose:
            assert data.shape[1] == 1, 'only support quantize one column each time'
            data = data.T
            
        cidx = self.get_centroid_cidx(index)
        vector_len = self.outlier_vector_len if cidx == 0 else self.vector_len
        if self.centroids[cidx] is None:
            # keep original data for further quantization
            quantized_data = data
            self.indices[cidx] = None
        else:
            # matrix to vectors
            data, is_padded, pad_cols = self.reshaper[cidx].add_padding(data)

            if is_padded:
                assert is_padded == self.reshaper[cidx].is_padded \
                    and pad_cols == self.reshaper[cidx].pad_cols, \
                    f'cidx {cidx} index {index} pad_cols {pad_cols}' \
                    f'self.pad_cols {self.reshaper[cidx].pad_cols}' \
                    f'Error maybe caused by incorrect block_size settings'

            padded_shape = data.shape
            data = data.reshape(-1, vector_len)
            
            quantized_data = []
            if self.num_res_layers[cidx] == 0:
                quantized_data = torch.zeros_like(data)
            else:
                
                for num_res_layer in range(self.num_res_layers[cidx]):
                    indices = self.indices[cidx][num_res_layer][:,index]
                    quantized_data.append(self.centroids[cidx][num_res_layer][indices])
                quantized_data = torch.cat(quantized_data,dim=-1)
            quantized_data = quantized_data.reshape(padded_shape)
            quantized_data = self.reshaper[cidx].remove_padding(quantized_data)

        if self.enable_transpose:
            quantized_data = quantized_data.T
        return quantized_data
    
    def fakequant(self, weight):
        self.step = 1 if self.enable_transpose else self.step
        qweight = torch.zeros_like(weight).float()
        for i in range(0, self.columns, self.block_size):
            j = min(i + self.block_size, self.columns)
            size = j - i
            block_weight = weight[:, i:j].clone().float()
            block_qweight = torch.zeros_like(block_weight)
            for k in range(0, size, self.step):
                tile_weight = block_weight[:, k:k + self.step]
                tile_qweight = self.vq_dequant(tile_weight, i + k)
                tile_qweight = tile_qweight.reshape(-1, self.step)
                block_qweight[:, k:k + self.step] = tile_qweight
            qweight[:, i:j] = block_qweight.to(weight)
        
        if self.low_rank:
            weight_svd= torch.matmul(self.U, torch.matmul(torch.diag_embed(self.S), self.Vt))
            qweight += weight_svd
        if self.enable_perm:
            inv_perm = torch.argsort(self.perm)
            qweight = qweight[:, inv_perm]
        if self.enable_norm:
            scale = self.weight_scale.unsqueeze(self.norm_dim) if self.norm_dim == 1 else self.weight_scale
            bias = self.weight_bias.unsqueeze(self.norm_dim) if self.norm_dim == 1 else self.weight_bias
            qweight = qweight * scale + bias
        return qweight
    
    def low_rank_approx(self, data, hessian=None):  
        if self.U is not None and self.S is not None and self.Vt is not None:
            return  torch.matmul(self.U, torch.matmul(torch.diag_embed(self.S), self.Vt))
        if self.low_rank>0:
            if hessian is not None:
                damp = 0.01 * torch.mean(torch.diag(hessian))
                diag = torch.arange(data.shape[-1], device=hessian.device)
                hessian[diag, diag] += damp
                sqrt_hessian = torch.linalg.cholesky(hessian) # 使用 Cholesky 分解求逆
                inv_sqrt_hessian = torch.linalg.solve(sqrt_hessian,torch.eye(sqrt_hessian.shape[0], device=sqrt_hessian.device))
                data_h = torch.matmul(data, sqrt_hessian.to(data))  # 因为 sqrt_hessian 是 Cholesky 分解的结果，所以 L*L.T = hessian
                U, S, Vt = torch.linalg.svd(data_h, full_matrices=False)# 使用 SVD 进行低秩近似
                U = U[..., :self.low_rank]
                S = S[..., :self.low_rank]  # 只取前 self.low_rank 个奇异值
                Vt = torch.matmul(Vt[:self.low_rank], inv_sqrt_hessian.to(data))
                data_ = torch.matmul(U, torch.matmul(torch.diag_embed(S), Vt))
            else:
                U, S, Vt = torch.linalg.svd(data, full_matrices=False)
                U = U[..., :self.low_rank]
                S = S[..., :self.low_rank]
                Vt = Vt[:self.low_rank]
                data_ = torch.matmul(U, torch.matmul(torch.diag_embed(S), Vt))
            self.U=U
            self.S=S
            self.Vt=Vt
            return data_

