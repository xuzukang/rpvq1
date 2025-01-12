import torch
import torch.nn as nn
import torch.optim as optim

class QuantizedLinear(nn.Module):
    def __init__(self, linear,quantizer,quant_flag=False,finetune=None):
        super(QuantizedLinear, self).__init__()
        self.weight = nn.Parameter(linear.weight.data, requires_grad=False)
        if linear.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(linear.bias.data, requires_grad=False)
        
        self.quantizer = quantizer
        self.quant_flag = quant_flag
        
        self.finetune = finetune
        if self.quant_flag:
            self._initialize_quantizer(finetune)
            
    def _initialize_quantizer(self, finetune):
        if finetune == "block_codebook":
            self._set_grad_lowrank(False)
            self._set_grad_codebook(requires_grad=True)
        elif finetune == "block_lowrank":
            self._set_grad_lowrank(True)
            self._set_grad_codebook(requires_grad=False)
        elif finetune == "block_codebook_lowrank":
            self._set_grad_lowrank(True)
            self._set_grad_codebook(requires_grad=True)
    
    def _set_grad_lowrank(self, requires_grad):
        for attr in ["U", "S", "Vt"]:
            setattr(self.quantizer, attr, nn.Parameter(getattr(self.quantizer, attr).data, requires_grad=requires_grad))
    
    def _set_grad_codebook(self, requires_grad):
        for key1, val1 in self.quantizer.centroids.items():
            if isinstance(val1, dict):
                for key2, val2 in val1.items():
                    self.quantizer.centroids[key1][key2] = torch.nn.Parameter(val2.data, requires_grad=requires_grad)
            elif isinstance(val1, torch.Tensor):
                self.quantizer.centroids[key1] = torch.nn.Parameter(val1.data, requires_grad=requires_grad)
            else:
                raise ValueError("Unrecognized type in centroids")

    def _smooth_lowrank(self):
        self.quantizer.U.data = torch.matmul(self.quantizer.U.data, torch.diag_embed(torch.sqrt(self.quantizer.S.data)))
        self.quantizer.S.data = torch.ones_like(self.quantizer.S.data,requires_grad=False)
        self.quantizer.Vt.data = torch.matmul(torch.diag_embed(torch.sqrt(self.quantizer.S.data)), self.quantizer.Vt.data)

    def forward(self, x):
        dtype = x.dtype
        if self.quant_flag:
            weight = self.quantizer.fakequant(self.weight) 
            if not self.finetune:
                self.quant_flag = False
                self.weight = nn.Parameter(weight, requires_grad=self.weight.requires_grad)
        else:
            weight = self.weight
        result = x @ weight.T.to(dtype) + (self.bias.to(dtype).reshape(1, -1) if self.bias is not None else 0)
        return result.to(dtype)
    
    def _move_to_device(self, device):
        self.weight = nn.Parameter(self.weight.to(device), requires_grad=self.weight.requires_grad)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(device), requires_grad=self.bias.requires_grad)
        for attr in ["U", "S", "Vt", "perm", "weight_bias", "weight_scale"]:
            setattr(self.quantizer, attr, getattr(self.quantizer, attr).to(device))
        for key1, codebooks in self.quantizer.centroids.items():
            for key2, val2 in codebooks.items():
                self.quantizer.centroids[key1][key2] = val2.to(device)
        for key1, indices in self.quantizer.indices.items():
            for key2, val2 in indices.items():
                self.quantizer.indices[key1][key2] = val2.to(device)
        torch.cuda.empty_cache()

    def to(self, device):
        self._move_to_device(device)
        return self

    def cpu(self):
        return self.to('cpu')

    def cuda(self, idx=0):
        return self.to(f'cuda:{idx}')
        