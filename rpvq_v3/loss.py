import torch
import torch.nn as nn
from contextlib import contextmanager

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)

def calculate_linear_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            target = target.to(device, non_blocking=True)
            total_loss += nn.MSELoss()(layer(source.to(device)),target)
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

def calculate_block_mse_loss(layer, dataloader, device):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],
                                            device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            total_loss += nn.MSELoss()(layer(source.to(device),
                                             position_ids=position_ids)[0],
                                       target)
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

def calculate_ce_loss_model(model, dataloader, start_dev, in_q, out_q):
    model.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            in_q.put(source)
            output = model(source.to(start_dev))['logits'][:, :-1].contiguous()
            output = output.view(-1, output.shape[-1])
            target = out_q.get().to(output.device)
            target = target.view(-1, target.shape[-1])
            total_loss += nn.CrossEntropyLoss()(output, target)
            ct += 1
    model.train()
    return (total_loss / ct).cpu().item()


