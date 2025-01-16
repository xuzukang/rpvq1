import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

@contextmanager
def use_tf32():
    fp32_matmul_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('high')
    yield
    torch.set_float32_matmul_precision(fp32_matmul_precision)

def calculate_linear_loss(layer, dataloader, device, loss_type="mse"):
    layer.eval()
    total_loss = 0
    ct = 0
    with torch.no_grad():
        for source, target in dataloader:
            target = target.to(device, non_blocking=True)
            if loss_type == "mse":
                total_loss += nn.MSELoss()(layer(source.to(device)),target)
            elif loss_type == "kl":
                total_loss += nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(layer(source.to(device)), dim=1),
                    F.softmax(target, dim=1))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

def calculate_block_loss(layer, dataloader, device, loss_type="mse"):
    layer.eval()
    total_loss = 0
    ct = 0
    position_ids = None
    with torch.no_grad():
        for source, target in dataloader:
            if position_ids is None:
                position_ids = torch.arange(source.shape[1],device=device).unsqueeze(0)
            target = target.to(device, non_blocking=True)
            if loss_type == "mse":
                total_loss += nn.MSELoss()(layer(source.to(device),position_ids=position_ids)[0],target)
            elif loss_type == "kl":
                # 计算 KL 散度
                total_loss += nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(layer(source.to(device), position_ids=position_ids)[0], dim=-1),
                    F.softmax(target, dim=-1))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            ct += 1
    layer.train()
    return (total_loss / ct).cpu().item()

def calculate_ce_loss_model_ori(model, dataloader, start_dev, in_q, out_q):
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

def calculate_ce_loss_model(model, dataloader, device, in_q=None, out_q=None):
    model.eval()
    total_loss = 0.0
    count = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for source, target in dataloader:
            source = source.to(device)
            target = target.to(device)
            if in_q and out_q:
                in_q.put(source)
                logits = out_q.get().to(device)
            else:
                logits = model(source)['logits']
            logits = logits.view(-1, logits.size(-1))
            loss = loss_fn(logits, target.view(-1,).long())
            total_loss += loss.item()
            count += 1
    average_loss = total_loss / count if count > 0 else float('inf')
    model.train()
    return average_loss
