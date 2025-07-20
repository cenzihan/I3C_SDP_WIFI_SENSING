import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    It's designed to address class imbalance by down-weighting easy examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class CombinedLoss(nn.Module):
    """
    A loss function that combines multiple weighted losses.
    """
    def __init__(self, losses, weights):
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, inputs, targets):
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss

def get_loss_function(config):
    """
    Builds the loss function based on the configuration.
    It can be a single loss or a combination of multiple losses.
    """
    loss_configs = config['training']['loss_components']
    
    if len(loss_configs) == 1:
        # If only one loss is specified, return it directly
        return _create_loss_from_config(loss_configs[0], config)

    losses = []
    weights = []
    for loss_config in loss_configs:
        losses.append(_create_loss_from_config(loss_config, config))
        weights.append(loss_config['weight'])
        
    return CombinedLoss(losses, weights)

def _create_loss_from_config(loss_config, full_config):
    """
    Helper to instantiate a single loss function from its config.
    """
    name = loss_config['name']
    params = loss_config.get('params', {})
    
    if name == 'bce':
        # BCEWithLogitsLoss uses the global pos_weight
        pos_weight = torch.tensor(full_config['training']['pos_weight'], dtype=torch.float)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight, **params)
    elif name == 'focal':
        return FocalLoss(**params)
    else:
        raise ValueError(f"Unknown loss function name: {name}") 