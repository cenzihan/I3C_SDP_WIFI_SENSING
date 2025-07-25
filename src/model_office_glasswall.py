import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN designed for the Office Scenario data.
    Input shape: (Batch, 40, 100, 64) -> (B, C, H, W)
    Output shape: (Batch, 5)
    """
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        
        input_channels = config['model']['input_channels']
        num_classes = config['model']['num_classes']

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 32, 50, 32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 64, 25, 16)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> (B, 128, 12, 8)

        # Flatten the output for the fully connected layer
        # The size is 128 * 12 * 8
        self.flattened_size = 128 * 12 * 8
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and fully connected layers
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # Output logits
        
        return x

class ViTOffice(nn.Module):
    """
    A Vision Transformer adapted for the Office Scenario data.
    Input shape: (Batch, 40, 100, 64) -> (B, C, H, W)
    Output shape: (Batch, 5)
    """
    def __init__(self, config):
        super(ViTOffice, self).__init__()
        
        image_size = (100, 64) # H, W for the office data
        patch_size_config = config['vit']['patch_size']
        patch_size = tuple(patch_size_config)

        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = config['model']['input_channels'] * patch_size[0] * patch_size[1]
        
        embed_dim = config['model']['embed_dim']
        num_classes = config['model']['num_classes']
        
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=config['model']['num_heads'],
            dim_feedforward=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['model']['num_layers'])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        self.patch_size = patch_size

    def forward(self, img):
        b, c, h, w = img.shape
        p_h, p_w = self.patch_size

        patches = img.unfold(2, p_h, p_h).unfold(3, p_w, p_w)
        patches = patches.contiguous().view(b, c, -1, p_h, p_w)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(b, patches.size(1), -1)

        x = self.patch_embedding(patches)
        
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.position_embedding
        
        x = self.transformer_encoder(x)
        
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)


def get_office_model(config):
    """
    Model factory for the Office Scenario.
    """
    model_name = config['model']['name']
    if model_name == 'simple_cnn':
        return SimpleCNN(config)
    elif model_name == 'vit':
        return ViTOffice(config)
    else:
        raise ValueError(f"Unknown office model name: {model_name}") 