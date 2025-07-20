import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super(SimpleTransformer, self).__init__()
        
        self.embed_dim = config['model']['embed_dim']
        self.seq_len = config['max_packets_per_interval']
        
        # Convolutional embedding layer
        self.conv_embed = nn.Conv2d(
            in_channels=config['model']['input_channels'], 
            out_channels=config['model']['embed_dim'], 
            kernel_size=(1, config['model']['feature_dim'])
        )
        
        self.positional_encoding = PositionalEncoding(self.embed_dim, config['model']['dropout'])
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=config['model']['num_heads'], 
            dim_feedforward=config['model']['hidden_dim'], 
            dropout=config['model']['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['model']['num_layers'])
        
        self.dropout = nn.Dropout(config['model']['dropout'])

        self.fc_out = nn.Linear(self.embed_dim, config['model']['num_classes'])

    def forward(self, src):
        # src shape: (batch_size, channels, seq_len, features)
        # e.g., (32, 16, 70, 250)
        
        x = self.conv_embed(src) # -> (batch_size, embed_dim, seq_len, 1)
        x = x.squeeze(3) # -> (batch_size, embed_dim, seq_len)
        x = x.permute(0, 2, 1) # -> (batch_size, seq_len, embed_dim)

        # Transformer
        x = self.positional_encoding(x)
        output = self.transformer_encoder(x)
        
        # Global Average Pooling
        output = output.mean(dim=1)
        
        # Final classification
        output = self.fc_out(output)
        return output

class ViT(nn.Module):
    def __init__(self, config):
        super(ViT, self).__init__()
        # Image size is (seq_len, feature_dim) -> (70, 250)
        # Channels is 16
        image_size = (config['max_packets_per_interval'], config['model']['feature_dim']) 
        patch_size = (config['vit']['patch_size'], config['vit']['patch_size'])
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = config['model']['input_channels'] * patch_size[0] * patch_size[1]

        self.patch_embedding = nn.Linear(patch_dim, config['model']['embed_dim'])
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + 1, config['model']['embed_dim']))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['model']['embed_dim']))
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config['model']['embed_dim'],
            nhead=config['model']['num_heads'],
            dim_feedforward=config['model']['hidden_dim'],
            dropout=config['model']['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['model']['num_layers'])
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(config['model']['embed_dim']),
            nn.Linear(config['model']['embed_dim'], config['model']['num_classes'])
        )

        self.patch_size = patch_size
        self.embed_dim = config['model']['embed_dim']

    def forward(self, img):
        # img shape: (batch_size, channels, height, width) -> (b, 16, 70, 250)
        b, c, h, w = img.shape
        p_h, p_w = self.patch_size

        # Create patches
        patches = img.unfold(2, p_h, p_h).unfold(3, p_w, p_w) # -> (b, c, num_patches_h, num_patches_w, p_h, p_w)
        patches = patches.contiguous().view(b, c, -1, p_h, p_w) # -> (b, c, num_patches, p_h, p_w)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous() # -> (b, num_patches, c, p_h, p_w)
        patches = patches.view(b, patches.size(1), -1) # -> (b, num_patches, c*p_h*p_w)

        # Patch embedding
        x = self.patch_embedding(patches) # -> (b, num_patches, embed_dim)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # -> (b, num_patches+1, embed_dim)
        
        # Add positional embedding
        x += self.position_embedding # -> (b, num_patches+1, embed_dim)
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Classification head
        cls_token_final = x[:, 0]
        return self.mlp_head(cls_token_final)


def get_model(config):
    """
    Model factory.
    """
    model_name = config['model']['name']
    if model_name == 'simple_transformer':
        return SimpleTransformer(config)
    elif model_name == 'vit':
        return ViT(config)
    else:
        raise ValueError(f"Unknown model name: {model_name}") 