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
        x = x.permute(0, 2, 1).contiguous() # -> (batch_size, seq_len, embed_dim), ensure contiguous

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
        
        patch_size_config = config['vit']['patch_size']
        if isinstance(patch_size_config, int):
            patch_size = (patch_size_config, patch_size_config)
        else:
            patch_size = tuple(patch_size_config)

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


class DualStreamTransformer(nn.Module):
    def __init__(self, config):
        super(DualStreamTransformer, self).__init__()
        
        self.embed_dim = config['model']['embed_dim']
        self.seq_len = config['max_packets_per_interval']
        
        # --- MODIFICATION: Two separate embedding layers for each stream ---
        self.conv_embed_a = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2, # Halved for each stream
            out_channels=self.embed_dim, 
            kernel_size=(1, config['model']['feature_dim'])
        )
        self.conv_embed_b = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2, # Halved for each stream
            out_channels=self.embed_dim, 
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

    def forward(self, src_a, src_b):
        # src_a and src_b shape: (batch_size, channels/2, seq_len, features)
        
        # Process stream A
        x_a = self.conv_embed_a(src_a) # -> (batch_size, embed_dim, seq_len, 1)
        x_a = x_a.squeeze(3) # -> (batch_size, embed_dim, seq_len)
        x_a = x_a.permute(0, 2, 1).contiguous() # -> (batch_size, seq_len, embed_dim)

        # Process stream B
        x_b = self.conv_embed_b(src_b)
        x_b = x_b.squeeze(3)
        x_b = x_b.permute(0, 2, 1).contiguous()

        # --- MODIFICATION: Fuse features by element-wise addition ---
        x = x_a + x_b
        
        # Shared Transformer
        x = self.positional_encoding(x)
        output = self.transformer_encoder(x)
        
        output = output.mean(dim=1)
        output = self.fc_out(output)
        return output


class MultiTaskTransformer(nn.Module):
    def __init__(self, config, initial_weights=None):
        super(MultiTaskTransformer, self).__init__()
        
        self.embed_dim = config['model']['embed_dim']
        
        # --- 1. Adaptive Weighting Layer (Re-parameterized with Logits) ---
        # We create 3 pairs of logits, one for each task. These are trainable 
        # parameters. The softmax function, applied in the forward pass, will 
        # ensure that the weights generated for each task always sum to 1.
        if initial_weights is None:
            # Default to 0.5/0.5 weights if none are provided.
            # The corresponding logits for [0.5, 0.5] are [0, 0].
            logits1 = torch.zeros(2, dtype=torch.float32)
            logits2 = torch.zeros(2, dtype=torch.float32)
            logits3 = torch.zeros(2, dtype=torch.float32)
        else:
            def probs_to_logits(p_a, p_b):
                """Converts a pair of probabilities into a pair of logits."""
                # Add epsilon to prevent log(0) and division by zero
                p_a = max(p_a, 1e-6)
                p_b = max(p_b, 1e-6)
                # Normalize just in case they don't sum to 1
                total = p_a + p_b
                p_a /= total
                p_b /= total
                # To get logits from probabilities for a 2-element softmax,
                # we can set one logit to 0 for stability and calculate the other.
                # l_a = log(p_a / p_b)
                l_a = torch.log(torch.tensor(p_a / p_b, dtype=torch.float32))
                l_b = torch.tensor(0.0, dtype=torch.float32)
                return torch.stack([l_a, l_b])

            logits1 = probs_to_logits(
                initial_weights['Predict Room A']['weight_a'], 
                initial_weights['Predict Room A']['weight_b']
            )
            logits2 = probs_to_logits(
                initial_weights['Predict Room B']['weight_a'], 
                initial_weights['Predict Room B']['weight_b']
            )
            logits3 = probs_to_logits(
                initial_weights['Predict Living Room']['weight_a'], 
                initial_weights['Predict Living Room']['weight_b']
            )

        self.task1_logits = nn.Parameter(logits1)
        self.task2_logits = nn.Parameter(logits2)
        self.task3_logits = nn.Parameter(logits3)
        
        # --- 2. Dual-Stream Embedding Layers (Shared across tasks) ---
        self.conv_embed_a = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2,
            out_channels=self.embed_dim, 
            kernel_size=(1, config['model']['feature_dim'])
        )
        self.conv_embed_b = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2,
            out_channels=self.embed_dim, 
            kernel_size=(1, config['model']['feature_dim'])
        )
        
        # --- 3. Shared Body ---
        self.positional_encoding = PositionalEncoding(self.embed_dim, config['model']['dropout'])
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=config['model']['num_heads'], 
            dim_feedforward=config['model']['hidden_dim'], 
            dropout=config['model']['dropout'],
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config['model']['num_layers'])
        
        # --- 4. Multi-Head Classifier ---
        self.head_a = nn.Linear(self.embed_dim, 1) # Binary output
        self.head_b = nn.Linear(self.embed_dim, 1) # Binary output
        self.head_lr = nn.Linear(self.embed_dim, 1) # Binary output

    @property
    def weights_task1(self):
        """Computes softmax on logits to get weights for task 1."""
        return torch.softmax(self.task1_logits, dim=0)
    
    @property
    def weights_task2(self):
        """Computes softmax on logits to get weights for task 2."""
        return torch.softmax(self.task2_logits, dim=0)

    @property
    def weights_task3(self):
        """Computes softmax on logits to get weights for task 3."""
        return torch.softmax(self.task3_logits, dim=0)

    def _get_shared_features(self, src_a, src_b, w_a, w_b):
        """ Helper to compute features for one task. """
        # Apply weights and process through embedding layers
        x_a = self.conv_embed_a(src_a * w_a).squeeze(3).permute(0, 2, 1).contiguous()
        x_b = self.conv_embed_b(src_b * w_b).squeeze(3).permute(0, 2, 1).contiguous()
        
        # Fuse and pass through shared body
        x = self.positional_encoding(x_a + x_b)
        shared_features = self.transformer_encoder(x)
        
        # Global Average Pooling
        return shared_features.mean(dim=1)

    def forward(self, src_a, src_b):
        # --- Generate weights from logits using softmax ---
        w1 = self.weights_task1
        w2 = self.weights_task2
        w3 = self.weights_task3
        
        # Task 1: Predict Room A
        features_task1 = self._get_shared_features(src_a, src_b, w1[0], w1[1])
        output_a = self.head_a(features_task1)
        
        # Task 2: Predict Room B
        features_task2 = self._get_shared_features(src_a, src_b, w2[0], w2[1])
        output_b = self.head_b(features_task2)
        
        # Task 3: Predict Living Room
        features_task3 = self._get_shared_features(src_a, src_b, w3[0], w3[1])
        output_lr = self.head_lr(features_task3)
        
        return output_a, output_b, output_lr


def get_model(config):
    """
    Model factory.
    """
    model_name = config['model']['name']
    if model_name == 'simple_transformer':
        return SimpleTransformer(config)
    elif model_name == 'vit':
        return ViT(config)
    elif model_name == 'dual_stream_transformer':
        return DualStreamTransformer(config)
    elif model_name == 'multi_task_transformer':
        # --- MODIFICATION: Read initial weights from the config file ---
        initial_weights = config.get('training', {}).get('initial_weights')
        return MultiTaskTransformer(config, initial_weights=initial_weights)
    else:
        raise ValueError(f"Unknown model name: {model_name}") 