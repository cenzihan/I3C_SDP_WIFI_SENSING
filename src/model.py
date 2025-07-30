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
        # e.g., (32, 16, 70, 254) or (32, 16, 70, 250)
        
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

# 删除ViT类定义和相关内容


class DualStreamTransformer(nn.Module):
    def __init__(self, config):
        super(DualStreamTransformer, self).__init__()
        
        self.embed_dim = config['model']['embed_dim']
        self.seq_len = config['max_packets_per_interval']
        
        # --- MODIFICATION: Two separate embedding layers for each stream ---
        # Each stream has 8 channels with 254 or 250 features
        self.conv_embed_a = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2, # 8 channels per stream
            out_channels=self.embed_dim, 
            kernel_size=(1, config['model']['feature_dim'])
        )
        self.conv_embed_b = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2, # 8 channels per stream
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
        # Now each has 8 channels with 254 or 250 features
        
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
        self.use_adaptive_weights = config['training'].get('use_adaptive_weights', True)
        
        # --- 1. Adaptive Weighting Layer (Conditionally Initialized) ---
        # The adaptive weights are only created if the config flag is True.
        if self.use_adaptive_weights:
            if initial_weights is None:
                # Default to 0.5/0.5 weights if none are provided.
                logits1 = torch.zeros(2, dtype=torch.float32)
                logits2 = torch.zeros(2, dtype=torch.float32)
                logits3 = torch.zeros(2, dtype=torch.float32)
            else:
                def probs_to_logits(p_a, p_b):
                    """Converts a pair of probabilities into a pair of logits."""
                    p_a = max(p_a, 1e-6)
                    p_b = max(p_b, 1e-6)
                    total = p_a + p_b
                    p_a /= total
                    p_b /= total
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
        else:
            # If not using adaptive weights, set logits to None.
            # The forward pass will use fixed weights of 1.0.
            self.task1_logits = None
            self.task2_logits = None
            self.task3_logits = None
        
        # --- 2. Dual-Stream Embedding Layers (Shared across tasks) ---
        # Each stream has 8 channels with 254 or 250 features
        self.conv_embed_a = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2,  # 8 channels per stream
            out_channels=self.embed_dim, 
            kernel_size=(1, config['model']['feature_dim'])
        )
        self.conv_embed_b = nn.Conv2d(
            in_channels=config['model']['input_channels'] // 2,  # 8 channels per stream
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
        # --- MODIFICATION: Conditionally use adaptive or fixed weights ---
        if self.use_adaptive_weights:
            # Generate weights from logits using softmax
            w1 = self.weights_task1
            w2 = self.weights_task2
            w3 = self.weights_task3
            w1a, w1b = w1[0], w1[1]
            w2a, w2b = w2[0], w2[1]
            w3a, w3b = w3[0], w3[1]
        else:
            # Use fixed weights of 1.0 for simple stream addition
            w1a, w1b = 1.0, 1.0
            w2a, w2b = 1.0, 1.0
            w3a, w3b = 1.0, 1.0
        
        # Task 1: Predict Room A
        features_task1 = self._get_shared_features(src_a, src_b, w1a, w1b)
        output_a = self.head_a(features_task1)
        
        # Task 2: Predict Room B
        features_task2 = self._get_shared_features(src_a, src_b, w2a, w2b)
        output_b = self.head_b(features_task2)
        
        # Task 3: Predict Living Room
        features_task3 = self._get_shared_features(src_a, src_b, w3a, w3b)
        output_lr = self.head_lr(features_task3)
        
        return output_a, output_b, output_lr


class SeparateTaskTransformer(nn.Module):
    """
    A multi-task transformer model where each task has its own independent 
    transformer encoder body, but shares input embedding layers.
    """
    def __init__(self, config, initial_weights=None):
        super(SeparateTaskTransformer, self).__init__()
        
        self.embed_dim = config['model']['embed_dim']
        self.use_adaptive_weights = config['training'].get('use_adaptive_weights', True)
        
        # --- 1. Adaptive Weighting Layer (Conditionally Initialized) ---
        if self.use_adaptive_weights:
            if initial_weights is None:
                logits1 = torch.zeros(2, dtype=torch.float32)
                logits2 = torch.zeros(2, dtype=torch.float32)
                logits3 = torch.zeros(2, dtype=torch.float32)
            else:
                def probs_to_logits(p_a, p_b):
                    p_a = max(p_a, 1e-6)
                    p_b = max(p_b, 1e-6)
                    total = p_a + p_b
                    p_a /= total
                    p_b /= total
                    l_a = torch.log(torch.tensor(p_a / p_b, dtype=torch.float32))
                    l_b = torch.tensor(0.0, dtype=torch.float32)
                    return torch.stack([l_a, l_b])

                logits1 = probs_to_logits(initial_weights['Predict Room A']['weight_a'], initial_weights['Predict Room A']['weight_b'])
                logits2 = probs_to_logits(initial_weights['Predict Room B']['weight_a'], initial_weights['Predict Room B']['weight_b'])
                logits3 = probs_to_logits(initial_weights['Predict Living Room']['weight_a'], initial_weights['Predict Living Room']['weight_b'])

            self.task1_logits = nn.Parameter(logits1)
            self.task2_logits = nn.Parameter(logits2)
            self.task3_logits = nn.Parameter(logits3)
        else:
            self.task1_logits = None
            self.task2_logits = None
            self.task3_logits = None
        
        # --- 2. Dual-Stream Embedding Layers (Shared across tasks) ---
        # Each stream has 8 channels with 254 or 250 features
        self.conv_embed_a = nn.Conv2d(in_channels=config['model']['input_channels'] // 2, out_channels=self.embed_dim, kernel_size=(1, config['model']['feature_dim']))
        self.conv_embed_b = nn.Conv2d(in_channels=config['model']['input_channels'] // 2, out_channels=self.embed_dim, kernel_size=(1, config['model']['feature_dim']))
        
        # --- 3. Shared Positional Encoding & Separate Transformer Bodies ---
        self.positional_encoding = PositionalEncoding(self.embed_dim, config['model']['dropout'])

        def create_encoder():
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=self.embed_dim, 
                nhead=config['model']['num_heads'], 
                dim_feedforward=config['model']['hidden_dim'], 
                dropout=config['model']['dropout'],
                batch_first=True
            )
            return nn.TransformerEncoder(encoder_layers, num_layers=config['model']['num_layers'])

        self.transformer_encoder_a = create_encoder()
        self.transformer_encoder_b = create_encoder()
        self.transformer_encoder_lr = create_encoder()
        
        # --- 4. Multi-Head Classifier ---
        self.head_a = nn.Linear(self.embed_dim, 1)
        self.head_b = nn.Linear(self.embed_dim, 1)
        self.head_lr = nn.Linear(self.embed_dim, 1)

    @property
    def weights_task1(self):
        return torch.softmax(self.task1_logits, dim=0)
    
    @property
    def weights_task2(self):
        return torch.softmax(self.task2_logits, dim=0)

    @property
    def weights_task3(self):
        return torch.softmax(self.task3_logits, dim=0)

    def forward(self, src_a, src_b):
        if self.use_adaptive_weights:
            w1 = self.weights_task1
            w2 = self.weights_task2
            w3 = self.weights_task3
            w1a, w1b = w1[0], w1[1]
            w2a, w2b = w2[0], w2[1]
            w3a, w3b = w3[0], w3[1]
        else:
            w1a, w1b = 1.0, 1.0
            w2a, w2b = 1.0, 1.0
            w3a, w3b = 1.0, 1.0
        
        # Task 1: Predict Room A (Separate Path)
        x_a1 = self.conv_embed_a(src_a * w1a).squeeze(3).permute(0, 2, 1).contiguous()
        x_b1 = self.conv_embed_b(src_b * w1b).squeeze(3).permute(0, 2, 1).contiguous()
        features1 = self.positional_encoding(x_a1 + x_b1)
        encoded1 = self.transformer_encoder_a(features1)
        pooled1 = encoded1.mean(dim=1)
        output_a = self.head_a(pooled1)
        
        # Task 2: Predict Room B (Separate Path)
        x_a2 = self.conv_embed_a(src_a * w2a).squeeze(3).permute(0, 2, 1).contiguous()
        x_b2 = self.conv_embed_b(src_b * w2b).squeeze(3).permute(0, 2, 1).contiguous()
        features2 = self.positional_encoding(x_a2 + x_b2)
        encoded2 = self.transformer_encoder_b(features2)
        pooled2 = encoded2.mean(dim=1)
        output_b = self.head_b(pooled2)
        
        # Task 3: Predict Living Room (Separate Path)
        x_a3 = self.conv_embed_a(src_a * w3a).squeeze(3).permute(0, 2, 1).contiguous()
        x_b3 = self.conv_embed_b(src_b * w3b).squeeze(3).permute(0, 2, 1).contiguous()
        features3 = self.positional_encoding(x_a3 + x_b3)
        encoded3 = self.transformer_encoder_lr(features3)
        pooled3 = encoded3.mean(dim=1)
        output_lr = self.head_lr(pooled3)
        
        return output_a, output_b, output_lr


def get_model(config):
    """
    Model factory.
    """
    model_name = config['model']['name']
    
    # Dynamically set input_channels and feature_dim based on RSSI/Gain inclusion
    if config.get('include_rssi_gain', True):
        # 16 channels: 2 rooms * 8 channels
        actual_input_channels = 16
        # 254 features: 250 CSI + 2 RSSI + 2 Gain
        actual_feature_dim = 254
    else:
        # 16 channels: 2 rooms * 8 channels
        actual_input_channels = 16
        # 250 features: only CSI features
        actual_feature_dim = 250
    
    # Create a copy of config with updated parameters
    model_config = config.copy()
    model_config['model'] = model_config['model'].copy()
    model_config['model']['input_channels'] = actual_input_channels
    model_config['model']['feature_dim'] = actual_feature_dim
    
    if model_name == 'simple_transformer':
        return SimpleTransformer(model_config)
    elif model_name == 'dual_stream_transformer':
        return DualStreamTransformer(model_config)
    elif model_name == 'multi_task_transformer':
        # --- MODIFICATION: Read initial weights from the config file ---
        initial_weights = config.get('training', {}).get('initial_weights')
        return MultiTaskTransformer(model_config, initial_weights=initial_weights)
    elif model_name == 'separate_task_transformer':
        initial_weights = config.get('training', {}).get('initial_weights')
        return SeparateTaskTransformer(model_config, initial_weights=initial_weights)
    else:
        raise ValueError(f"Unknown model name: {model_name}") 