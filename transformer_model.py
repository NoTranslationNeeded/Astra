"""
Custom Transformer Model for Poker AI
Replaces LSTM with Transformer for better betting sequence understanding
"""
import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer
    Allows model to learn position information in sequence
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            x with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class TransformerPokerModel(TorchModelV2, nn.Module):
    """
    Custom Transformer model for poker
    
    Architecture:
        Input (60 dims) → Embedding (d_model)
        → Positional Encoding
        → Transformer Encoder (num_layers blocks)
        → Sequence Pooling (last token)
        → Policy Head (actions) + Value Head (value estimate)
    """
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Get custom config
        custom_config = model_config.get("custom_model_config", {})
        self.d_model = custom_config.get("d_model", 128)
        self.nhead = custom_config.get("nhead", 8)
        self.num_layers = custom_config.get("num_layers", 4)
        self.dim_feedforward = custom_config.get("dim_feedforward", 512)
        self.dropout = custom_config.get("dropout", 0.1)
        self.max_seq_len = custom_config.get("max_seq_len", 20)
        
        # Input dimension from observation space
        self.obs_dim = obs_space.shape[0]  # Should be 60
        
        # Input embedding layer (60 → d_model)
        self.input_embedding = nn.Linear(self.obs_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True  # (batch, seq, feature) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Policy head (actions)
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, num_outputs)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, 1)
        )
        
        # Store value output
        self._value_out = None
        
        # State for storing sequence history
        self._seq_lens = None
        
    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass
        
        Args:
            input_dict: Dict with 'obs' key containing observations
            state: RNN state (not used for Transformer)
            seq_lens: Sequence lengths for each batch element
            
        Returns:
            (logits, state)
        """
        obs = input_dict["obs"].float()
        
        # Store sequence lengths
        self._seq_lens = seq_lens
        
        # Check if we have sequential data (training) or single step (inference)
        if len(obs.shape) == 3:
            # Sequential: (batch, time, features)
            batch_size, time_steps, _ = obs.shape
            
            # Embed input
            embedded = self.input_embedding(obs)  # (batch, time, d_model)
            
            # Add positional encoding
            embedded = self.pos_encoder(embedded)
            
            # Create attention mask for padding
            # This prevents attention to padding tokens
            mask = self._create_padding_mask(batch_size, time_steps, seq_lens)
            
            # Transformer encoding
            encoded = self.transformer_encoder(embedded, src_key_padding_mask=mask)
            
            # Take last non-padded token for each sequence
            # Use seq_lens to index the correct position
            last_outputs = []
            for i, length in enumerate(seq_lens):
                last_outputs.append(encoded[i, length - 1, :])
            output = torch.stack(last_outputs)
            
        else:
            # Single step: (batch, features)
            batch_size = obs.shape[0]
            
            # Embed and add positional encoding
            embedded = self.input_embedding(obs).unsqueeze(1)  # (batch, 1, d_model)
            embedded = self.pos_encoder(embedded)
            
            # Pass through transformer
            encoded = self.transformer_encoder(embedded)
            
            # Take the single output
            output = encoded[:, 0, :]  # (batch, d_model)
        
        # Compute action logits
        logits = self.policy_head(output)
        
        # Compute value estimate
        self._value_out = self.value_head(output).squeeze(-1)
        
        return logits, state
    
    @override(TorchModelV2)
    def value_function(self):
        """
        Returns the value function estimate for the latest forward pass
        """
        assert self._value_out is not None, "must call forward() first"
        return self._value_out
    
    def _create_padding_mask(self, batch_size, time_steps, seq_lens):
        """
        Create padding mask for Transformer
        
        Args:
            batch_size: Number of sequences in batch
            time_steps: Maximum sequence length
            seq_lens: Actual length of each sequence
            
        Returns:
            mask: Boolean tensor (batch, time_steps) where True = padding
        """
        mask = torch.zeros(batch_size, time_steps, dtype=torch.bool, device=seq_lens.device)
        for i, length in enumerate(seq_lens):
            if length < time_steps:
                mask[i, length:] = True
        return mask
    
    def get_initial_state(self):
        """
        Transformer doesn't need recurrent state, but RLlib expects this method
        """
        return []


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer Poker Model...")
    
    from gymnasium import spaces
    
    # Create mock spaces
    obs_space = spaces.Box(low=0, high=1, shape=(60,), dtype=np.float32)
    action_space = spaces.Discrete(5)
    
    # Model config
    model_config = {
        "custom_model_config": {
            "d_model": 128,
            "nhead": 8,
            "num_layers": 4,
            "dim_feedforward": 512,
            "dropout": 0.1,
            "max_seq_len": 20,
        }
    }
    
    # Create model
    model = TransformerPokerModel(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=5,
        model_config=model_config,
        name="test_transformer"
    )
    
    print(f"✅ Model created successfully!")
    print(f"   - d_model: {model.d_model}")
    print(f"   - nhead: {model.nhead}")
    print(f"   - num_layers: {model.num_layers}")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test single step forward (inference mode)
    print("\n Testing single-step forward pass...")
    batch_size = 4
    obs_single = torch.randn(batch_size, 60)
    logits, _ = model({"obs": obs_single}, [], [1] * batch_size)
    value = model.value_function()
    
    print(f"✅ Single-step forward pass successful!")
    print(f"   - Logits shape: {logits.shape} (expected: {batch_size}, 5)")
    print(f"   - Value shape: {value.shape} (expected: {batch_size})")
    
    # Test sequential forward (training mode)
    print("\nTesting sequential forward pass...")
    time_steps = 10
    obs_seq = torch.randn(batch_size, time_steps, 60)
    seq_lens = torch.tensor([10, 8, 6, 10])  # Variable length sequences
    logits, _ = model({"obs": obs_seq}, [], seq_lens)
    value = model.value_function()
    
    print(f"✅ Sequential forward pass successful!")
    print(f"   - Logits shape: {logits.shape} (expected: {batch_size}, 5)")
    print(f"   - Value shape: {value.shape} (expected: {batch_size})")
    print(f"   - Sequence lengths: {seq_lens.tolist()}")
    
    print("\n" + "=" * 80)
    print("✅ All tests passed! Transformer model is ready.")
    print("=" * 80)
