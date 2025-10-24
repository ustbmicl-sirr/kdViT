"""
Policy Network for RL-PyramidKD

Implements the PPO-based policy network that learns to select
pyramid layers dynamically for each sample.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class PolicyNetwork(nn.Module):
    """
    Policy Network for Layer Selection

    Architecture:
        - State Encoder: Linear layers with LayerNorm
        - LSTM: For sequential decision making
        - Action Head: Outputs layer selection probabilities [P2, P3, P4, P5]
        - Value Head: Outputs state value for actor-critic

    Args:
        state_dim: Dimension of state vector (default: 1542)
            = 512 (global) + 4*256 (pyramid) + 1 (loss) + 4 (selected) + 1 (budget)
        hidden_dim: Hidden dimension (default: 256)
        num_layers: Number of pyramid layers (default: 4)
        use_lstm: Whether to use LSTM for sequential decisions (default: True)
    """

    def __init__(self, state_dim=1542, hidden_dim=256, num_layers=4, use_lstm=True):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm = use_lstm

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # LSTM for sequential decision (optional)
        if self.use_lstm:
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=2,
                batch_first=True
            )
            self.hidden_state = None

        # Action head (outputs selection probability for each layer)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_layers),
            nn.Sigmoid()  # Output [0, 1] for each layer
        )

        # Value head (for actor-critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, state, hidden=None):
        """
        Forward pass

        Args:
            state: State tensor [batch_size, state_dim]
            hidden: Hidden state for LSTM (optional)

        Returns:
            action_probs: Layer selection probabilities [batch_size, num_layers]
            value: State value [batch_size, 1]
            hidden: Updated hidden state (if LSTM is used)
        """
        # Encode state
        h = self.state_encoder(state)  # [B, hidden_dim]

        # LSTM (maintains history)
        if self.use_lstm:
            if hidden is None:
                hidden = self._init_hidden(state.size(0), state.device)

            h = h.unsqueeze(1)  # [B, 1, hidden_dim]
            h, hidden = self.lstm(h, hidden)
            h = h.squeeze(1)  # [B, hidden_dim]

        # Predict action probabilities
        action_probs = self.action_head(h)  # [B, num_layers]

        # Predict state value
        value = self.value_head(h)  # [B, 1]

        if self.use_lstm:
            return action_probs, value, hidden
        else:
            return action_probs, value

    def _init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state"""
        h0 = torch.zeros(2, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(2, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

    def select_action(self, state, hidden=None, deterministic=False):
        """
        Select action based on current state

        Args:
            state: State tensor [batch_size, state_dim]
            hidden: LSTM hidden state (optional)
            deterministic: If True, use greedy selection; else sample (default: False)

        Returns:
            action: Binary action vector [batch_size, num_layers]
            log_prob: Log probability of action [batch_size]
            value: State value [batch_size, 1]
            hidden: Updated hidden state (if LSTM)
        """
        # Forward pass
        if self.use_lstm:
            action_probs, value, hidden = self.forward(state, hidden)
        else:
            action_probs, value = self.forward(state)

        # Create Bernoulli distribution
        dist = Bernoulli(action_probs)

        if deterministic:
            # Greedy selection (for evaluation)
            action = (action_probs > 0.5).float()
        else:
            # Sample action (for training)
            action = dist.sample()

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1)  # [B]

        # Ensure at least one layer is selected
        action = self._ensure_valid_action(action, action_probs)

        if self.use_lstm:
            return action, log_prob, value, hidden
        else:
            return action, log_prob, value

    def _ensure_valid_action(self, action, action_probs):
        """
        Ensure at least one layer is selected

        If no layer is selected, force select the layer with highest probability
        """
        # Check if any layer is selected
        num_selected = action.sum(dim=-1, keepdim=True)  # [B, 1]

        # If no layer selected, select the one with highest prob
        no_selection_mask = (num_selected == 0).float()  # [B, 1]

        if no_selection_mask.sum() > 0:
            # Get the layer with max probability
            max_prob_layer = torch.argmax(action_probs, dim=-1)  # [B]

            # Create one-hot for max prob layer
            forced_action = F.one_hot(max_prob_layer, self.num_layers).float()  # [B, num_layers]

            # Apply correction where needed
            action = action * (1 - no_selection_mask) + forced_action * no_selection_mask

        return action

    def evaluate_actions(self, states, actions, hidden=None):
        """
        Evaluate actions (for PPO update)

        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, num_layers]
            hidden: LSTM hidden state (optional)

        Returns:
            log_probs: Log probabilities [batch_size]
            values: State values [batch_size, 1]
            entropy: Action entropy [batch_size]
        """
        # Forward pass
        if self.use_lstm:
            action_probs, values, _ = self.forward(states, hidden)
        else:
            action_probs, values = self.forward(states)

        # Create distribution
        dist = Bernoulli(action_probs)

        # Compute log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1)  # [B]

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1)  # [B]

        return log_probs, values, entropy


class StateEncoder(nn.Module):
    """
    State encoder for extracting sample features

    Used to convert raw images to state representation
    """

    def __init__(self, backbone='resnet18', pretrained=True):
        super().__init__()

        # Load pretrained backbone
        if backbone == 'resnet18':
            import torchvision.models as models
            resnet = models.resnet18(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.global_dim = 512
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")

    def forward(self, images):
        """
        Extract global features from images

        Args:
            images: Input images [B, 3, H, W]

        Returns:
            global_features: Global features [B, global_dim]
        """
        features = self.encoder(images)  # [B, global_dim, 1, 1]
        global_features = features.flatten(1)  # [B, global_dim]
        return global_features


# Example usage
if __name__ == "__main__":
    # Test PolicyNetwork
    batch_size = 8
    state_dim = 1542

    policy = PolicyNetwork(state_dim=state_dim, hidden_dim=256)
    state = torch.randn(batch_size, state_dim)

    # Forward pass
    action_probs, value = policy(state)
    print(f"Action probs shape: {action_probs.shape}")  # [8, 4]
    print(f"Value shape: {value.shape}")  # [8, 1]

    # Select action
    action, log_prob, value = policy.select_action(state)
    print(f"Action shape: {action.shape}")  # [8, 4]
    print(f"Log prob shape: {log_prob.shape}")  # [8]
    print(f"Action example: {action[0]}")  # e.g., [1, 0, 1, 1]

    # Evaluate actions
    log_probs, values, entropy = policy.evaluate_actions(state, action)
    print(f"Entropy: {entropy.mean().item():.4f}")

    print("\nPolicyNetwork test passed!")
