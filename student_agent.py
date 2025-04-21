import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os

# Noisy Linear Layer for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        # Initialize parameters
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        # Sample new noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product
        self.weight_epsilon.copy_(torch.ger(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        # Helper function for factorized Gaussian noise
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def forward(self, x):
        # For inference, we'll enable noise unlike typical evaluation
        # This helps avoid deterministic behavior
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return F.linear(x, weight, bias)

# Neural Network with Dueling Architecture and Noisy Networks
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions, noisy=True):
        super(DuelingDQN, self).__init__()
        
        self.noisy = noisy
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # Value stream with noisy networks
        if noisy:
            self.value_stream = nn.Sequential(
                NoisyLinear(conv_out_size, 512),
                nn.ReLU(),
                NoisyLinear(512, 1)
            )
            
            # Advantage stream with noisy networks
            self.advantage_stream = nn.Sequential(
                NoisyLinear(conv_out_size, 512),
                nn.ReLU(),
                NoisyLinear(512, n_actions)
            )
        else:
            # Regular linear layers for value stream
            self.value_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            
            # Regular linear layers for advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions)
            )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine value and advantage to get Q-values
        return value + advantage - advantage.mean(dim=1, keepdim=True)
    
    def reset_noise(self):
        # Reset noise for all noisy layers - now used during inference
        if not self.noisy:
            return
            
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# Main Agent Class
class Agent:
    def __init__(self):
        # Initialize parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = (4, 84, 84)    # 4 stacked frames, 84x84 each
        self.n_actions = 12               # COMPLEX_MOVEMENT has 12 actions
        
        # Initialize DQN networks with noisy layers
        self.policy_net = DuelingDQN(self.input_shape, self.n_actions, noisy=True).to(self.device)
        
        # Set to training mode to keep noise enabled (important change!)
        self.policy_net.train()
        
        # Load model if exists
        self.load_model()
        
        # For testing, add the same randomness we had in training
        self.epsilon = 0.05  # 5% random action probability like in training
        self.random_action_steps = 0  # Force random actions at the beginning
        self.initial_random_actions = 3  # Force several random actions at the start
        
        # Frame processing - maintain a stack of 4 frames
        self.frame_stack = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
        
        # Store action history to detect repetitive behaviors
        self.action_history = deque(maxlen=10)
        self.stuck_threshold = 5  # How many identical actions in a row before we consider "stuck"
    
    def preprocess_observation(self, observation):
        """Convert and stack frames for input to DQN"""
        # Check if observation is already a LazyFrames object (which is what the environment returns)
        if hasattr(observation, "_frames"):
            # Convert LazyFrames to numpy array
            observation = np.array(observation)
            return observation
        
        # If observation is already the right shape, just return it
        if len(observation.shape) == 3 and observation.shape[0] == 4:
            return observation
            
        # Handle RGB image (unlikely to reach this code with gym wrappers)
        if len(observation.shape) == 3 and observation.shape[2] == 3:
            # Convert to grayscale (simple average method)
            gray_frame = np.mean(observation, axis=2).astype(np.float32)
            
            # Resize to 84x84 if needed using numpy
            if gray_frame.shape != (84, 84):
                # Simple bilinear resize using numpy
                h, w = gray_frame.shape
                y_indices = np.linspace(0, h-1, 84).astype(np.float32)
                x_indices = np.linspace(0, w-1, 84).astype(np.float32)
                
                y_floor = np.floor(y_indices).astype(np.int32)
                y_ceil = np.minimum(y_floor + 1, h - 1)
                x_floor = np.floor(x_indices).astype(np.int32)
                x_ceil = np.minimum(x_floor + 1, w - 1)
                
                y_frac = y_indices - y_floor
                x_frac = x_indices - x_floor
                
                resized = np.zeros((84, 84), dtype=np.float32)
                for i in range(84):
                    for j in range(84):
                        # Bilinear interpolation
                        top_left = gray_frame[y_floor[i], x_floor[j]]
                        top_right = gray_frame[y_floor[i], x_ceil[j]]
                        bottom_left = gray_frame[y_ceil[i], x_floor[j]]
                        bottom_right = gray_frame[y_ceil[i], x_ceil[j]]
                        
                        top = top_left * (1 - x_frac[j]) + top_right * x_frac[j]
                        bottom = bottom_left * (1 - x_frac[j]) + bottom_right * x_frac[j]
                        
                        resized[i, j] = top * (1 - y_frac[i]) + bottom * y_frac[i]
                        
                gray_frame = resized
            
            # Normalize
            normalized_frame = gray_frame / 255.0
            
            # Update frame stack
            self.frame_stack.append(normalized_frame)
            
            # Convert stack to numpy array for network input
            stacked_frames = np.array(self.frame_stack)
            return stacked_frames
        
        # If we get here, the observation format is unexpected
        # Let's try to handle it gracefully by returning a zeros array
        print(f"Warning: Unexpected observation shape {observation.shape}")
        return np.zeros(self.input_shape, dtype=np.float32)
    
    def is_stuck(self):
        """Check if the agent is repeating the same action"""
        if len(self.action_history) < self.stuck_threshold:
            return False
        
        # Check if the last N actions are the same
        return self.action_history.count(self.action_history[-1]) >= self.stuck_threshold
    
    def act(self, observation):
        """Select action using the same strategy as during training"""
        # Reset noise for the noisy layers
        self.policy_net.reset_noise()
        
        # Preprocess observation
        state = self.preprocess_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Force random actions at the beginning
        if self.random_action_steps < self.initial_random_actions:
            action = random.randint(0, self.n_actions - 1)
            self.random_action_steps += 1
        # Occasionally take random actions (like in training)
        elif random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        # If stuck in a repetitive pattern, take random action
        elif self.is_stuck():
            action = random.randint(0, self.n_actions - 1)
            #print("Detected repetitive behavior, taking random action...")
        # Otherwise use the network with active noise
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        # Store the action in history
        self.action_history.append(action)
        
        return action
    
    def load_model(self, model_path="mario_model.pth"):
        """Load trained model if it exists"""
        if os.path.exists(model_path):
            try:
                # Load model
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Support both formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.policy_net.load_state_dict(checkpoint)
                
                print(f"Loaded model from {model_path}")
                
                # Keep in training mode to enable noise
                self.policy_net.train()
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"Warning: Model file {model_path} not found.")