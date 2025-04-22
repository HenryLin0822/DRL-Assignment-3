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
        
        # Set to training mode to keep noise enabled
        self.policy_net.train()
        
        # Load model if exists - try multiple potential paths
        self.model_loaded = self.load_model(["mario_model.pth", "./mario_model.pth", "../mario_model.pth"])
        
        # For testing, add the same randomness we had in training
        self.epsilon = 0.05  # 5% random action probability like in training
        self.random_action_steps = 0  # Force random actions at the beginning
        self.initial_random_actions = 3  # Force several random actions at the start
        
        # Frame processing - maintain a stack of 4 frames
        self.frame_stack = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
        self.frames_initialized = False
        
        # Store action history to detect repetitive behaviors
        self.action_history = deque(maxlen=10)
        self.stuck_threshold = 5  # How many identical actions in a row before we consider "stuck"
    
    def preprocess_observation(self, observation):
        """More robust observation preprocessing"""
        try:
            # First try to handle LazyFrames from FrameStack wrapper
            if hasattr(observation, "_frames"):
                observation = np.array(observation)
                if observation.shape == (4, 84, 84):
                    # Already the right shape
                    self.frames_initialized = True
                    return observation
            
            # If observation is already the right shape, just return it
            if isinstance(observation, np.ndarray) and observation.shape == (4, 84, 84):
                self.frames_initialized = True
                return observation
                
            # Handle single frame observation (should be uncommon with wrappers)
            if isinstance(observation, np.ndarray):
                if len(observation.shape) == 3 and (observation.shape[0] == 84 or observation.shape[2] == 1):
                    # Already grayscale
                    frame = observation.squeeze()
                    if frame.shape != (84, 84):
                        # Resize using simple numpy operations
                        from skimage import transform
                        frame = transform.resize(frame, (84, 84))
                        frame = (frame * 255).astype(np.uint8)
                    
                    # Normalize
                    normalized_frame = frame.astype(np.float32) / 255.0
                    
                    # Update frame stack
                    self.frame_stack.append(normalized_frame)
                    self.frames_initialized = True
                    
                    # Convert stack to numpy array for network input
                    stacked_frames = np.array(self.frame_stack)
                    return stacked_frames
                
                elif len(observation.shape) == 3 and observation.shape[2] == 3:
                    # RGB image
                    # Convert to grayscale
                    gray_frame = np.mean(observation, axis=2).astype(np.float32)
                    
                    # Resize if needed
                    if gray_frame.shape != (84, 84):
                        from skimage import transform
                        gray_frame = transform.resize(gray_frame, (84, 84))
                        gray_frame = (gray_frame * 255).astype(np.uint8)
                    
                    # Normalize
                    normalized_frame = gray_frame.astype(np.float32) / 255.0
                    
                    # Update frame stack
                    self.frame_stack.append(normalized_frame)
                    self.frames_initialized = True
                    
                    # Convert stack to numpy array for network input
                    stacked_frames = np.array(self.frame_stack)
                    return stacked_frames
            
            # If we get here, the observation format is unexpected
            print(f"Warning: Unexpected observation type {type(observation)} or shape {getattr(observation, 'shape', 'unknown')}")
            return np.zeros(self.input_shape, dtype=np.float32)
            
        except Exception as e:
            print(f"Error in preprocess_observation: {e}")
            # Return zeros as fallback
            return np.zeros(self.input_shape, dtype=np.float32)
    
    def is_stuck(self):
        """Check if the agent is repeating the same action"""
        if len(self.action_history) < self.stuck_threshold:
            return False
        
        # Check if the last N actions are the same
        return self.action_history.count(self.action_history[-1]) >= self.stuck_threshold
    
    def act(self, observation):
        """Select action using the same strategy as during training"""
        try:
            # Reset noise for the noisy layers
            self.policy_net.reset_noise()
            
            # Preprocess observation
            state = self.preprocess_observation(observation)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Force random actions at the beginning or if frames not initialized properly
            if self.random_action_steps < self.initial_random_actions or not self.frames_initialized:
                action = random.randint(0, self.n_actions - 1)
                self.random_action_steps += 1
            # Occasionally take random actions (like in training)
            elif random.random() < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            # If stuck in a repetitive pattern, take random action
            elif self.is_stuck():
                action = random.randint(0, self.n_actions - 1)
            # Otherwise use the network with active noise
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            # Store the action in history
            self.action_history.append(action)
            
            return action
            
        except Exception as e:
            print(f"Error in act method: {e}")
            # Return a default action as fallback
            return 1  # Default to 'right' action
    
    def load_model(self, model_paths):
        """Load trained model from multiple possible paths"""
        if isinstance(model_paths, str):
            model_paths = [model_paths]
            
        for path in model_paths:
            if os.path.exists(path):
                try:
                    # Load model
                    checkpoint = torch.load(path, map_location=self.device)
                    
                    # Support both formats
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.policy_net.load_state_dict(checkpoint)
                    
                    print(f"Successfully loaded model from {path}")
                    
                    # Keep in training mode to enable noise
                    self.policy_net.train()
                    return True
                    
                except Exception as e:
                    print(f"Error loading model from {path}: {e}")
        
        print("Warning: Model file not found in any specified path.")
        return False