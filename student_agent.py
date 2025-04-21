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
        
        # Initialize DQN network
        self.policy_net = DuelingDQN(self.input_shape, self.n_actions, noisy=True).to(self.device)
        self.policy_net.train()  # Keep in training mode for noise
        
        # Load model if exists
        self.load_model()
        
        # Exploration parameters
        self.epsilon = 0.05
        
        # Initialize frame stack with zeros - crucial for handling raw observations
        self.frame_stack = deque(maxlen=4)
        for _ in range(4):
            self.frame_stack.append(np.zeros((84, 84), dtype=np.float32))
        
        # Track action history for detecting stuck behavior
        self.action_history = deque(maxlen=10)
        self.stuck_threshold = 5
    
    def preprocess_frame(self, frame):
        """Process a single frame to 84x84 grayscale normalized"""
        try:
            # Handle different possible frame formats
            
            # If already grayscale and correct size
            if isinstance(frame, np.ndarray) and frame.shape == (84, 84):
                # Just normalize if needed
                if frame.dtype == np.uint8:
                    return frame.astype(np.float32) / 255.0
                return frame
            
            # If RGB frame (typical from raw environment)
            if isinstance(frame, np.ndarray) and len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert to grayscale using weighted average (matching human perception)
                gray = 0.2989 * frame[:, :, 0] + 0.5870 * frame[:, :, 1] + 0.1140 * frame[:, :, 2]
                
                # Resize to 84x84 using numpy (avoid external dependencies)
                h, w = gray.shape
                if h != 84 or w != 84:
                    # Simple resize using numpy
                    y = np.linspace(0, h-1, 84)
                    x = np.linspace(0, w-1, 84)
                    
                    # Get integer and fractional parts
                    y_floor = np.floor(y).astype(int)
                    y_ceil = np.minimum(y_floor + 1, h - 1)
                    x_floor = np.floor(x).astype(int)
                    x_ceil = np.minimum(x_floor + 1, w - 1)
                    
                    y_frac = y - y_floor
                    x_frac = x - x_floor
                    
                    # Prepare output array
                    resized = np.zeros((84, 84), dtype=np.float32)
                    
                    # Bilinear interpolation
                    for i in range(84):
                        for j in range(84):
                            tl = gray[y_floor[i], x_floor[j]]
                            tr = gray[y_floor[i], x_ceil[j]]
                            bl = gray[y_ceil[i], x_floor[j]]
                            br = gray[y_ceil[i], x_ceil[j]]
                            
                            # Calculate interpolated value
                            resized[i, j] = (tl * (1 - x_frac[j]) * (1 - y_frac[i]) +
                                            tr * x_frac[j] * (1 - y_frac[i]) +
                                            bl * (1 - x_frac[j]) * y_frac[i] +
                                            br * x_frac[j] * y_frac[i])
                    
                    gray = resized
                
                # Normalize to [0, 1]
                return gray.astype(np.float32) / 255.0
            
            # If already normalized grayscale but wrong size
            if isinstance(frame, np.ndarray) and len(frame.shape) == 2:
                # Resize using numpy
                h, w = frame.shape
                if h != 84 or w != 84:
                    # Simple resize using numpy
                    y = np.linspace(0, h-1, 84)
                    x = np.linspace(0, w-1, 84)
                    
                    # Get integer and fractional parts
                    y_floor = np.floor(y).astype(int)
                    y_ceil = np.minimum(y_floor + 1, h - 1)
                    x_floor = np.floor(x).astype(int)
                    x_ceil = np.minimum(x_floor + 1, w - 1)
                    
                    y_frac = y - y_floor
                    x_frac = x - x_floor
                    
                    # Prepare output array
                    resized = np.zeros((84, 84), dtype=np.float32)
                    
                    # Bilinear interpolation
                    for i in range(84):
                        for j in range(84):
                            tl = frame[y_floor[i], x_floor[j]]
                            tr = frame[y_floor[i], x_ceil[j]]
                            bl = frame[y_ceil[i], x_floor[j]]
                            br = frame[y_ceil[i], x_ceil[j]]
                            
                            # Calculate interpolated value
                            resized[i, j] = (tl * (1 - x_frac[j]) * (1 - y_frac[i]) +
                                            tr * x_frac[j] * (1 - y_frac[i]) +
                                            bl * (1 - x_frac[j]) * y_frac[i] +
                                            br * x_frac[j] * y_frac[i])
                    
                    frame = resized
                
                # Ensure normalized
                if frame.max() > 1.0:
                    return frame.astype(np.float32) / 255.0
                return frame.astype(np.float32)
            
            # Special case for FrameStack observations
            if hasattr(frame, "_frames"):
                try:
                    # Try to get the most recent frame
                    single_frame = frame[-1]
                    return self.preprocess_frame(single_frame)
                except:
                    pass  # Fall through to fallback
            
            # Fallback for unexpected formats
            print(f"Warning: Unexpected frame format {type(frame)} with shape {getattr(frame, 'shape', 'unknown')}")
            return np.zeros((84, 84), dtype=np.float32)
            
        except Exception as e:
            print(f"Error in preprocess_frame: {e}")
            return np.zeros((84, 84), dtype=np.float32)
    
    def preprocess_observation(self, observation):
        """Handle either frame-stacked or raw observations"""
        try:
            # Case 1: Already stacked frames with shape (4, 84, 84)
            if isinstance(observation, np.ndarray) and observation.shape == (4, 84, 84):
                return observation
            
            # Case 2: LazyFrames from FrameStack wrapper
            if hasattr(observation, "_frames"):
                try:
                    stacked = np.array(observation)
                    if stacked.shape == (4, 84, 84):
                        return stacked
                except:
                    pass  # Fall through to other cases
            
            # Case 3: Single frame that needs to be added to our frame stack
            # (This is likely what's happening in the submission environment)
            
            # Process the new frame
            processed_frame = self.preprocess_frame(observation)
            
            # Add to our frame stack
            self.frame_stack.append(processed_frame)
            
            # Convert deque to numpy array for the network
            stacked_frames = np.array(self.frame_stack)
            
            return stacked_frames
            
        except Exception as e:
            print(f"Error in preprocess_observation: {e}")
            # Return the existing frame stack as fallback
            return np.array(self.frame_stack)
    
    def is_stuck(self):
        """Check if the agent is repeating the same action"""
        if len(self.action_history) < self.stuck_threshold:
            return False
        
        # Check if the last N actions are the same
        return self.action_history.count(self.action_history[-1]) >= self.stuck_threshold
    
    def act(self, observation):
        """Select action based on the current observation"""
        try:
            # Reset noise for the policy network
            self.policy_net.reset_noise()
            
            # Preprocess the observation
            state = self.preprocess_observation(observation)
            
            # Convert to tensor and get action from network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Occasionally take random actions for exploration
            if random.random() < self.epsilon:
                action = random.randint(0, self.n_actions - 1)
            # If stuck in a pattern, take random action
            elif self.is_stuck():
                action = random.randint(0, self.n_actions - 1)
            # Otherwise use the network with active noise
            else:
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
            
            # Store action in history
            self.action_history.append(action)
            
            return action
            
        except Exception as e:
            print(f"Error in act method: {e}")
            # Return right movement as fallback
            return 1  # 'right' action
    
    def load_model(self, model_path="mario_model.pth"):
        """Load trained model if it exists"""
        try:
            if os.path.exists(model_path):
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
                return True
            else:
                print(f"Warning: Model file {model_path} not found.")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False