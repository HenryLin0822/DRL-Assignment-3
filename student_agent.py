import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import os

# Neural Network with Dueling Architecture
class DuelingDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        conv_out_size = self._get_conv_out(input_shape)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
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
        
        # Combine value and advantage to get Q-values using the dueling formula
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# Main Agent Class
class Agent:
    def __init__(self):
        # Initialize parameters
        self.device = torch.device("cpu")  # Use CPU for leaderboard submission
        self.input_shape = (4, 84, 84)    # 4 stacked frames, 84x84 each
        self.n_actions = 12               # COMPLEX_MOVEMENT has 12 actions
        
        # Initialize DQN networks
        self.policy_net = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
        
        # Load model if exists
        self.load_model()
        
        # Epsilon for exploration-exploitation trade-off
        self.epsilon = 0.01  # Very low epsilon for evaluation
        
        # Frame processing - maintain a stack of 4 frames
        self.frame_stack = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
    
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
    
    def act(self, observation):
        """Select action based on epsilon-greedy policy"""
        # Preprocess observation
        state = self.preprocess_observation(observation)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions - 1)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        return action
    
    def load_model(self, model_path="1300.pth"):
        """Load trained model if it exists"""
        if os.path.exists(model_path):
            try:
                # Load without weights_only option
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Support both newer format (with model_state_dict) and direct state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.policy_net.load_state_dict(checkpoint)
                
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")