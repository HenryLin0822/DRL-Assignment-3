import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
import cv2

# Define COMPLEX_MOVEMENT mapping
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down'],
    ['up'],
]

# Dueling DQN architecture
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
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        return value + advantage - advantage.mean(dim=1, keepdim=True)

# Main Agent Class
class Agent:
    def __init__(self):
        # Initialize parameters
        self.device = torch.device("cpu")  # Use CPU for leaderboard submission
        self.input_shape = (4, 84, 84)    # 4 stacked frames, 84x84 each
        self.n_actions = 12               # COMPLEX_MOVEMENT has 12 actions
        
        # Use Dueling DQN architecture for better performance
        self.policy_net = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net = DuelingDQN(self.input_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Load model if exists
        self.load_model()
        
        # Epsilon for exploration
        self.epsilon = 0.05  # Low epsilon for evaluation
        
        # Frame processing - maintain a stack of 4 frames
        self.frame_stack = deque([np.zeros((84, 84), dtype=np.float32) for _ in range(4)], maxlen=4)
    
    def preprocess_observation(self, observation):
        """Convert and stack frames for input to DQN"""
        # Check if observation is already processed
        if isinstance(observation, deque) and len(observation) == 4:
            frames = np.array(observation)
            return frames
        
        # Process raw observation (RGB image)
        if len(observation.shape) == 3:  # Handle RGB image
            # Convert to grayscale
            gray_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) if observation.shape[2] == 3 else observation
            
            # Resize to 84x84 if needed
            if gray_frame.shape != (84, 84):
                gray_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
            
            # Normalize
            normalized_frame = gray_frame.astype(np.float32) / 255.0
            
            # Update frame stack
            self.frame_stack.append(normalized_frame)
            
        # Convert stack to numpy array for network input
        stacked_frames = np.array(self.frame_stack)
        return stacked_frames
    
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
    
    def load_model(self, model_path="./checkpoints/best_mario_model.pth"):
        """Load trained model if it exists"""
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {model_path}")
                
                # Print model info if available
                if 'avg_reward' in checkpoint:
                    print(f"Model average reward: {checkpoint['avg_reward']:.2f}")
                if 'episode' in checkpoint:
                    print(f"Model saved at episode: {checkpoint['episode']}")
                
            except Exception as e:
                print(f"Error loading model: {e}")