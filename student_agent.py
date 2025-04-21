import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from skimage import transform

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
        # Apply noise to weights and biases
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        
        return F.linear(x, weight, bias)

# Dueling DQN Network with Noisy Networks
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
        # Reset noise for all noisy layers - used during inference for exploration
        if not self.noisy:
            return
            
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# Main Agent Class
class Agent:
    def __init__(self, debug=False):
        # Debug flag for printing information
        self.debug = debug
        
        # Initialize parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.debug:
            print(f"Using device: {self.device}")
            
        self.input_shape = (4, 84, 84)    # 4 stacked frames, 84x84 each
        self.n_actions = 12               # COMPLEX_MOVEMENT has 12 actions
        
        # Initialize DQN network
        self.policy_net = DuelingDQN(self.input_shape, self.n_actions, noisy=True).to(self.device)
        self.policy_net.train()  # Keep in training mode for noise
        
        # Load model
        self.load_model("mario_model.pth")
        
        # Initialize frame processing variables
        self.reset_frame_stack()
        
        # For exploration - use a higher epsilon for raw observations
        self.epsilon = 0.075  # Increased from 0.05 for more exploration
        
        # Starting sequence - improved to better handle raw environments
        # Mix of right, jump, and run actions to build momentum
        self.start_actions = [1, 2, 4, 1, 2, 4, 1]  # right, right+jump, right+jump+run
        self.start_action_idx = 0
        
        # Frame skipping (to match training environment)
        self.frame_skip = 4
        self.current_frame = 0
    
    def reset_frame_stack(self):
        """Initialize frame stack with empty frames"""
        if self.debug:
            print("Initializing frame stack")
            
        self.frame_stack = deque(maxlen=4)
        # Fill with black frames initially
        empty_frame = np.zeros((84, 84), dtype=np.float32)
        for _ in range(4):
            self.frame_stack.append(empty_frame.copy())
    
    def preprocess_observation(self, observation):
        """Process observations to match training pipeline exactly"""
        # Skip frames to match training environment (process every 4th frame)
        self.current_frame = (self.current_frame + 1) % self.frame_skip
        process_frame = (self.current_frame == 0)
        
        # Case 1: Already processed observations (4, 84, 84)
        if isinstance(observation, np.ndarray) and observation.shape == (4, 84, 84):
            if self.debug:
                print("Case 1: Already processed observation (4, 84, 84)")
            return observation
            
        # Case 2: LazyFrames object from FrameStack wrapper
        if hasattr(observation, "_frames"):
            if self.debug:
                print("Case 2: LazyFrames object")
            return np.array(observation)
            
        # Case 3: Raw RGB frame (240, 256, 3)
        if isinstance(observation, np.ndarray) and len(observation.shape) == 3 and observation.shape[2] == 3:
            if self.debug:
                print(f"Case 3: Raw RGB frame {observation.shape}, process_frame={process_frame}")
            
            if process_frame:  # Only process every nth frame to match training
                try:
                    # 1. Convert to grayscale - match exactly how training does it
                    # Use weighted RGB to grayscale conversion (same weights used in training)
                    gray = np.dot(observation[..., :3], [0.299, 0.587, 0.114])
                    
                    # 2. Crop to focus on gameplay area - removes UI elements
                    # Adjust these values if needed based on the game
                    cropped = gray[40:220, :]  # Remove score and status bars
                    
                    # 3. Resize to 84x84 using the same method as in training
                    # Use anti_aliasing for better quality
                    resized = transform.resize(cropped, (84, 84), anti_aliasing=True)
                    
                    # 4. Normalize to [0, 1] range
                    normalized = resized.astype(np.float32)
                    
                    # 5. Add to frame stack
                    self.frame_stack.append(normalized)
                    
                    if self.debug:
                        print(f"Processed frame shape: {normalized.shape}")
                        print(f"Frame min/max values: {normalized.min():.4f}/{normalized.max():.4f}")
                except Exception as e:
                    if self.debug:
                        print(f"Error in preprocessing: {e}")
                    # On error, don't update the frame stack
            
            # Return the current stack regardless of whether we updated it
            stacked_frames = np.array(self.frame_stack)
            
            if self.debug and process_frame:
                print(f"Frame stack shape: {stacked_frames.shape}")
                
            return stacked_frames
        
        # Fallback
        if self.debug:
            print("Fallback case: Unknown observation format")
        return np.array(self.frame_stack)
    
    def act(self, observation):
        """Select action based on observation"""
        try:
            # Check if we're in a raw environment at the start
            if self.start_action_idx < len(self.start_actions) and isinstance(observation, np.ndarray) and len(observation.shape) == 3:
                action = self.start_actions[self.start_action_idx]
                if self.debug:
                    print(f"Using start action #{self.start_action_idx}: {action}")
                self.start_action_idx += 1
                return action
                
            # Process the observation
            state = self.preprocess_observation(observation)
            
            # Reset noise for the policy network to encourage exploration
            self.policy_net.reset_noise()
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Epsilon-greedy action selection with right-biased random actions
            if random.random() < self.epsilon:
                # Random action with bias toward right movement (actions 1-4)
                if random.random() < 0.7:  # 70% chance of right-moving action
                    action = random.choice([1, 2, 3, 4])  # Right-moving actions
                    if self.debug:
                        print(f"Random RIGHT-BIASED action: {action}")
                else:
                    action = random.randint(0, self.n_actions - 1)
                    if self.debug:
                        print(f"Random action: {action}")
                return action
            else:
                # Model-based action
                with torch.no_grad():
                    q_values = self.policy_net(state_tensor)
                    action = q_values.max(1)[1].item()
                    
                    if self.debug:
                        print(f"Model action: {action}, Q-values min/max: {q_values.min().item():.2f}/{q_values.max().item():.2f}")
                    
                    # Override with right movement if model seems stuck
                    # (e.g., repeatedly chooses non-movement actions)
                    if action not in [1, 2, 3, 4] and random.random() < 0.2:
                        action = random.choice([1, 2, 3, 4])
                        if self.debug:
                            print(f"Overriding with RIGHT action: {action}")
                    
                    return action
                    
        except Exception as e:
            if self.debug:
                print(f"ERROR in act method: {e}")
                import traceback
                traceback.print_exc()
            # In case of any error, just go right
            return 1  # right action
    
    def load_model(self, model_path):
        """Load trained model if it exists"""
        try:
            if os.path.exists(model_path):
                if self.debug:
                    print(f"Loading model from {model_path}")
                    
                # Load model
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Support both formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                    if self.debug:
                        print("Loaded model from checkpoint dictionary")
                else:
                    self.policy_net.load_state_dict(checkpoint)
                    if self.debug:
                        print("Loaded model directly")
                
                # Keep in training mode to enable noise
                self.policy_net.train()
                if self.debug:
                    print("Model loaded successfully")
                return True
            else:
                if self.debug:
                    print(f"Model file {model_path} not found")
                return False
        except Exception as e:
            if self.debug:
                print(f"Error loading model: {e}")
            return False