import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import os
from skimage import transform
import cv2

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
        #self.load_model("./checkpoints/best_processed_mario_model.pth")
        
        # For exploration - use a higher epsilon for raw observations
        self.epsilon = 0.01  # Low epsilon to trust the model more
        
        # Starting sequence - improved to better handle raw environments
        # Mix of right, jump, and run actions to build momentum
        # 1=right, 2=right+jump, 4=right+jump+run
        self.start_actions = [1, 2, 4, 1, 2, 4, 2, 4, 2]  # Effective starting sequence
        self.start_action_idx = 0
        
        # Frame skipping settings
        self.frame_skip = 4
        self.step_counter = 0  # Count all env steps for proper action timing
        self.action_counter = 0  # Count actions selected
        self.frame_count = 0   # Count frames for raw observation handling
        
        # Action selection
        self.current_action = 1  # Default to 'right' action
        self.last_action = 1
        
        # Episode tracking
        self.is_first_frame = True
        
        # Track consecutive non-right actions to prevent getting stuck
        self.non_right_count = 0
        self.max_non_right = 8  # Max consecutive non-right actions before overriding
        
        # Frame stacking
        self.initialize_frame_stacks()
    
    def initialize_frame_stacks(self):
        """Initialize frame stacks for both raw and processed observations"""
        # For raw observation processing
        self.frame_stack = deque(maxlen=4)  # Final processed stack
        
        # Fill with black frames initially
        empty_frame = np.zeros((84, 84), dtype=np.float32)
        for _ in range(4):
            self.frame_stack.append(empty_frame.copy())
    
    def reset_frame_stack(self):
        """Reset frame stacks at the beginning of each episode"""
        # Reset counters
        self.step_counter = 0
        self.action_counter = 0
        self.frame_count = 0
        
        # Reset actions
        self.current_action = 1
        self.last_action = 1
        
        # Reset episode state
        self.is_first_frame = True
        self.start_action_idx = 0
        self.non_right_count = 0
        
        # Reinitialize frame stacks
        self.initialize_frame_stacks()
    
    def process_single_frame(self, frame):
        """
        Process a single raw frame to exactly match training preprocessing
        Ensures bit-identical output to the wrapper pipeline
        """
        # 1. Convert RGB to grayscale using OpenCV (same as GrayScaleObservation)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 2. Resize to 84x84 using skimage.transform (same as ResizeObservation wrapper)
        resized = transform.resize(gray, (84, 84))
        resized *= 255
        resized = resized.astype(np.uint8)
        
        # 3. Normalize to [0, 1] range (same as TransformObservation)
        normalized = resized / 255.0
        
        # Return as float32 for PyTorch compatibility
        return normalized.astype(np.float32)
            
    def preprocess_observation(self, observation):
        """Process observations to match training pipeline exactly"""
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
            # Process the current frame
            processed_frame = self.process_single_frame(observation)
            
            # Handle first frame specially (fill whole stack with first frame)
            if self.is_first_frame:
                if self.debug:
                    print("Processing first frame of episode")
                
                # Initialize the entire stack with the first processed frame
                self.frame_stack.clear()
                for _ in range(4):
                    self.frame_stack.append(processed_frame.copy())
                
                self.is_first_frame = False
                self.frame_count = 0
            else:
                # We update the stack with new processed frame every time for raw observations
                # But we only use it for decision making every frame_skip steps
                self.frame_stack.append(processed_frame)
            
            # Increment raw frame counter 
            self.frame_count += 1
            
            # Return stacked frames in the correct format for the neural network (4, 84, 84)
            stacked_frames = np.array(self.frame_stack)
            return stacked_frames
        
        # Fallback for any unexpected observation format
        if self.debug:
            print("Fallback case: Unknown observation format")
        return np.array(self.frame_stack)
    
    def should_select_new_action(self):
        """
        Determine if we should select a new action or reuse the current one.
        For raw observations, we select a new action every frame_skip steps.
        For processed observations, we select a new action every step.
        """
        # For processed observations (already skipped at the env level), always select new action
        if self.is_first_frame:
            return True
            
        # For raw observations, only select new action every frame_skip frames
        if self.frame_count % self.frame_skip == 0:
            return True
            
        return False
    
    def act(self, observation):
        """
        Select action based on observation.
        Handles both raw and processed observations with consistent behavior.
        """
       
        # Process the observation first to get proper state for the neural network
        state = self.preprocess_observation(observation)
        
        # First frame of episode special handling
        if self.is_first_frame:
            self.reset_frame_stack()
            self.is_first_frame = False
            
            # Always start with right (action index 1) after death/reset
            self.current_action = 1  # right action
            self.last_action = 1
            if self.debug:
                print("Starting new episode with right action")
            return 1  # right action
        
        # Decide whether to select a new action or reuse previous one
        if not self.should_select_new_action():
            if self.debug:
                print(f"Reusing action {self.current_action} (frame {self.frame_count % self.frame_skip} of {self.frame_skip})")
            return self.current_action
        
        # For raw observations at the start of an episode, use predetermined actions
        if isinstance(observation, np.ndarray) and len(observation.shape) == 3 and observation.shape[2] == 3:
            if self.start_action_idx < len(self.start_actions):
                action = self.start_actions[self.start_action_idx]
                if self.debug:
                    print(f"Using start action #{self.start_action_idx}: {action}")
                self.start_action_idx += 1
                self.last_action = action
                self.current_action = action
                return action
        
        # Reset noise for the policy network to encourage exploration
        self.policy_net.reset_noise()
        
        # Convert to tensor, ensuring correct shape [batch, channels, height, width]
        if len(state.shape) == 3 and state.shape[0] == 4:  # Already stacked frames in correct format
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:  # Need to ensure correct shape
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if state_tensor.shape != torch.Size([1, 4, 84, 84]):
                # Reshape if necessary to match expected input
                state_tensor = state_tensor.reshape(1, 4, 84, 84)
        
        # Get action from model
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
        
        # Apply epsilon-greedy exploration
        if random.random() < self.epsilon:
            # Random action with bias toward right movement
            if random.random() < 0.8:  # 80% chance of right-moving action
                action = random.choice([1, 2, 4])  # Right-moving actions
                if self.debug:
                    print(f"Random RIGHT-BIASED action: {action}")
            else:
                action = random.randint(0, self.n_actions - 1)
                if self.debug:
                    print(f"Random action: {action}")
        else:
            if self.debug:
                print(f"Model action: {action}, Q-values min/max: {q_values.min().item():.2f}/{q_values.max().item():.2f}")
            
            # Track non-right actions
            if action not in [1, 2, 3, 4]:  # Actions that move right
                self.non_right_count += 1
            else:
                self.non_right_count = 0
            
            # Override with right movement if model seems stuck
            if self.non_right_count >= self.max_non_right:
                action = random.choice([1, 2, 4])  # Force right action
                self.non_right_count = 0
                if self.debug:
                    print(f"Overriding with RIGHT action: {action} after {self.max_non_right} non-right actions")
        
        # Store the selected action
        self.last_action = action
        self.current_action = action
        
        # Count the actions we've selected
        self.action_counter += 1
        
        return action
                    

    
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