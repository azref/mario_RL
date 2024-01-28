#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required packages
get_ipython().system('pip install gym==0.21.0')
get_ipython().system('pip install gym-super-mario-bros==7.3.0')
get_ipython().system('pip install nes-py==8.2.1')
get_ipython().system('pip install pyglet==1.5.21')
get_ipython().system('pip install stable-baselines3==1.5.0')
get_ipython().system('pip install torch==1.11.0')
get_ipython().system('pip install opencv-python')
get_ipython().system('pip install pygame')
get_ipython().system('pip install tensorboard')


# In[2]:


# Import required libraries
import gym
import gym_super_mario_bros
from gym import spaces
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import NatureCNN
import torch.nn as nn
import numpy as np
import cv2
import pygame
from pygame.surfarray import make_surface


# In[3]:


# Custom CNN for grayscale input and frame stacking
class CustomCNN(NatureCNN):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # Adjust the first convolution layer for 4-channel (stacked grayscale frames) input
        # The kernel size, stride, and padding may need to be adjusted as well
        self.cnn[0] = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)


# In[4]:


# Initialize the Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
eval_env = JoypadSpace(env, SIMPLE_MOVEMENT)


# In[5]:


# After initializing eval_env
obs_shape = (4, 84, 84)  # 4 channels (stacked frames), 84x84 each
eval_env.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)


# In[6]:


# Initialize the model with the custom CNN
model = PPO("CnnPolicy", eval_env, policy_kwargs={"features_extractor_class": CustomCNN}, verbose=1)


# In[7]:


# Function to preprocess frames to grayscale and resize
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # Resize to 84x84
    return frame


# In[8]:


# Function to stack frames
def stack_frames(stacked_frames, frame, is_new_episode):
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = np.zeros((84, 84, 4))
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames[:, :, 0] = frame
        stacked_frames[:, :, 1] = frame
        stacked_frames[:, :, 2] = frame
        stacked_frames[:, :, 3] = frame
    else:
        # Shift the oldest frame out and new frame in
        stacked_frames[:, :, :-1] = stacked_frames[:, :, 1:]
        stacked_frames[:, :, 3] = frame

    return stacked_frames


# In[9]:


# Initialize Pygame for rendering
pygame.init()
screen_width, screen_height = 256 * 3, 240 * 3  # Adjust as needed
screen = pygame.display.set_mode((screen_width, screen_height))


# In[ ]:


# Main loop
render = True
while render:
    obs = eval_env.reset()
    stacked_frames = np.zeros((84, 84, 4))  # Reset the stacked frames at the start of each episode
    obs = preprocess_frame(obs)
    stacked_frames = stack_frames(stacked_frames, obs, True)
    
    while True:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                render = False
                break  # Exit the main loop if the Pygame window is closed

        # Convert observation to the expected format
        obs = stacked_frames.reshape(1, 84, 84, 4).transpose(0, 3, 1, 2)

        # Predict action and step in the environment
        action, _ = model.predict(obs)
        if isinstance(eval_env.action_space, gym.spaces.Discrete):
            action = action[0]
        obs, _, done, _ = eval_env.step(action)
        obs = preprocess_frame(obs)
        stacked_frames = stack_frames(stacked_frames, obs, False)

        # Get the frame from the environment and process it
        frame = eval_env.render(mode='rgb_array')
        frame = cv2.resize(frame, (screen_width, screen_height))
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # If the frame is mirrored, flip it horizontally
        frame = cv2.flip(frame, 1)

        # Render the frame using Pygame
        frame_surface = make_surface(frame)
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        # Check if the episode is done
        if done:
            break  # Exit the inner loop and start a new episode

# [Optional] Clean up Pygame and close the window
#pygame.quit()


# In[ ]:




