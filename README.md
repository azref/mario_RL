# mario_RL
Attempt #5 at creating a RL using gym super mario bros with visulization/render. Working - not perfect

Super Mario Bros Reinforcement Learning Project
Introduction

Hello! I'm just a random internet dweller with a strong interest in Machine Learning (ML) and Reinforcement Learning (RL). My journey into this fascinating world began when I watched the Dota AI in action back in 2017. Despite having no coding experience, this sparked a deep interest in me. Fast forward to today, and I've just completed my first project in RL, which I'm excited to share with you all.

This project is my venture into learning and applying ML/RL concepts. The total journey, from watching instructional videos to completing this project, spanned approximately 3 weeks, with around 15-20 hours of dedicated coding, learning, and, admittedly, a bit of mouse-throwing!

As someone new to coding, this project represents a significant personal achievement. It's a step towards my goal of becoming proficient in coding, particularly in the realms of ML and RL. With each project, I hope to show my learning progress and, one day, be able to code independently, relying less on AI assistance.
Project Evolution

This is version #5 of my Super Mario Bros RL project. It underwent massive changes from the original version, which was initially based on following Nicolas Renotte's instructional video. I encountered various challenges, including version issues, frame dimension adjustments, implementing frame stacking and countless other issues. Each obstacle was a learning opportunity, helping me grow my understanding of coding and ML/RL concepts.
Future Improvements

I'm eager to continue developing this project and have several ideas for its improvement:

    Utilizing GPU Functionality: To enhance the model's training efficiency and speed.
    Multi-Screen Rendering: Displaying one large screen alongside three smaller screens to show different perspectives or stages of the game.
    Live Control Inputs Visualization: For the large screen, to provide insights into the model's decision-making process.
    Logging System: Implementing a log file system to track the model's performance and learning progress over time.
    Open to Suggestions: Any additional ideas or suggestions to improve this project are most welcome!

References and Acknowledgments

    YouTube Channel - @NicholasRenotte: His video on RL with Super Mario Bros ([watch here](https://www.youtube.com/watch?v=2eeYqJ0uBKE)) was an excellent resource that I revisited multiple times. Although, I faced version compatibility issues since the video was based on older library versions.
    ChatGPT-4: A remarkable AI that assisted me throughout the project. From coding guidance to troubleshooting, ChatGPT-4 was an invaluable learning tool.
    Stack Overflow: For issues that required deeper digging, Stack Overflow was a go-to resource, especially for understanding version-related issues.
    Gym Super Mario Bros: PyPI
    NES Py: PyPI
    OpenAI Gym: Official Site
    PyTorch: Getting Started
    Stable Baselines3 - PPO: Documentation
    OpenAI Spinning Up: RL Intro

What a time to be alive!
Overview

This project is a Reinforcement Learning (RL) application using the gym_super_mario_bros environment. The goal is to train a neural network to play the Super Mario Bros game, leveraging the capabilities of RL to improve performance over time.
Features

    Reinforcement Learning: Utilizes Stable Baselines3 for RL algorithms.
    Custom Convolutional Neural Network: Tailored for handling the game's visual input.
    Real-Time Game Rendering: Utilizes Pygame for displaying the game's progress.

Installation and Setup
Prerequisites

    Python 3.11
    Pip 23.3.2

Libraries

    gym==0.21.0
    gym-super-mario-bros==7.3.0
    nes-py==8.2.1
    pyglet==1.5.21
    stable-baselines3==1.5.0
    torch==1.11.0
    opencv-python
    pygame
    tensorboard

(Not sure how to ensure the install works for those that download it. Happy to take advice to assist others)
To install the required libraries, run:

bash

pip install gym==0.21.0 gym-super-mario-bros==7.3.0 nes-py==8.2.1 pyglet==1.5.21 stable-baselines3==1.5.0 torch==1.11.0 opencv-python pygame tensorboard

Usage

Run the script Mario_RL5.py to initiate the training process. The script sets up the game environment, processes the game frames, and trains the RL model. The progress of the AI agent can be viewed in real-time through a Pygame-rendered window.
Project Structure

    CustomCNN: A convolutional neural network class, customized for the Mario game frames.
    Environment Setup: Configuration of the Mario game environment.
    Frame Processing: Functions for converting frames to grayscale, resizing, and stacking.
    Pygame Integration: For rendering the game screen with adjustments to frame size and orientation.
    Training Loop: Where the model interacts with the environment, learning from each step.

Contributing

Contributions, bug reports, and suggestions are welcome. Feel free to fork, modify, and make pull requests or create issues for any improvements.
License

This project is open-sourced under the MIT License.
