---
published: true
layout: post
title: 'My experience with OpenAI Spinning Up'
date: 2020-12-20
tags: reinforcement-learning, diary
---


_The official website for OpenAI SpinningUp can be found at [spinningup.openai.com](https://spinningup.openai.com/en/latest/index.html)_ 


## Introduction

Thanks to OpenAI providing OpenAI Spinning Up, a well-structured reinforcement learning educational resource. 

What is OpenAI Spinning Up? It is more a repository than a library. It provides reinforcement learning methods, mostly policy gradient based, with basic formulation, example codes and exercises. It makes use of OpenAI gym as the interactive environment to experiment RL solutions. What's more, it also introduces a good list RL related papers to grow as a RL scientist. 


## Day 1

In day 1, I read through the whole website. A big portion of my effort was spent on understanding what contents are in OpenAI SpinningUp and how they are structured. I have also downloaded the required package to my Ubuntu system.

## Day 2

I attempted exercise 1-1 and 1-2. The exercise used PyTorch, which is a bit new to me. PyTorch is easy to use. Most of the time, I just guess how the syntax should be, and it just worked. 

Exercise 1-2 requires a MuJoCo library, which is not free. I managed to get a licence afterwards. I have also come across a few issues during installation and put my solution at the bottom.

## Day 3

During the days waiting for MuJoCo to be installed, I read up the basic policy gradient and its 6 variants introduced in the website. Except for vanilla policy gradient (VPG), the rest are very technical for me to understand. It really requires tons of maths to understand fully. I plan to digest slowly in the future, with more readings from other sources. 

## Day 4
MuJoGo is installed. Exercise 1-2 passed. Exercise 1-3 requires to partially implement Twin Delayed Deep Determinist Policy Gradient (TD3). As I did not really understand the math yet, this exercise is challenging. 


## Others

Installing MuJoCo and mujoco-py is a bit more troublesome than needed. Sharing my experience on Ubuntu 18.04. 

```shell
# add the following line to .bashrc
LD_LIBRARY_PATH=$HOME/.mujoco/mujoco200/bin pip install mujoco-py
```
```shell
source ~/.bashrc 
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
sudo apt install patchelf
conda activate openai_env
pip install -U 'mujoco-py<2.1,>=2.0'
```

#### Solve Error: GLEW initalization error: Missing GL version
OS: Ubuntu 20.04

Step 1: Run update command to update package repositories and get the latest package information.
`sudo apt-get update -y`

Step 2: flag to quickly install the packages and dependencies.
`sudo apt-get install -y libglew-dev`
Got these from : https://zoomadmin.com/HowToInstall/UbuntuPackage/libglew-dev

Step 3:`export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so`
