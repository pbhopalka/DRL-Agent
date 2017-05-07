# Playing Atari Games using Deep Reinforcement Learning

This is the final year project for Team Convolution. We have done a survey about Deep Reinforcement Learning and 
are testing its applications on various Atari Games.

We are trying to build a Reinforcement Learning Agent for atari game using Asynchronous Advantage
Actor-Critic (A3C) algorithm which has been described in [this paper](https://arxiv.org/pdf/1602.01783.pdf).

This code is heavily inspired from the works of [OpenAI/universe-starter-agent](https://github.com/openai/universe-starter-agent) and [Deep-RL agent](https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb). You may go through these codes if you
feel like doing so.

We have implemented the A3C algorithm and have tested the various Gradient Descent/Ascent Optimization Techniques like Adadelta, 
RMSProp and Adam. You can read about them [here](http://sebastianruder.com/optimizing-gradient-descent/).

## Running the code

Firstly make sure you run `script_install_before.sh` in your terminal so that all the prerequisite libraries are installed.

To run the code, please check the file `run.sh`. Each command in that file needs to run on a separate terminal. 
A terminal manager like `tmux` can also be used.

To see the progress, run  `tensorboard --logdir=/tmp` in your terminal and then open `http://localhost:6006/` in your browser.

# Team Members

Piyush Bhopalka,
Saksham Agarwal,
Mahesh Uligade

Please shoot an email at `pbhopalka@gmail.com`, if you have any queries :)

(National Institute of Technology, Calicut)
