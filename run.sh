# All commands needs to be run on a separate terminal. 
# Please copy each command individually and paste it in each of the terminals
# You can change the number of workers and accordingly you need that many tasks defined

# Alternatively, a terminal manager like tmux can also be used. We prefered working on
# multiple terminals.
python worker.py --job-name ps --env-id Breakout-v0 --num-workers 4
python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 0
python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 1
python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 2
python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 3
tensorboard --logdir=~/Desktop/Breakout-v0/
