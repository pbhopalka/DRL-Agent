##Run this file in one terminal (only for linux) 
##It opens multiple terminals automatically
##@author Mahesh Uligade
##@TODO need to fix out of memory bug  and need to improve script
gnome-terminal -e "python worker.py --job-name ps --env-id Breakout-v0 --num-workers 4"
sleep 10 
gnome-terminal -e "python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 0"
sleep 4
gnome-terminal -e "python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 1"
sleep 4
gnome-terminal -e "python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 2"
sleep 4
gnome-terminal -e "python worker.py --job-name worker --env-id Breakout-v0 --num-workers 4 --task 3"
sleep 4
gnome-terminal -e "tensorboard --logdir=/tmp/breakout-rmsprop-LSTM/"
