python worker.py --job-name ps --env-id Pong-v0 --num-workers 6
python worker.py --job-name worker --env-id Pong-v0 --num-workers 6 --task 0
python worker.py --job-name worker --env-id Pong-v0 --num-workers 4 --task 1
python worker.py --job-name worker --env-id Pong-v0 --num-workers 4 --task 2
python worker.py --job-name worker --env-id Pong-v0 --num-workers 4 --task 3
python worker.py --job-name worker --env-id Pong-v0 --num-workers 4 --task 4
python worker.py --job-name worker --env-id Pong-v0 --num-workers 4 --task 5