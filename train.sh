python3 train.py --job_name learner --task 0 &
CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 0 &
CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 1 &
CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 2 &
CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 3 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 4 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 5 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 6 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 7 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 8 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 9 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 10 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 11 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 12 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 13 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 14 &
#CUDA_VISIBLE_DEVICES=-1 python3 train.py --job_name actor --task 15 &

