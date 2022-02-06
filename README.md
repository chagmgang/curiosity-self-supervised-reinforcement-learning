# Unofficial curiosity-self-supervised-reinforcement-learning

## Information

* 4 actors with 1 learner.
* Tensorflow implementation with `distributed tensorflow` of server-client architecture.
* [Intrinsically Motivated Self-supervised Learning in Reinforcement Learning](https://arxiv.org/abs/2106.13970)

## Dependency
```
opencv-python
gym[atari]
tensorboardX
tensorflow==1.14.0
```

## How to Run

```
CUDA_VISIBLE_DEVICES=-1 python train.py --job_name --job_name learner --task 0

CUDA_VISIBLE_DEVICES=-1 python train.py --job_name --job_name actor --task 0
CUDA_VISIBLE_DEVICES=-1 python train.py --job_name --job_name actor --task 1
CUDA_VISIBLE_DEVICES=-1 python train.py --job_name --job_name actor --task 2
CUDA_VISIBLE_DEVICES=-1 python train.py --job_name --job_name actor --task 3
```

# Reference

1. [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
2. [distributed_reinforcement_learning](https://github.com/chagmgang/distributed_reinforcement_learning)
3. [deepmind/scalable_agent](https://github.com/deepmind/scalable_agent)
4. [google-research/seed-rl](https://github.com/google-research/seed_rl)
5. [Asynchronous_Advatnage_Actor_Critic](https://github.com/alphastarkor/distributed_tensorflow_a3c)
