import tensorflow as tf
import numpy as np

import collections

def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def main_to_target(src, dst):
    src = get_vars(src)
    dst = get_vars(dst)
    main_target = tf.group([tf.assign(v_targ, v_main)
            for v_main, v_targ in zip(src, dst)])
    return main_target

def check_properties(data):
    ## check available_action < model_output
    for a in data['available_action']:
        assert data['model_output'] >= a

    ## check available_action size == actor size
    assert data['num_actors'] == len(data['available_action'])
    ## check actor size == env size
    assert data['num_actors'] == len(data['env'])
    ## check available_action size == env size
    assert len(data['available_action']) == len(data['env'])
    assert data['reward_clipping'] in ['abs_one', 'soft_asymmetric']

class BatchSampleReplayBuffer:

    def __init__(self, batch_size, collection_size):

        self.batch_size = batch_size
        self.state = collections.deque(maxlen=collection_size)
        self.action = collections.deque(maxlen=collection_size)
        self.reward = collections.deque(maxlen=collection_size)
        self.done = collections.deque(maxlen=collection_size)
        self.behavior_policy = collections.deque(maxlen=collection_size)

    def append(self, state, reward, action, done, behavior_policy):

        self.state.append(state)
        self.reward.append(reward)
        self.action.append(action)
        self.done.append(done)
        self.behavior_policy.append(behavior_policy)

    def sample(self):

        idx = [i for i in range(len(self.state))]
        np.random.shuffle(idx)
        select_idx = idx[:self.batch_size]

        state = [self.state[i] for i in idx]
        action = [self.action[i] for i in idx]
        reward = [self.reward[i] for i in idx]
        done = [self.done[i] for i in idx]
        behavior_policy = [self.behavior_policy[i] for i in idx]

        return dict(
                state=np.stack(state),
                action=np.stack(action),
                reward=np.stack(reward),
                done=np.stack(done),
                behavior_policy=np.stack(behavior_policy))

    def __len__(self):
        return len(self.state)

class UnrolledTrajectory:

    def __init__(self):
        self.trajectory_data = collections.namedtuple(
                'trajectory_data',
                ['state', 'next_state', 'reward', 'done',
                 'action', 'behavior_policy'])

    def initialize(self):
        self.unroll_data = self.trajectory_data(
            [], [], [], [], [], [])

    def append(self, state, next_state, reward, done,
               action, behavior_policy):

        self.unroll_data.state.append(state)
        self.unroll_data.next_state.append(next_state)
        self.unroll_data.reward.append(reward)
        self.unroll_data.done.append(done)
        self.unroll_data.action.append(action)
        self.unroll_data.behavior_policy.append(behavior_policy)

    def extract(self):
        data = {
            'state': self.unroll_data.state,
            'next_state': self.unroll_data.next_state,
            'reward': self.unroll_data.reward,
            'done': self.unroll_data.done,
            'action': self.unroll_data.action,
            'behavior_policy': self.unroll_data.behavior_policy
        }

        return data
