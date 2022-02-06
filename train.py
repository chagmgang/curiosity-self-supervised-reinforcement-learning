import json
import time
import utils
import wrappers

import tensorflow as tf
import numpy as np

from tensorboardX import SummaryWriter
from distributed_queue import buffer_queue
from agent import impala

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS

flags.DEFINE_integer('task', -1, "Task id. Use -1 for local training")
flags.DEFINE_enum('job_name',
                  'learner',
                  ['learner', 'actor'],
                  'Job name. Ignore when task is set to -1')

def main(_):
    data = json.load(open('config.json'))
    local_job_device = f'/job:{FLAGS.job_name}/task:{FLAGS.task}'
    shared_job_device = '/job:learner/task:0'
    is_learner = FLAGS.job_name == 'learner'

    cluster = tf.train.ClusterSpec({
        'actor': ['localhost:{}'.format(data['server_port']+1+i) for i in range(data['num_actors'])],
        'learner': ['{}:{}'.format(data['server_ip'], data['server_port'])]})

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task)

    with tf.device(shared_job_device):
        with tf.device('/cpu'):
            queue = buffer_queue.FIFOQueue(
                    trajectory=data['trajectory'],
                    input_shape=data['model_input'],
                    output_size=data['model_output'],
                    queue_size=data['queue_size'],
                    batch_size=data['batch_size'],
                    num_actors=data['num_actors'])

        learner = impala.Agent(
                trajectory=data['trajectory'],
                input_shape=data['model_input'],
                num_actions=data['model_output'],
                discount_factor=data['discount_factor'],
                start_learning_rate=data['start_learning_rate'],
                end_learning_rate=data['end_learning_rate'],
                learning_frame=data['learning_frame'],
                baseline_loss_coef=data['baseline_loss_coef'],
                entropy_coef=data['entropy_coef'],
                self_sup_coef=data['self_sup_coef'],
                intrinsic_reward_coef=data['intrinsic_reward_coef'],
                gradient_clip_norm=data['gradient_clip_norm'],
                reward_clipping=data['reward_clipping'],
                model_name='learner',
                learner_name='learner')
    with tf.device(local_job_device):

        actor = impala.Agent(
                trajectory=data['trajectory'],
                input_shape=data['model_input'],
                num_actions=data['model_output'],
                discount_factor=data['discount_factor'],
                start_learning_rate=data['start_learning_rate'],
                end_learning_rate=data['end_learning_rate'],
                learning_frame=data['learning_frame'],
                baseline_loss_coef=data['baseline_loss_coef'],
                entropy_coef=data['entropy_coef'],
                self_sup_coef=data['self_sup_coef'],
                intrinsic_reward_coef=data['intrinsic_reward_coef'],
                gradient_clip_norm=data['gradient_clip_norm'],
                reward_clipping=data['reward_clipping'],
                model_name=f'actor_{FLAGS.task}',
                learner_name='learner')

    sess = tf.Session(server.target)
    queue.set_session(sess)
    learner.set_session(sess)

    if not is_learner:
        actor.set_session(sess)

    if is_learner:

        writer = SummaryWriter('runs/learner')
        train_step = 0

        batch_sample_replay_buffer = utils.BatchSampleReplayBuffer(
                batch_size=data['batch_size'],
                collection_size=int(data['replay_buffer_size']))
        while True:

            ### fifo to replay buffer
            size = queue.get_size()
            if size > 3 * data['batch_size']:
                batch = queue.sample_batch()
                for i in range(data['batch_size']):
                    batch_sample_replay_buffer.append(
                            state=batch.state[i],
                            reward=batch.reward[i],
                            action=batch.action[i],
                            done=batch.done[i],
                            behavior_policy=batch.behavior_policy[i])

            ### sample and training
            if len(batch_sample_replay_buffer) > 3 * data['batch_size']:
                train_step += 1
                sample_from_buffer = batch_sample_replay_buffer.sample()
                s = time.time()
                pi_loss, baseline_loss, entropy, learning_rate, simclr_loss = learner.train(
                    state=np.stack(sample_from_buffer['state']),
                    reward=np.stack(sample_from_buffer['reward']),
                    action=np.stack(sample_from_buffer['action']),
                    done=np.stack(sample_from_buffer['done']),
                    behavior_policy=np.stack(sample_from_buffer['behavior_policy'])
                )

                writer.add_scalar('data/replay_buffer_size', len(batch_sample_replay_buffer), train_step)
                writer.add_scalar('data/simclr_loss', simclr_loss, train_step)
                writer.add_scalar('data/pi_loss', pi_loss, train_step)
                writer.add_scalar('data/baseline_loss', baseline_loss, train_step)
                writer.add_scalar('data/entropy', entropy, train_step)
                writer.add_scalar('data/learning_rate', learning_rate, train_step)
                writer.add_scalar('data/time', time.time() - s, train_step)

                if train_step % 300 == 0:
                    learner.save_weights('saved_impala/model', step=train_step)
                print(f'train : {train_step}')

    else:

        trajectory = utils.UnrolledTrajectory()
        env = wrappers.make_uint8_env(data['env'][FLAGS.task])
        state = env.reset()
        episode = 0
        score = 0
        episode_step = 0
        total_max_prob = 0
        lives = 4

        writer = SummaryWriter('runs/{}/actor_{}'.format(data['env'][FLAGS.task], FLAGS.task))

        while True:

            trajectory.initialize()
            actor.parameter_sync()

            for _ in range(data['trajectory']):

                action, behavior_policy, max_prob = actor.get_policy_and_action(state)

                episode_step += 1
                total_max_prob += max_prob

                next_state, reward, done, info = env.step(action)

                score += reward
                
                if lives != info['ale.lives']:
                    r = -1
                    d = True
                else:
                    r = reward
                    d = False

                trajectory.append(
                        state=state,
                        next_state=next_state,
                        done=d,
                        action=action,
                        reward=r,
                        behavior_policy=behavior_policy)

                state = next_state
                lives = info['ale.lives']

                if done:

                    print(episode, score)
                    writer.add_scalar('data/{}/prob'.format(data['env'][FLAGS.task]), total_max_prob / episode_step, episode)
                    writer.add_scalar('data/{}/score'.format(data['env'][FLAGS.task]), score, episode)
                    writer.add_scalar('data/{}/episode_step'.format(data['env'][FLAGS.task]), episode_step, episode)

                    episode += 1
                    score = 0
                    episode_step = 0
                    total_max_prob = 0
                    lives = 4
                    state = env.reset()

            unrolled_data = trajectory.extract()
            queue.append_to_queue(
                    task=FLAGS.task,
                    unrolled_state=unrolled_data['state'],
                    unrolled_next_state=unrolled_data['next_state'],
                    unrolled_reward=unrolled_data['reward'],
                    unrolled_done=unrolled_data['done'],
                    unrolled_behavior_policy=unrolled_data['behavior_policy'],
                    unrolled_action=unrolled_data['action'])
        

if __name__ == '__main__':
    tf.app.run()
