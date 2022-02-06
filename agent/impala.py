import tensorflow as tf
import copy
import utils
import numpy as np
from optimizer import vtrace
from optimizer import byol
from model.actor_critic import build_network
from augmentation import transform

class Agent(object):

    def __init__(self, trajectory, input_shape, num_actions,
                 discount_factor, start_learning_rate, end_learning_rate,
                 learning_frame, baseline_loss_coef, entropy_coef,
                 self_sup_coef, intrinsic_reward_coef,
                 gradient_clip_norm, reward_clipping,
                 model_name, learner_name):

        self.compose = transform.Compose(
                [transform.Resize(),
                 transform.RandomCrop(crop_size=(input_shape[0], input_shape[1])),
                 transform.Pad(pad_size=(input_shape[0], input_shape[1]))
                 ])
        self.input_shape = input_shape
        self.trajectory = trajectory
        self.num_actions = num_actions
        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.learning_frame = learning_frame
        self.baseline_loss_coef = baseline_loss_coef
        self.entropy_coef = entropy_coef
        self.self_sup_coef = self_sup_coef
        self.intrinsic_reward_coef = intrinsic_reward_coef
        self.gradient_clip_norm = gradient_clip_norm
        self.reward_clipping = reward_clipping

        with tf.variable_scope(model_name):
            with tf.device('cpu'):

                self.s_ph = tf.placeholder(tf.float32, shape=[None, *input_shape])
                
                self.t_s_ph = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
                self.aug_t_s_ph = tf.placeholder(tf.float32, shape=[None, trajectory, *input_shape])
                self.a_ph = tf.placeholder(tf.int32, shape=[None, trajectory])
                self.r_ph = tf.placeholder(tf.float32, shape=[None, trajectory])
                self.d_ph = tf.placeholder(tf.bool, shape=[None, trajectory])
                self.b_ph = tf.placeholder(tf.float32, shape=[None, trajectory, num_actions])

                if reward_clipping == 'abs_one':
                    self.clipped_r_ph = tf.clip_by_value(self.r_ph, -1.0, 1.0)
                elif reward_clipping == 'soft_asymmetric':
                    squeezed = tf.tanh(self.r_ph / 5.0)
                    self.clipped_r_ph = tf.where(self.r_ph < 0, .3 * squeezed, squeezed) * 5.

                self.discounts = tf.to_float(~self.d_ph) * discount_factor
                with tf.variable_scope('main_model', reuse=tf.AUTO_REUSE):
                    self.policy, \
                    self.main_z1s, self.main_qz1s, self.p1s, self.v1s, \
                    self.main_z2s, self.main_qz2s, self.p2s, self.v2s, \
                    self.main_z3s, self.main_qz3s, self.p3s, self.v3s = build_network(
                            state=self.s_ph, traj_state=self.t_s_ph,
                            num_actions=num_actions, num_traj=trajectory)

                with tf.variable_scope('main_model', reuse=tf.AUTO_REUSE):
                    _, \
                    self.target_z1s, self.target_qz1s, _, _, \
                    self.target_z2s, self.target_qz2s, _, _, \
                    self.target_z3s, self.target_qz3s, _, _ = build_network(
                            state=self.s_ph, traj_state=self.aug_t_s_ph,
                            num_actions=num_actions, num_traj=trajectory)

                self.simclr_loss2 = byol.byol_loss(
                        main_z=self.main_z2s,
                        main_qz=self.main_qz2s,
                        target_z=self.target_z2s,
                        target_qz=self.target_qz2s)
                self.simclr_loss3 = byol.byol_loss(
                        main_z=self.main_z3s,
                        main_qz=self.main_qz3s,
                        target_z=self.target_z3s,
                        target_qz=self.target_qz3s)

                mean_simclr_loss2 = tf.math.reduce_mean(self.simclr_loss2, axis=1, keepdims=True)
                mean_simclr_loss3 = tf.math.reduce_mean(self.simclr_loss3, axis=1, keepdims=True)
                std_simclr_loss2 = tf.math.reduce_std(self.simclr_loss2, axis=1, keepdims=True)
                std_simclr_loss3 = tf.math.reduce_std(self.simclr_loss3, axis=1, keepdims=True)

                norm_simclr_loss2 = (self.simclr_loss2 - mean_simclr_loss2) / std_simclr_loss2
                norm_simclr_loss3 = (self.simclr_loss3 - mean_simclr_loss2) / std_simclr_loss3

                self.a1s, self.a2s, self.a3s = vtrace.split_data(self.a_ph)
                self.r1s, self.r2s, self.r3s = vtrace.split_data(self.clipped_r_ph)
                self.d1s, self.d2s, self.d3s = vtrace.split_data(self.discounts)
                self.b1s, self.b2s, self.b3s = vtrace.split_data(self.b_ph)

                self.added_r1s = self.r1s + intrinsic_reward_coef * norm_simclr_loss2
                self.added_r2s = self.r2s + intrinsic_reward_coef * norm_simclr_loss3

                self.vs, self.clipped_rho = vtrace.from_softmax(
                        behavior_policy_softmax=self.b1s,
                        target_policy_softmax=self.p1s,
                        actions=self.a1s,
                        discounts=self.d1s,
                        rewards=self.added_r1s,
                        values=self.v1s,
                        next_values=self.v2s,
                        action_size=num_actions)

                self.vs_plus_1, _ = vtrace.from_softmax(
                        behavior_policy_softmax=self.b2s,
                        target_policy_softmax=self.p2s,
                        actions=self.a2s,
                        discounts=self.d2s,
                        rewards=self.added_r2s,
                        values=self.v2s,
                        next_values=self.v3s,
                        action_size=num_actions)

                self.pg_advantage = tf.stop_gradient(
                        self.clipped_rho * \
                                (self.added_r1s + self.d1s * self.vs_plus_1 - self.v1s))

                self.pi_loss = vtrace.compute_policy_gradient_loss(
                        softmax=self.p1s,
                        actions=self.a1s,
                        advantages=self.pg_advantage,
                        output_size=num_actions)

                self.baseline_loss = vtrace.compute_baseline_loss(
                        vs=tf.stop_gradient(self.vs),
                        value=self.v1s)

                self.entropy = vtrace.compute_entropy_loss(
                        softmax=self.p1s)

                simclr_tot_loss = (self.simclr_loss2 + self.simclr_loss3) / 2
                self.simclr_tot_loss = tf.math.reduce_mean(simclr_tot_loss)

                self.total_loss = self.pi_loss + \
                        self.baseline_loss * baseline_loss_coef + \
                        self.entropy * entropy_coef + \
                        self.simclr_tot_loss * self_sup_coef

            self.num_env_frames = tf.train.get_or_create_global_step()
            self.learning_rate = tf.train.polynomial_decay(start_learning_rate, self.num_env_frames, learning_frame, end_learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, momentum=0, epsilon=0.1)
            gradients, variable = zip(*self.optimizer.compute_gradients(self.total_loss))
            gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
            self.train_op = self.optimizer.apply_gradients(zip(gradients, variable), global_step=self.num_env_frames)

        self.global_to_session = utils.copy_src_to_dst(learner_name, model_name)
        self.saver = tf.train.Saver()

    def save_weights(self, path, step):
        self.saver.save(self.sess, path, global_step=step)

    def load_weights(self, path):
        self.saver.restore(self.sess, path)

    def parameter_sync(self):
        self.sess.run(self.global_to_session)

    def set_session(self, sess):
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())

    def get_policy_and_action(self, state):
        normalized_state = np.stack(state) / 255
        policy = self.sess.run(
                self.policy,
                feed_dict={
                    self.s_ph: [normalized_state]})

        policy = policy[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action, policy, max(policy)

    def train(self, state, reward, action, done, behavior_policy):
        normalized_state = np.stack(state) / 255
        augmented_normalized_state = self.augment_state(normalized_state)

        feed_dict = {
                self.t_s_ph: normalized_state,
                self.aug_t_s_ph: augmented_normalized_state,
                self.a_ph: action,
                self.d_ph: done,
                self.r_ph: reward,
                self.b_ph: behavior_policy}

        pi_loss, value_loss, entropy, learning_rate, simclr_loss, _ = self.sess.run(
                [self.pi_loss, self.baseline_loss, self.entropy,
                 self.learning_rate, self.simclr_tot_loss, self.train_op],
                feed_dict=feed_dict)

        return pi_loss, value_loss, entropy, learning_rate, simclr_loss

    def augment_state(self, state):
        batch_size, traj, _, _, _ = state.shape
        augmented_state = copy.deepcopy(state)
        for b in range(batch_size):
            for t in range(traj):
                original_state = state[b][t]
                new_state = self.compose(original_state)
                augmented_state[b][t] = new_state

        return np.stack(augmented_state)
