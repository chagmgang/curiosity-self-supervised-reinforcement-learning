import tensorflow as tf

def encoder(x):
    x = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
    x = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=tf.nn.relu)

    return tf.layers.flatten(x)

def project_layer(x, dims):

    for d in dims:
        x = tf.layers.dense(x, units=d, activation=tf.nn.relu)

    return tf.layers.dense(x, units=dims[-1], activation=None)

def layer(x, dims, last_dim, last_activation):

    for d in dims:
        x = tf.layers.dense(x, units=d, activation=tf.nn.relu)

    return tf.layers.dense(x, units=last_dim, activation=last_activation)

def network(x, dims=[4, 4, 4], num_actions=2):

    latent_vector = encoder(x)
    z = project_layer(latent_vector, dims)
    qz = project_layer(z, dims)

    z = z / (1e-12 + tf.norm(z, axis=1, keepdims=True))
    qz = qz / (1e-12 + tf.norm(qz, axis=1, keepdims=True))

    actor = layer(latent_vector, dims=dims, last_dim=num_actions, last_activation=tf.nn.softmax)
    critic = tf.squeeze(layer(latent_vector, dims=dims, last_dim=1, last_activation=None), axis=1)

    return z, qz, actor, critic

def split_value(x):

    return x[:, :-2], x[:, 1:-1], x[:, 2:]

def rolling(x, dims, num_actions, num_traj, scope_name, network_func):

    zs, qzs, ps, vs = list(), list(), list(), list()

    for i in range(num_traj):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            z, qz, p, v = network_func(
                    x=x[:, i], dims=dims,
                    num_actions=num_actions)
            zs.append(z)
            qzs.append(qz)
            ps.append(p)
            vs.append(v)

    zs = tf.stack(zs, axis=1)
    qzs = tf.stack(qzs, axis=1)
    ps = tf.stack(ps, axis=1)
    vs = tf.stack(vs, axis=1)

    return zs, qzs, ps, vs

def build_network(state, traj_state, num_actions, num_traj, dims=[256, 256, 256], scope_name='impala', network_func=network):

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        z, qz, p, _ = network_func(
                x=state, dims=dims,
                num_actions=num_actions)

    r1_traj_state, r2_traj_state, r3_traj_state = split_value(traj_state)

    z1s, qz1s, p1s, v1s = rolling(
            x=r1_traj_state, dims=dims,
            num_actions=num_actions, num_traj=num_traj-2,
            scope_name=scope_name,
            network_func=network_func)

    z2s, qz2s, p2s, v2s = rolling(
            x=r2_traj_state, dims=dims,
            num_actions=num_actions, num_traj=num_traj-2,
            scope_name=scope_name,
            network_func=network_func)

    z3s, qz3s, p3s, v3s = rolling(
            x=r3_traj_state, dims=dims,
            num_actions=num_actions, num_traj=num_traj-2,
            scope_name=scope_name,
            network_func=network_func)

    return z, qz, p, z1s, qz1s, p1s, v1s, z2s, qz2s, p2s, v2s, z3s, qz3s, p3s, v3s
