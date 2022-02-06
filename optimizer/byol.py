import tensorflow as tf

def byol_loss(main_z, main_qz, target_z, target_qz):

    loss1 = main_z * target_qz
    loss2 = target_z * main_qz

    loss1 = tf.math.reduce_sum(loss1, axis=2)
    loss2 = tf.math.reduce_sum(loss2, axis=2)

    loss1 = 2 - 2 * loss1
    loss2 = 2 - 2 * loss2

    loss = loss1 + loss2
    loss = loss / 2
    return loss
