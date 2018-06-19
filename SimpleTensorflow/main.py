import tensorflow as tf


def regression():
    # model paramaster
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)

    # model input output
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    y = tf.placeholder(tf.float32)
    loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # train data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)  # reset value to wrong

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
    x = tf.square(10)
    a = sess.run(x)
    print(a)


if __name__ == '__main__':
    regression()
