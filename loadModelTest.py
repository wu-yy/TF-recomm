#coding:utf-8
#加载训练模型，对结果进行预测
import dataio
import ops
import tensorflow as tf
import svd_train_val
import numpy as np
np.random.seed(13575)

BATCH_SIZE = 1000
USER_NUM = 6040
ITEM_NUM = 3952
DIM = 15
EPOCH_MAX = 100
DEVICE = "/cpu:0"  #"/cpu:0"
def clip(x):
    return np.clip(x, 1.0, 5.0)

if __name__ == '__main__':
    df_train, test = svd_train_val.get_data()

    # 创建saver 对象

    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])

    infer, regularizer = ops.inference_svd(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM,
                                           device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = ops.optimization(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 可以执行或不执行，restore的值会override初始值
        saver.restore(sess, "tmp/svd/model/train.ckpt")


        iter_test = dataio.OneEpochIterator([test["user"],
                                             test["item"],
                                             test["rate"]],
                                            batch_size=-1)
        test_err2 = np.array([])
        for users, items, rates in iter_test:
            pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                    item_batch: items})
            pred_batch = clip(pred_batch)
            test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
            test_err = np.sqrt(np.mean(test_err2))
            print("test error {:f}".format( test_err))
