import numpy as np
import tensorflow as tf
#print( 10000 // 1000)

input=[]
input.append([1,2,3]) #id
input.append([4,5,6])
print(np.array(input[0]))
inputs = np.transpose(np.vstack([np.array(input[i]) for i in range(len(input))]))
print(inputs)
print(len(inputs[0]))

ids = np.random.randint(0, 10000, (100,))
print("ids:",ids)

def inteface(embd):

    return embd


def initialize(global_step):
    global_step = tf.train.get_global_step()
    assert global_step is not None
    print ("global_step:>>>>>",global_step)
    return np.array([[1,2],[1,2]])


embd_user=tf.get_variable("e1",shape=[2],initializer=tf.ones_initializer(),dtype=tf.int32)
x=np.array([[1,2],[3,4]])
embd_item=tf.get_variable("e2",shape=[2],initializer=tf.ones_initializer(),dtype=tf.int32)
user_batch = tf.placeholder(tf.int32, shape=[None,2], name="user_batch")
intefacer=inteface(user_batch)
a = tf.constant([1.0,2.0],shape=[1,2])
anser=tf.add(intefacer,1)
global_step = tf.contrib.framework.get_or_create_global_step()
b=tf.add(initialize(global_step),1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #summary_writer = tf.summary.FileWriter(logdir="tmp/svd/log", graph=sess.graph)
    sess.run(embd_user)
    #print('embd_user =%s' % (intefacer))
    print('array:',np.array(x))
    _, pred_batch=sess.run([anser,intefacer],feed_dict={user_batch:x})
    print("pre_batch",pred_batch)
    for i in range(2):
        print("xunhaun>>>>>")
        sess.run(b)

