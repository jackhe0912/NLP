
import tensorflow as tf
from mymodel import reader

DATA_PATH='D:/machine learning/tensorflow/PTB-examples/data'

train_data,valid_data,test_data,_=reader.ptb_raw_data(DATA_PATH)

batch=reader.ptb_producer(train_data,4,5)   # 将数据组织成batch size为4，截断长度为5的数据组

x,y=batch

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()  #q开启多线程
    threads = tf.train.start_queue_runners(coord=coord)

    # 读取前两个batch，其中包括每个时刻的输入和对应的答案，ptb_producer()会自动迭代
    for i in range(2):

        x,y=sess.run(batch)

        print('x:',x)
        print('y:',y)
    # 关闭多线程
    coord.request_stop()
    coord.join(threads)

