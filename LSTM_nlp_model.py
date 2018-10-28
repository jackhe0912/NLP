import numpy as np
import tensorflow as tf
from mymodel import reader

DATA_PATH='D:/machine learning/tensorflow/PTB-examples/data'
VOCAB_SIZE=10000   #词典规模

#神经网络参数
HIDDEN_SIZE=200               #隐藏层规模
NUM_LAYERS=2                  #深层循环神经网络中LSTM结构的层数
LEARNING_RATE=1.0             #学习速率
KEEP_PROB=0.5                 #节点不被dropout的概率
MAX_GRAD_NORM=5              #用于控制梯度膨胀的参数

#训练数据的参数
TRAIN_BACTH_SIZE=20               #训练数据batch的大小
TRAIN_NUM_STEP=35                 #训练数据的截断长度

#测试数据的参数
EVAL_BATCH_SIZE=1               #测试数据batch的大小
EVAL_NUM_STEP=1                 #测试数据截断长度
NUM_EPOCH=2                     #使用训练数据的轮数

#通过一个PTBModel类来描述 模型
class PTBModel(object):

    def __init__(self,is_training,batch_size,num_steps):
        #记录使用的batch大小和截断长度
        self.batch_size=batch_size
        self.num_steps=num_steps

        #定义输入层，可以看出输入层的维度为batch_size*num_steps
        self.input_data=tf.placeholder(tf.int32,[batch_size,num_steps])

        #定义预期输出
        self.targets=tf.placeholder(tf.int32,[batch_size,num_steps])

        #定义lstm结构为 循环体的结构且使用dropout的深层循环网络
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)

        if is_training:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=KEEP_PROB)

        stacked_cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS)

        #初始 化最初的状态，也就是全零的向量

        self.initial_state=stacked_cell.zero_state(batch_size,tf.float32)

        #将ID转换成单词向量，。因为因为总共有 VOCAB_SIZE个单词，每个单词向量的维度为HIDDEN_SIZE
        #所以embedding参数的维度为[VOCAB_SIZE,HIDDEN_SIZE]

        embedding=tf.get_variable('embedding',[VOCAB_SIZE,HIDDEN_SIZE])

        #将原本为 batch_size*num_steps个单词ID转换成为词向量，转换后的输入层为 batch_size*num_steps*HIDDEN_SIZE

        inputs=tf.nn.embedding_lookup(embedding,self.input_data)

        #只在训练时使用dropout
        if is_training: inputs=tf.nn.dropout(inputs,KEEP_PROB)

        #定义 输出列表，先将不同时刻LSTM结构的输出收集起来 ，再通过一个全连接层得到最终的输出。

        outputs=[]

        #state存储不同batch中LSTM的状态，将其初始化为0

        state=self.initial_state
        with tf.variable_scope('rnn'):
            for time_step in range(num_steps):
                if time_step>0: tf.get_variable_scope().reuse_variables()

                #从输入数据中获取当前时刻获得的 输入并传入LSTM结构
                stacked_cell_output,state=stacked_cell(inputs[:,time_step,:],state)

                #将当前输入加入输出队列
                outputs.append(stacked_cell_output)

        #将输出队列展开成[batch,hidden_size*num_steps]的形状,然后再reshape成[batch*num_steps,hidden_size]形状
        output=tf.reshape(tf.concat(outputs,1),[-1,HIDDEN_SIZE])

        #把从LSTM中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在每个时刻上都是一个 长度为VOCAB_SIZE
        # 的数组，经过softmax层之后，表示下一个位置是不同单词的概率。

        weight=tf.get_variable('weight',[HIDDEN_SIZE,VOCAB_SIZE])
        bias=tf.get_variable('bias',[VOCAB_SIZE])
        logits=tf.matmul(output,weight)+bias

        #定义交叉熵损失函数。sequence_loss_by_example函数可以用来计算一个序列的交叉熵的和
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],                                              #预测的结果
            [tf.reshape(self.targets,[-1])],                      #期待的正确答案，将二维数组压成一维数组
            [tf.ones([batch_size*num_steps],dtype=tf.float32)])   #损失的权重，设置为1，也就是说不同batch和不同 时刻同等重要

        #计算得到每个batch的平均损失。
        self.cost=tf.reduce_sum(loss)/batch_size
        self.final_state=state

        #只在训练模型的时定义反向传播操作

        if not is_training:return

        trainable_variables=tf.trainable_variables()

        #通过clip_by_global_norm函数控制梯度的大小，避免梯度膨胀的问题

        grads,_=tf.clip_by_global_norm(tf.gradients(self.cost,trainable_variables),MAX_GRAD_NORM)

        #定义优化算法。

        optimizer=tf.train.GradientDescentOptimizer(LEARNING_RATE)

        #定义训练步骤

        self.train_op=optimizer.apply_gradients(zip(grads,trainable_variables))

#使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity

def run_epoch(session,model,data_queue,train_op,output_log,epoch_size):

    #计算perplexity的辅助变量
    total_costs=0.0
    iters=0
    state=session.run(model.initial_state)

    #使用当前数据训练或者测试模型
    for step in range(epoch_size):
        # 生成输入和答案
        feed_dict={}
        x,y=session.run(data_queue)
        feed_dict[model.input_data]=x
        feed_dict[model.targets]=y

        #将状态转为字典
        for i,(c,h) in enumerate(model.initial_state):

            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h

        # 获取损失值和下一个状态,在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的是下一个单词为给定单词的概率
        cost,state,_=session.run([model.cost,model.final_state,train_op],feed_dict=feed_dict)

        total_costs +=cost
        iters += model.num_steps

        #只在训练时输出日志
        if output_log and step % 100 == 0:
            print('After %d steps,perplexity is %.3f' %
                  (step, np.exp(total_costs / iters)))

    return np.exp(total_costs / iters)

def main(_):
    #获取原始数据
    train_data,valid_data,test_data,_=reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len=len(train_data)                       # 训练数据集的大小
    train_batch_len=train_data_len//TRAIN_BACTH_SIZE     # batch的个数
    train_epoch_size=(train_batch_len-1)//TRAIN_NUM_STEP  #epoch的大小

    valid_data_len=len(valid_data)
    valid_batch_len=valid_data_len//EVAL_BATCH_SIZE
    valid_epoch_size=(valid_batch_len-1)//EVAL_NUM_STEP

    test_data_len=len(test_data)
    test_batch_len=test_data_len//EVAL_BATCH_SIZE
    test_epoch_size=(test_batch_len-1)//EVAL_NUM_STEP



    #定义初始化函数
    initializer=tf.random_uniform_initializer(-0.05,0.05)

    #定义训练用的模型
    with tf.variable_scope('language_model',reuse=None,initializer=initializer):
        train_model=PTBModel(True,TRAIN_BACTH_SIZE,TRAIN_NUM_STEP)

    #定义评估用的模型
    with tf.variable_scope('language_model',reuse=True,initializer=initializer):
        eval_model=PTBModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)

    # 生成数据队列，必须放在开启多线程之前
    train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)

    valid_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)

    test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()  # q开启多线程
        threads = tf.train.start_queue_runners(coord=coord)

        #使用训练数据训练模型
        for i in range(NUM_EPOCH):

            print('In iteration: %d' % (i + 1))
            #在所有训练数据上训练循环神经网络模型
            run_epoch(sess,train_model,train_queue,train_model.train_op,True,train_epoch_size)  #训练模型

            valid_perplexity=run_epoch(sess,eval_model,valid_queue,tf.no_op(),False,valid_epoch_size)

            print('Epoch: %d Validation Perplexity: %.3f' % (i + 1,valid_perplexity))
        # 使用测试数据测试模型
        test_perplexity=run_epoch(sess,eval_model,test_queue,tf.no_op(),False,test_epoch_size)

        print('Test Perplexity: %.3f' % test_perplexity)

        ## 关闭多线程
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    tf.app.run()















