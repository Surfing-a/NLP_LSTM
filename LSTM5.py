"""
该文件下设计双向LSTM神经网络
具体实现与双层无太大区别，只在forward函数中将输入X进行逆序并重复第一层的训练
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    # 每次读入一个句子
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer
        word = ["<sos>"] + word
        word = word + ["<eos>"]

        # 因为要预测输出 所以长度最小为n_step+1
        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word  # pad加后面也是可以的

        for word_index in range(len(word) - n_step):
            #  每五个预测出来一个单词
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number_dict[word[word_index + n_step]]  # create (n) as target, We usually call this 'casual
            # language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch  # (batch num, batch size, n_step) (batch num, batch size)


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0  # 空白字符
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1  # 未知字符
    number2word_dict[1] = "<unk_word>"
    word2number_dict["<sos>"] = 2  # start
    number2word_dict[2] = "<sos>"
    word2number_dict["<eos>"] = 3  # end
    number2word_dict[3] = "<eos>"

    return word2number_dict, number2word_dict


class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        # 词嵌入仍然需要
        """
        nn.embedding 参数： 输入词表大小 向量表示的维度
        """
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        '''通过C后输入的X为  batch_size，n_step，embed_size'''
        self.W_f1 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_f1 = nn.Parameter(torch.ones([n_hidden]))
        self.W_f2 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_f2 = nn.Parameter(torch.ones([n_hidden]))
        self.ft1 = nn.Sigmoid()
        self.ft2 = nn.Sigmoid()

        self.W_i1 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_i1 = nn.Parameter(torch.ones([n_hidden]))
        self.W_i2 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_i2 = nn.Parameter(torch.ones([n_hidden]))
        self.it1 = nn.Sigmoid()
        self.it2 = nn.Sigmoid()

        self.W_o1 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_o1 = nn.Parameter(torch.ones([n_hidden]))
        self.W_o2 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_o2 = nn.Parameter(torch.ones([n_hidden]))
        self.ot1 = nn.Sigmoid()
        self.ot2 = nn.Sigmoid()

        self.W_g1 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_g1 = nn.Parameter(torch.ones([n_hidden]))
        self.W_g2 = nn.Linear(in_features=emb_size+n_hidden, out_features=n_hidden, bias=False)
        self.b_g2 = nn.Parameter(torch.ones([n_hidden]))

        self.gt1 = nn.Tanh()
        self.ht1 = nn.Tanh()
        self.gt2 = nn.Tanh()
        self.ht2 = nn.Tanh()

        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embedding size]
        # print("X.shape",X.shape)
        # 只有转置以后才能够达到每一个batch同时训练的效果
        h_t1 = torch.zeros(batch_size,n_hidden)
        c_t1 = torch.zeros(batch_size,n_hidden)
        h_t2 = torch.zeros(batch_size, n_hidden)
        c_t2 = torch.zeros(batch_size, n_hidden)
        """
        双层LSTM 上一层的h 输入到当前层
        """
        for x in X:
            """第一层的输入是x和h，代表句子从前到后的关系"""
            # print('x',x.shape)
            # print('h_t1',h_t1.shape)
            i_1 = self.it1(self.W_i1(torch.cat((x,h_t1),1)) + self.b_i1)
            f_1 = self.ft1(self.W_f1(torch.cat((x,h_t1),1)) + self.b_f1)
            o_1 = self.ot1(self.W_o1(torch.cat((x,h_t1),1)) + self.b_o1)
            g_1 = self.gt1(self.W_g1(torch.cat((x,h_t1),1)) + self.b_g1)
            c_t1 = f_1 * c_t1 + i_1 * g_1
            h_t1 = o_1 * self.ht1(c_t1)

        # 通过将tenors->numpy->逆序->tensor
        # print('X1',X,X.shape)
        X = X.detach().numpy()
        X = X[::-1]  # X[start,end,step] 当step为负时相当于逆序遍历
        # print('X2',X,X.shape)
        X = torch.tensor(X.copy())
        # print('X3',X,X.shape)
        # exit()

        for x in X:
            """第二层的输入也是x和h，代表句子从后到前的关系"""
            i_2 = self.it2(self.W_i2(torch.cat((x, h_t2), 1)) + self.b_i2)
            f_2 = self.ft2(self.W_f2(torch.cat((x, h_t2), 1)) + self.b_f2)
            o_2 = self.ot2(self.W_o2(torch.cat((x, h_t2), 1)) + self.b_o2)
            g_2 = self.gt2(self.W_g2(torch.cat((x, h_t2), 1)) + self.b_g2)
            c_t2 = f_2 * c_t2 + i_2 * g_2
            h_t2 = o_2 * self.ht1(c_t2)
        """最后的输出是两个隐层状态的结合  包含了过去和未来的信息"""
        model = self.W(h_t1 + h_t2) + self.b  # model : [batch_size, n_class]
        return model


def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    """"
    模型每次反向传导都会给各个可学习参数p计算出一个偏导数g_t，用于更新对应的参数p。
    通常偏导数g_t不会直接作用到对应的可学习参数p上，而是通过优化器做一下处理，得到一个新的值\widehat{g}_t，
    处理过程用函数F表示（不同的优化器对应的F的内容不同），即\widehat{g}_t=F(g_t)，
    然后和学习率lr一起用于更新可学习参数p，即p=p-\widehat{g}_t*lr。
    """

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)
            # print("output.shape",output.shape)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 100 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1
        print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
              'loss =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(data_root, word2number_dict, n_step)
        all_valid_batch = torch.LongTensor(all_valid_batch).to(device)  # list to tensor
        all_valid_target = torch.LongTensor(all_valid_target).to(device)

        total_valid = len(all_valid_target) * 128  # valid and test batch size is 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'loss =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/LSTMlm_model_epoch{epoch + 1}.ckpt')


def test_LSTMlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model
    model.to(device)

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(data_root, word2number_dict, n_step)
    all_test_batch = torch.LongTensor(all_test_batch).to(device)  # list to tensor
    all_test_target = torch.LongTensor(all_test_target).to(device)
    total_test = len(all_test_target) * 128  # valid and test batch size is 128
    model.eval()    # 这段代码是告诉模型是预测不是训练
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('loss =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 10  # number of cells(= number of Step)  一次输入多少个词
    n_hidden = 128  # number of hidden units in one cell   隐藏层
    batch_size = 128  # batch size
    learn_rate = 0.001
    all_epoch = 5  # the all epoch for training
    emb_size = 256  # embedding size       词嵌入矩阵大小
    save_checkpoint_epoch = 5  # save a checkpoint per save_checkpoint_epoch epochs !!! Note the save path !!!
    data_root = 'penn_small'
    train_path = os.path.join(data_root, 'train.txt')  # the path of train dataset

    print("print parameter ......")
    print("n_step:", n_step)
    print("n_hidden:", n_hidden)
    print("batch_size:", batch_size)
    print("learn_rate:", learn_rate)
    print("all_epoch:", all_epoch)
    print("emb_size:", emb_size)
    print("save_checkpoint_epoch:", save_checkpoint_epoch)
    print("train_data:", data_root)

    word2number_dict, number2word_dict = make_dict(train_path)
    # print(word2number_dict)

    print("The size of the dictionary is:", len(word2number_dict))
    n_class = len(word2number_dict)  # n_class (= dict size)

    print("generating train_batch ......")
    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    train_batch_list = [all_input_batch, all_target_batch]

    print("The number of the train batch is:", len(all_input_batch))
    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)
    # print(all_input_batch.shape)
    # print(all_target_batch.shape)
    all_input_batch = all_input_batch.reshape(-1, batch_size, n_step)
    all_target_batch = all_target_batch.reshape(-1, batch_size)

    import time
    start = time.time()
    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)
    end = time.time()
    print('time=', end - start)
