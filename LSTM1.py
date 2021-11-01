”“”
该版本就是在源代码的基础上手写了LSTM
改进的部分为TextLSTM的init和forwa函数
但是在该过程中LSTM的传播过程是串行的
每一个batch的每一个句子都是串行训练，后续会进行改进
“”“
import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test
import _pickle as cpickle

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
        self.W_if = nn.Linear(in_features=emb_size,out_features=n_hidden,bias=False)
        self.b_if = nn.Parameter(torch.ones([n_hidden]))
        self.W_hf = nn.Linear(in_features=n_hidden,out_features=n_hidden,bias=False)
        self.b_hf = nn.Parameter(torch.ones([n_hidden]))
        self.ft = nn.Sigmoid()

        self.W_ii = nn.Linear(in_features=emb_size, out_features=n_hidden, bias=False)
        self.b_ii = nn.Parameter(torch.ones([n_hidden]))
        self.W_hi = nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=False)
        self.b_hi = nn.Parameter(torch.ones([n_hidden]))
        self.it = nn.Sigmoid()

        self.W_io = nn.Linear(in_features=emb_size, out_features=n_hidden, bias=False)
        self.b_io = nn.Parameter(torch.ones([n_hidden]))
        self.W_ho = nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=False)
        self.b_ho = nn.Parameter(torch.ones([n_hidden]))
        self.ot = nn.Sigmoid()

        self.W_ig = nn.Linear(in_features=emb_size, out_features=n_hidden, bias=False)
        self.b_ig = nn.Parameter(torch.ones([n_hidden]))
        self.W_hg = nn.Linear(in_features=n_hidden, out_features=n_hidden, bias=False)
        self.b_hg = nn.Parameter(torch.ones([n_hidden]))

        self.gt = nn.Tanh()
        self.ht = nn.Tanh()

        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

        # self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        # self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        # self.W = nn.Linear(n_hidden, n_class, bias=False)
        # self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        # print("X.shape",X.shape)
        hidden_state = torch.zeros(n_hidden)
        cell_state = torch.zeros(n_hidden)
        h = []
        for k in X:
            hidden_state = torch.zeros(n_hidden)
            cell_state = torch.zeros(n_hidden)
            for x in k:
                i = self.it(self.W_ii(x)+self.b_ii + self.W_hi(hidden_state)+self.b_hi)
                f = self.ft(self.W_if(x)+self.b_if + self.W_hf(hidden_state)+self.b_hf)
                o = self.ot(self.W_io(x)+self.b_io + self.W_ho(hidden_state)+self.b_ho)
                g = self.gt(self.W_ig(x)+self.b_ig + self.W_hg(hidden_state)+self.b_hg)
                cell_state = f*cell_state + i*g
                ht = self.ht(cell_state)
                hidden_state = o*ht
            h.append(hidden_state)
        # print("hidden_state.shape",hidden_state.shape)
        h = torch.tensor([item.detach().numpy() for item in h])
        # print("h.shape", h.shape)
        model = self.W(h) + self.b  # model : [batch_size, n_class]
        return model



def train_LSTMlm():
    model = TextLSTM()
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

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
    n_step = 5  # number of cells(= number of Step)  一次输入多少个词
    n_hidden = 128  # number of hidden units in one cell   隐藏层
    batch_size = 128  # batch size
    learn_rate = 0.0005
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

    print("\nTrain the LSTMLM……………………")
    train_LSTMlm()

    print("\nTest the LSTMLM……………………")
    select_model_path = "models/LSTMlm_model_epoch5.ckpt"
    test_LSTMlm(select_model_path)
