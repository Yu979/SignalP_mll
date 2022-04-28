import torch
import torch.nn as nn
import torch.nn.functional as F
from LSTM import *
from CNN import *
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from CRF import CRF
embedding_feature_dim_msa = 1280
embedding_feature_dim_pro = 1024

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class TargetModel(nn.Module):

    def __init__(self, config, config1, use_CRF=False):
        super(TargetModel, self).__init__()

        self.num_classes = 20
        self.max_len = config1['max_text_len']

        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['embedding_size'])

        self.ef1 = 256
        self.ef2 = 72
        self.ef3 = 9

        self.ef4 = 128

        self.ef5 = 256
        self.ef6 = 64

        if (use_CRF):
            self.crf = CRF(num_tags=2)
            self.ef3 = 2
            self.use_CRF = True

        else:
            self.use_CRF = False


        self.linear=nn.Sequential(nn.Linear(config['embedding_size'], config1['input_dim']-4), nn.ReLU())
        self.linear2=nn.Linear(self.max_len, self.max_len)
        self.lstm=BLSTM(config1)

        # att weight layer
        self.fcn1= nn.Linear(self.ef1, self.ef2)
        self.att_act= nn.Tanh()
        self.fcn2= nn.Linear(self.ef2, self.ef3)

        self.fcn3= nn.Linear(config1['max_text_len'], 1)
        self.fcn4 = nn.Sequential(nn.Linear(self.ef1*self.ef3+self.ef6+self.ef6, self.ef4), nn.ReLU())

        self.fcn5 = NormedLinear(self.ef4, 4)

        self.fcn_embedding_msa = \
            nn.Sequential(nn.Linear(embedding_feature_dim_msa, self.ef5),  NormedLinear(self.ef5, self.ef6))
        self.fcn_embedding_pro = \
            nn.Sequential(nn.Linear(embedding_feature_dim_pro, self.ef5), NormedLinear(self.ef5, self.ef6))

    def forward(self, input):
        input = input.float()
        aux = input[:, 70:74]
        embedding = input[:, 74:]
        aux = aux.unsqueeze(dim=1)
        aux = torch.repeat_interleave(aux, repeats=self.max_len, dim=1)

        batch = input.shape[0]
        input = input[:, :70]
        input=self.embedding(input.long())
        # input_ = input.reshape(-1, 1)
        # input = torch.eye(self.num_classes)[input_]
        # input = input.reshape(batch, self.max_len, self.num_classes).cuda()

        input=self.linear(input)

        input=torch.cat([input, aux], dim=2)
        input=self.lstm(input)

        # att weight layer
        input2=self.fcn1(input)
        input2=self.att_act(input2)
        input2=self.fcn2(input2)

        # used for predicting cs label
        cs_model = input2

        input = input.unsqueeze(dim=3)
        input = torch.repeat_interleave(input, repeats=self.ef3, dim=3)
        input = F.softmax(input, dim=1)
        input2 = input2.unsqueeze(dim=2)
        input2 = torch.repeat_interleave(input2, repeats=self.ef1, dim=2)

        outputs = input * input2

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.fcn3(outputs)
        outputs = outputs.squeeze(dim=-1)

        outputs = torch.cat([outputs.reshape(outputs.shape[0], -1),
                             self.fcn_embedding_msa(embedding[:, :embedding_feature_dim_msa])
                             ], dim=1)
        outputs = self.fcn4(outputs)

        model = self.fcn5(outputs)

        return model, cs_model

class TargetModel_CRF(nn.Module):

    def __init__(self, config, config1, use_CRF=False):
        super(TargetModel_CRF, self).__init__()

        self.num_classes = 20
        self.max_len = config1['max_text_len']

        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['embedding_size'])

        self.ef1 = 256
        self.ef2 = 72
        self.ef3 = 9

        self.ef4 = 128

        self.ef5 = 256
        self.ef6 = 64

        if (use_CRF):
            self.crf = CRF(num_tags=2)
            self.ef3 = 9
            self.use_CRF = True

        else:
            self.use_CRF = False


        self.linear=nn.Sequential(nn.Linear(config['embedding_size'], config1['input_dim']-4), nn.ReLU())
        self.linear2=nn.Linear(self.max_len, self.max_len)
        self.lstm=BLSTM(config1)

        # att weight layer
        self.fcn1= nn.Linear(self.ef1, self.ef2)
        self.att_act= nn.Tanh()
        self.fcn2= nn.Linear(self.ef2, self.ef3)

        self.fcn3= nn.Linear(config1['max_text_len'], 1)
        self.fcn4 = nn.Sequential(nn.Linear(self.ef1*self.ef3+self.ef6, self.ef4), nn.ReLU())

        self.fcn5 = NormedLinear(self.ef4, 4)

        self.fcn_embedding = nn.Linear(1280, self.ef5)
        self.fcn_embedding2 = nn.Linear(self.ef5, self.ef6)

    def forward(self, input):
        input = input.float()
        aux = input[:, 70:74]
        embedding = input[:, 74:]
        aux = aux.unsqueeze(dim=1)
        aux = torch.repeat_interleave(aux, repeats=self.max_len, dim=1)

        batch = input.shape[0]
        input = input[:, :70]
        input=self.embedding(input.long())
        # input_ = input.reshape(-1, 1)
        # input = torch.eye(self.num_classes)[input_]
        # input = input.reshape(batch, self.max_len, self.num_classes).cuda()

        input=self.linear(input)

        input=torch.cat([input, aux], dim=2)
        input=self.lstm(input)

        # att weight layer
        input2=self.fcn1(input)
        input2=self.att_act(input2)
        input2=self.fcn2(input2)

        # used for predicting cs label
        cs_model = input2

        input = input.unsqueeze(dim=3)
        input = torch.repeat_interleave(input, repeats=self.ef3, dim=3)
        input = F.softmax(input, dim=1)
        input2 = input2.unsqueeze(dim=2)
        input2 = torch.repeat_interleave(input2, repeats=self.ef1, dim=2)

        outputs = input * input2

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.fcn3(outputs)
        outputs = outputs.squeeze(dim=-1)

        outputs = torch.cat([outputs.reshape(outputs.shape[0], -1),
                             self.fcn_embedding2(self.fcn_embedding(embedding))], dim=1)
        outputs = self.fcn4(outputs)

        model = self.fcn5(outputs)

        return model, cs_model

class BertModel(nn.Module):

    def __init__(self, config, config1):
        super(BertModel, self).__init__()

        self.num_classes = 20
        self.max_len = config1['max_text_len']
        self.ef0 = 32
        self.ef1 = 256
        self.ef2 = 72
        self.ef3 = 9
        self.ef4 = 128
        self.ef5 = 256
        self.ef6 = 64

        self.embedding = nn.Embedding(num_embeddings=config['vocab_size'], embedding_dim=config['embedding_size'])
        self.linear = nn.Sequential(nn.Linear(config['embedding_size'], self.ef0), nn.ReLU())

        self.lstm = BLSTM(config1)

        # att weight layer
        self.fcn1 = nn.Linear(self.ef1, self.ef2)
        self.att_act = nn.Tanh()
        self.fcn2 = nn.Linear(self.ef2, self.ef3)

        self.fcn3 = nn.Linear(config1['max_text_len'], 1)
        self.fcn4 = nn.Sequential(nn.Linear(self.ef1 * self.ef3 + self.ef6, self.ef4), nn.ReLU())

        self.fcn5 = NormedLinear(self.ef4, 4)

        self.fcn_embedding_msa = \
            nn.Sequential(nn.Linear(embedding_feature_dim_msa, self.ef5), NormedLinear(self.ef5, self.ef6))
        self.fcn_embedding_pro = \
            nn.Sequential(nn.Linear(embedding_feature_dim_pro, self.ef5), NormedLinear(self.ef5, self.ef6))

    def forward(self, input):
        input = input.float()
        aux = input[:, 70:74]
        embedding = input[:, 74:]
        pro_input = embedding[:, embedding_feature_dim_msa:].reshape(-1, self.max_len, embedding_feature_dim_pro)
        input = self.linear(self.embedding(input[: :70]))

        aux = aux.unsqueeze(dim=1)
        aux = torch.repeat_interleave(aux, repeats=self.max_len, dim=1)

        input = torch.cat([pro_input, aux], dim=2)
        input = self.lstm(input)

        # att weight layer
        input2 = self.fcn1(input)
        input2 = self.att_act(input2)
        input2 = self.fcn2(input2)

        # used for predicting cs label
        cs_model = input2

        input = input.unsqueeze(dim=3)
        input = torch.repeat_interleave(input, repeats=self.ef3, dim=3)
        input = F.softmax(input, dim=1)
        input2 = input2.unsqueeze(dim=2)
        input2 = torch.repeat_interleave(input2, repeats=self.ef1, dim=2)

        outputs = input * input2

        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.fcn3(outputs)
        outputs = outputs.squeeze(dim=-1)

        outputs = torch.cat([outputs.reshape(outputs.shape[0], -1),
                             self.fcn_embedding_msa(embedding[:, :embedding_feature_dim_msa])
                             ], dim=1)
        outputs = self.fcn4(outputs)

        model = self.fcn5(outputs)

        return model, cs_model

if __name__ == '__main__':
    # The code below is used for test!

    # Training
    batch_size = 128
    nb_epoch = 100

    # Embedding
    # 暂时考虑amino acid用26个字母表示+1个为空
    vocab_size = 27
    embedding_size = 128

    # Convolution
    # 第一个CNN Module：filter_length = 3
    filter_length1 = 3
    pool_length = 2
    feature_size = 32

    # LSTM
    lstm_output_size = 64

    x = np.random.randn(64, 70, 64)
    x = torch.tensor(x, dtype=torch.float32).cuda()

    lstm_config = {'dropout_rate': 0.2, 'input_dim': 64,
                   'hidden_dim':64, 'output_dim': 2, 'num_layers': 2, 'max_text_len': 70, 'classifier': True,
                   'use_norm': True, 'use_blstm': True}

    model = BLSTM(lstm_config).cuda()
    print(x.dtype)
    y = model(x)
    sum = torch.sum(y)

    grads = torch.autograd.grad(sum, model.parameters())

    # print(grads)


