# Load necessry libraries including huggingface transformers
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm

# Load the vocabulary and ProtBert-BFD Model
tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False )
model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
# Load the model into the GPU if avilabile
fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0 )
max_len = 70
pro_embedding_feature_dim = 1024

def padding(list):
    str = list[0]
    while(len(str)<max_len):
        str = str + "X"
    return [str]

def trans_data_pro(str_array):
    # padding to standard length
    sequences_Example = [" ".join("".join(padding(sample.split()))) for sample in str_array]
    #  Create or load sequences and map rarely occured amino acids (U,Z,O,B) to (X)
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    embedding = fe(sequences_Example)
    embedding = np.array(embedding)[:, 1:max_len+1, :].mean(axis=1).reshape(-1, pro_embedding_feature_dim)

    return embedding

def trans_data_pro_in_batches(str_array, split=100, path="./pro_embedding/train_feature.npy"):

    if (os.path.exists(path)):
        embedding_result = np.load(path)
        print("Pro feature shape:")
        print(embedding_result.shape)
    else:
        divide_num = int(len(str_array)/split)
        results=[]

        for i in range(1, divide_num+1):
            print("process batch "+str(i)+":")
            results.append(torch.tensor(trans_data_pro(str_array[(i-1)*split:i*split])))

        if (len(str_array) % split != 0):
            print("process batch " + str(i+1) + ":")
            results.append(torch.tensor(trans_data_pro(str_array[divide_num * split:len(str_array)])))

        embedding_result = torch.cat(results).detach().cpu().numpy()
        print("Pro feature shape:")
        print(embedding_result.shape)
        np.save(path, embedding_result)
    return embedding_result

def createDatasetEmbedding(data_path, save_path):
    raw_data=[]
    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = line.strip('\n\t')
            raw_data.append(str)

    features = trans_data_pro_in_batches(raw_data, path=save_path)


def Round(deci=3 ,path="./pro_embedding/train_feature.npy"):

    embedding_result = np.round(np.load(path), decimals=deci)

    print("Pro feature shape:")
    print(embedding_result.shape)
    np.save(path, embedding_result)

createDatasetEmbedding('../data/data_list.txt', "./pro_embedding/train_feature.npy")
createDatasetEmbedding('../test_data/data_list.txt', "./pro_embedding/test_feature.npy")
# Round(3,"./embedding/train_feature.npy" )
# Round(3,"./embedding/test_feature.npy" )
