from __future__ import print_function
import sys
sys.path.append("..")
from utils_tools.utils import *
from utils_tools.Losses import *
import os
import esm

# Load ESM-1b model
esm_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
esm_model = (esm_model).cuda()
batch_converter = alphabet.get_batch_converter()
embedding_feature_dim = 1280
mean_embedding_feature_dim =40

def trans_data_msa(str_array):

    # 批量处理
    batch_labels, batch_strs, batch_tokens = batch_converter(str_array)
    batch_tokens = batch_tokens.cuda()

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, (_, seq) in enumerate(str_array):
        temp_tensor = token_representations[i, 1: len(seq) + 1]
        sequence_representations.append(temp_tensor.mean(0).detach().cpu().numpy())

    result = torch.tensor(np.array(sequence_representations))

    return result

def trans_data_msa_in_batches(str_array, split=100, path="./embedding/train_feature.npy"):
    if(os.path.exists(path)):
        embedding_result = np.load(path)
        print("MSA feature shape:")
        print(embedding_result.shape)
    else:
        divide_num = int(len(str_array)/split)
        results=[]

        for i in range(1, divide_num+1):
            print("msa process batch "+str(i)+":")
            results.append(trans_data_msa(str_array[(i-1)*split:i*split]))

        if (len(str_array) % split != 0):
            print("msa process batch " + str(i+1) + ":")
            results.append(trans_data_msa(str_array[divide_num * split:len(str_array)]))

        embedding_result = torch.cat(results).detach().cpu().numpy()
        print("MSA feature shape:")
        print(embedding_result.shape)
        np.save(path, embedding_result)
    return embedding_result

def createDatasetEmbedding(data_path, save_path):
    raw_data=[]
    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = line.strip('\n\t')
            raw_data.append(("protein", str))

    features = trans_data_msa_in_batches(raw_data, path=save_path)


def Round(deci=3 ,path="./embedding/train_feature.npy"):

    embedding_result = np.round(np.load(path), decimals=deci)

    print("MSA feature shape:")
    print(embedding_result.shape)
    np.save(path, embedding_result)

if __name__ == '__main__':

    createDatasetEmbedding('../data/data_list.txt', "./embedding/train_feature.npy")
    createDatasetEmbedding('../test_data/data_list.txt', "./embedding/test_feature.npy")
    # Round(3,"./embedding/train_feature.npy" )
    # Round(3,"./embedding/test_feature.npy" )
