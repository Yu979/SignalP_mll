from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from target2 import *
kingdom_dic = {'EUKARYA':0, 'ARCHAEA':1, 'POSITIVE':2, 'NEGATIVE':3}
decode = {0:'NO_SP', 1:'SP', 2:'LIPO', 3:'TAT'}

if __name__ == '__main__':

    input_data = "MEEMPLRESSPQRAERCKKSWLLCIVALLLMLLCSLGTLIYTSLKPTAIESCMVKFELSS" \
                 "SKWHMTSPKPHCVNTTSDGKLKILQSGTYLIYGQVIPVDKKYIKDNAPFVVQIYKKNDVL" \
                 "QTLMNDFQILPIGGVYELHAGDNIYLKFNSKDHIQKTNTYWGIILMPDLPFIS";
    input_species = 'EUKARYA';

    x = np.array(trans_data(input_data[0:70].strip('\n'), 70)).reshape(1, 70)
    kingdom = np.eye(len(kingdom_dic.keys()))[kingdom_dic[input_species.strip('\n\t')]].reshape(1, -1)

    X = torch.tensor(np.concatenate((x,kingdom), axis=1)).cuda()
    model = torch.load("./checkpoint_LDAM/best_ckpt_cnn_51_0.935.pth").cuda()
    y, _ = model(X)
    print(decode[y.argmax(dim=1).item()])