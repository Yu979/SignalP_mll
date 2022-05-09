from __future__ import print_function
import numpy.random as random
import torch.backends.cudnn as cudnn
from utils import *
from Net.ComModel import *
from utils_tools.Losses import *
import argparse
import time
from sklearn import preprocessing


# Load ESM-1b model
embedding_feature_dim = 1280

dic = {'NO_SP': 0, 'SP': 1, 'LIPO': 2, 'TAT': 3 }
metric_basic = ['acc', 'F1_score']
metric_ad = ['MCC']
metric_ad_aa = ['recall', 'precision']
avg = ['micro', 'macro']
kingdom_dic = {'EUKARYA':0, 'ARCHAEA':1, 'POSITIVE':2, 'NEGATIVE':3}
filename_list=['./test_data/benchmark_set.fasta', './test_data/target_list.txt', './test_data/data_list.txt']
max_len=70
np.random.seed(1337)  # for reproducibility

# position specific class encoder
position_specific_classes_enc = preprocessing.LabelEncoder()
position_specific_classes_enc.fit(
    np.array(PositionSpecificLetter.values()).reshape((len(PositionSpecificLetter.values()), 1))
)

def trans_data(str1, padding_length):
    # 对氨基酸进行编码转换
    a = []
    trans_dic = {'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':0}
    for i in range(len(str1)):
        if (str1[i] in trans_dic.keys()):
            a.append(trans_dic.get(str1[i]))
        else:
            print("Unknown letter:" + str(str1[i]))
            a.append(trans_dic.get('X'))
    while(len(a)<padding_length):
        a.append(0)

    return a


def trans_data_msa_in_batches(str_array, split=1000, path="./embedding/train_feature.npy"):
    if (os.path.exists(path)):
        embedding_result = np.load(path)
        print("MSA feature shape:")
        print(embedding_result.shape)
    else:
        embedding_result = None
        assert Exception("Embedding dataset not found!")
    return embedding_result

def trans_label(str1):
    # 对标签进行编码转换
    if((str1) in dic.keys()):
        a = dic.get(str1)
    else:
        print(str1)
        raise Exception('Unknown category!')

    return a

def createTrainTestData(data_path='./data/data_list.txt', label_path="./data/target_list.txt"
              ,kingdom_path='./data/kingdom_list.txt', aa_path="./data/aa_list.txt",
              maxlen=max_len, test_split=0.02, seed=1993
              ):
    # 初始化
    raw_data=[]
    data_list=[]
    label_list=[]
    kingdom_list=[]
    aa_list=[]

    index_list={'NO_SP':[], 'LIPO':[], 'TAT':[], 'SP':[]}
    X_list={'NO_SP':[], 'LIPO':[], 'TAT':[], 'SP':[]}
    labels_list={'NO_SP':[], 'LIPO':[], 'TAT':[], 'SP':[]}
    kingdoms_list={'NO_SP':[], 'LIPO':[], 'TAT':[], 'SP':[]}
    aas_list={'NO_SP':[], 'LIPO':[], 'TAT':[], 'SP':[]}
    features_list={'NO_SP':[], 'LIPO':[], 'TAT':[], 'SP':[]}
    # 加载数据
    with open(data_path, 'r') as data_file:
        for line in data_file:
            data_list.append(trans_data(line.strip('\n'), maxlen))

    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = line.strip('\n\t')
            raw_data.append(("protein", str))

    features = trans_data_msa_in_batches(raw_data)

    with open(label_path, 'r') as label_file:
        for line in label_file:
            label_list.append(trans_label(line.strip('\n')))

    with open(kingdom_path, 'r') as kingdom_file:
        for line in kingdom_file:
            kingdom_list.append(np.eye(len(kingdom_dic.keys()))[ kingdom_dic[line.strip('\n\t')] ])

    with open(aa_path, 'r') as aa_file:
        for line in aa_file:
            aa_list.append(classes_sequence_from_ann_sequence(line.strip("\n\t"), position_specific_classes_enc))

    data_file.close()
    label_file.close()
    kingdom_file.close()
    aa_file.close()

    X=data_list
    labels=label_list
    kingdoms=kingdom_list
    aas=aa_list

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
    np.random.seed(seed)
    np.random.shuffle(kingdoms)
    np.random.seed(seed)
    np.random.shuffle(aas)
    np.random.seed(seed)
    np.random.shuffle(features)

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) <= maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels

    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')

    # 保持训练集和验证集同分布
    for key in dic.keys():
        index_list[key] = np.where(np.array(labels) == dic[key])[0]
        X_list[key]=np.array(X)[index_list[key]].reshape(len(index_list[key]), maxlen)
        labels_list[key]=np.array(labels)[index_list[key]].reshape(len(index_list[key]), 1)
        kingdoms_list[key]=np.array(kingdoms)[index_list[key]].reshape(len(index_list[key]),  -1)
        aas_list[key] = np.array(aas)[index_list[key]].reshape(len(index_list[key]), -1)
        features_list[key] = features[index_list[key]].reshape(len(index_list[key]), -1)

    cls_num_list=[int(len(X_list[key]) * (1 - test_split)) for key in dic.keys()]
    print("训练集各类样本个数：")
    print(cls_num_list)

    print(X_list)
    X_train = np.vstack((X_list[key][:int(len(X_list[key]) * (1 - test_split))]
                        for key in dic.keys()
                         ))
    y_train = np.vstack((np.array(labels_list[key][:int(len(labels_list[key]) * (1 - test_split))])
                         for key in dic.keys()))
    kingdom_train = np.vstack((np.array(kingdoms_list[key][:int(len(kingdoms_list[key]) * (1 - test_split))])
                         for key in dic.keys()))
    aa_train = np.vstack((np.array(aas_list[key][:int(len(aas_list[key]) * (1 - test_split))])
                               for key in dic.keys()))
    feature_train = np.vstack((np.array(features_list[key][:int(len(features_list[key]) * (1 - test_split))])
                          for key in dic.keys()))

    X_val = np.vstack((X_list[key][int(len(X_list[key]) * (1 - test_split)):]
                        for key in dic.keys()
                         ))
    y_val = np.vstack((np.array(labels_list[key][int(len(labels_list[key]) * (1 - test_split)):])
                         for key in dic.keys()))
    kingdom_val = np.vstack((np.array(kingdoms_list[key][int(len(kingdoms_list[key]) * (1 - test_split)):])
                             for key in dic.keys()))
    aa_val = np.vstack((np.array(aas_list[key][int(len(aas_list[key]) * (1 - test_split)):])
                             for key in dic.keys()))
    feature_val = np.vstack((np.array(features_list[key][int(len(features_list[key]) * (1 - test_split)):])
                             for key in dic.keys()))

    print(X_train.shape)
    print(y_train.shape)
    print(kingdom_train.shape)
    print(aa_train.shape)
    print(feature_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(kingdom_val.shape)
    print(aa_val.shape)
    print(feature_val.shape)
    print("=======================")

    # shuffle

    X_train = np.concatenate((X_train, kingdom_train, feature_train), axis=1)
    y_train = np.concatenate((y_train, aa_train), axis=1)

    X_val = np.concatenate((X_val, kingdom_val, feature_val), axis=1)
    y_val = np.concatenate((y_val, aa_val), axis=1)

    return (X_train, y_train), (X_val, y_val), cls_num_list

def create_test_files():
    # 初始化
    target_list = []
    data_list = []

    outf1= open(filename_list[1], 'w')
    outf2 = open(filename_list[2], 'w')

    count=0

    with open(filename_list[0], 'r') as fastaf:
        for line in fastaf:
            count+=1
            if count==1:
                name = line.split('|')[2]
                target_list.append(name)

            elif count==2:
                data_list.append(line.replace('\n', ''))   # 读取整个fasta文件构成字典

            elif count==3:
                count=0

    print(len(target_list))
    print(len(data_list))

    for target in target_list:
        outf1.write(target)
        outf1.write('\n')

    for data in data_list:
        outf2.write(data)
        outf2.write('\n')

    fastaf.close()
    outf1.close()
    outf2.close()

def create_test_files_cls():
    # make preprocessing files for test (n class)
    target_list = {'EUKARYA':[],'ARCHAEA':[],'POSITIVE':[],'NEGATIVE':[] }
    data_list = {'EUKARYA':[],'ARCHAEA':[],'POSITIVE':[],'NEGATIVE':[] }
    kingdom_list = {'EUKARYA': [], 'ARCHAEA': [], 'POSITIVE': [], 'NEGATIVE': []}
    aa_list = {'EUKARYA': [], 'ARCHAEA': [], 'POSITIVE': [], 'NEGATIVE': []}

    filename_target={'EUKARYA':"./test_data/target_list_EUKARYA.txt",
                'ARCHAEA':"./test_data/target_list_ARCHAEA.txt",
                'POSITIVE':"./test_data/target_list_POSITIVE.txt",
                'NEGATIVE':"./test_data/target_list_NEGATIVE.txt"}

    filename_data={'EUKARYA':"./test_data/data_list_EUKARYA.txt",
                'ARCHAEA':"./test_data/data_list_ARCHAEA.txt",
                'POSITIVE':"./test_data/data_list_POSITIVE.txt",
                'NEGATIVE':"./test_data/data_list_NEGATIVE.txt"}

    filename_kingdom={'EUKARYA':"./test_data/kingdom_list_EUKARYA.txt",
                'ARCHAEA':"./test_data/kingdom_list_ARCHAEA.txt",
                'POSITIVE':"./test_data/kingdom_list_POSITIVE.txt",
                'NEGATIVE':"./test_data/kingdom_list_NEGATIVE.txt"}

    filename_aa={'EUKARYA':"./test_data/aa_list_EUKARYA.txt",
                'ARCHAEA':"./test_data/aa_list_ARCHAEA.txt",
                'POSITIVE':"./test_data/aa_list_POSITIVE.txt",
                'NEGATIVE':"./test_data/aa_list_NEGATIVE.txt"}

    outf1={}
    outf2={}
    outf3={}
    outf4={}

    for key in target_list.keys():
        outf1[key]=open(filename_target[key], 'w')
        outf2[key]=open(filename_data[key], 'w')
        outf3[key]=open(filename_kingdom[key], 'w')
        outf4[key] = open(filename_aa[key], 'w')

    count=0

    skip=True

    with open(filename_list[0], 'r') as fastaf:
        for line in fastaf:
            count+=1
            if count==1:
                name = line.split('|')[2]
                key = line.split('|')[1]

                target_list[key].append(name)
                kingdom_list[key].append(key)

            elif count==2:
                data_list[key].append(line.replace('\n', ''))   # 读取整个fasta文件构成字典

            elif count==3:
                aa_list[key].append(line.replace('\n', ''))
                count=0

    for key in target_list.keys():
        print("total number of class "+key+" is:"+str(len(data_list[key])))

    for key in target_list.keys():
        for target in target_list[key]:
            outf1[key].write(target)
            outf1[key].write('\n')

    for key in data_list.keys():
        for data in data_list[key]:
            outf2[key].write(data)
            outf2[key].write('\n')

    for key in kingdom_list.keys():
        for item in kingdom_list[key]:
            outf3[key].write(item)
            outf3[key].write('\n')

    for key in aa_list.keys():
        for item in aa_list[key]:
            outf4[key].write(item)
            outf4[key].write('\n')

    fastaf.close()
    for key in target_list.keys():
        outf1[key].close()
        outf2[key].close()
        outf3[key].close()
        outf4[key].close()

    return filename_data, filename_target, filename_kingdom, filename_aa

def createTestData(data_path='./test_data/data_list.txt', label_path="./test_data/target_list.txt",
                    kingdom_path='./test_data/kingdom_list.txt', aa_path = "./test_data/aa_list.txt",
                   maxlen=max_len, test_path="./embedding/test_feature.npy"
                   ):
    # 初始化
    data_list = []
    label_list = []
    kingdom_list=[]
    aa_list=[]
    raw_data=[]
    # 加载数据
    with open(data_path, 'r') as data_file:
        for line in data_file:
            data_list.append((trans_data(line.strip('\n'), maxlen)))

    with open(data_path, 'r') as data_file:
        for line in data_file:
            str = line.strip('\n\t')
            raw_data.append(("protein", str))

    features = trans_data_msa_in_batches(raw_data, path=test_path)

    with open(label_path, 'r') as label_file:
        for line in label_file:
            label_list.append(trans_label(line.strip('\n')))

    with open(kingdom_path, 'r') as kingdom_file:
        for line in kingdom_file:
            kingdom_list.append(np.eye(len(kingdom_dic.keys()))[kingdom_dic[line.strip('\n\t')]])

    count = 0
    with open(aa_path, 'r') as aa_file:
        for line in aa_file:

            # if(len(line.strip("\n\t"))<70):
            #     remove_indexes.append(count)
            #     aa_list.append(np.zeros((70,)))
            #
            # else:
            aa_list.append(classes_sequence_from_ann_sequence(line.strip("\n\t"), position_specific_classes_enc))
            count+=1

    data_file.close()
    label_file.close()
    kingdom_file.close()
    aa_file.close()

    # for r in reversed(remove_indexes):
    #
    #     data_list.pop(r)
    #     label_list.pop(r)
    #     kingdom_list.pop(r)
    #     aa_list.pop(r)
    X = np.array(data_list)
    labels = np.array(label_list)
    kingdoms= np.array(kingdom_list)
    aas = np.array(aa_list)

    X = np.concatenate((X, kingdoms, features), axis=1)
    labels = labels.reshape(labels.shape[0], 1)
    labels = np.concatenate((labels, aas), axis=1)
    return X, labels

def train(train_loader, model, criterion, CSLoss, optimizer, epoch, args):
    # 训练模式
    model.train()

    end=time.time()

    for i, (input, target) in enumerate(train_loader):
        cs_target = target[:, 1:]
        target = target[:, 0]
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
        else:
            input = input.cuda()
            target = target.cuda()
            cs_target = cs_target.cuda()
        # print('input:')
        # print(input)
        # compute output
        # output, _ = model(input)
        output, _ = model(input)
        # print('output:')
        # print(output)
        loss = criterion(output, target) + CSLoss (_, cs_target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        duration = time.time()-end
        end = time.time()
        losses=loss.item()
        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}]\t'   
                      'Loss {loss:.4f} \t'
                      'Duration:{duration:.3f}'.format(
                epoch, i, len(train_loader),loss=losses, duration=duration))  # TODO
            #print(output)

def validate(train_loader, model, criterion, epoch, args, best_acc):

    # 评估模式，模型不再更新参数
    model.eval()
    all_preds = []
    all_targets = []

    index_pred = {'NO_SP':[],  'SP':[], 'TAT':[], 'LIPO':[]}
    index_true = {'NO_SP':[],  'SP':[], 'TAT':[], 'LIPO':[]}

    with torch.no_grad():
        end = time.time()

        for i, (input, target) in enumerate(val_loader):
            target=target[:, 0]
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            else:
                input = input.cuda()
                target = target.cuda()
            # compute output

            output, _ = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = accuracy(output.cpu(), target.cpu())
            losses = loss.item()

            # measure elapsed time
            duration=time.time()-end
            end = time.time()

            all_preds.extend(pred(output).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss:.4f} \t'
                          'Accuracy:{acc:.3f}\t'
                          'Duration:{duration:.3f}\t'.format(
                    epoch, i, len(train_loader), loss=losses, acc=acc, duration=duration))  # TODO
                #print(output)
        # 获得预测中为0元素的索引
        for key in index_pred.keys():

            index_pred[key] = np.argwhere(np.array(all_preds)==dic[key]).squeeze(1).tolist()
            index_true[key] = np.argwhere(np.array(all_targets)==dic[key]).squeeze(1).tolist()
            set1 = set(index_pred[key])
            set2 = set(index_true[key])
            iset = list(set1.intersection(set2))
            true_neg = len(iset)
            total_neg = len(index_true[key])
            print("     accuracy for "+key+": "+str(float(true_neg/total_neg)))




        total_acc=np.sum(np.equal(np.array(all_preds), np.array(all_targets)))/len(all_preds)

        print("total_acc:"+str(total_acc))

        if total_acc >= best_acc:

            best_acc = total_acc

        print("best_acc by now:"+str(best_acc))

        return best_acc

def relabel(y, label_test, keep, mode):
    y_ = y.tolist()
    label_ = label_test
    if(mode=="part"):
        new_y=[]
        new_label=[]
        for index, i in enumerate(label_):
            if i==0:
                new_label.append(i)
                new_y.append(y_[index])
        for index, i in enumerate(label_):
            if i==keep:
                new_label.append(i)
                new_y.append(y_[index])
        new_y=np.array(new_y)
        new_label=np.array(new_label)
        new_y[np.where(new_y != keep)] = 0
        new_y[np.where(new_y==keep)]=1
        new_label[np.where(new_label != keep)] = 0
        new_label[np.where(new_label == keep)] = 1


    elif (mode == "all"):
        new_y = np.array(y)
        new_label = np.array(label_test)
        new_y[np.where(new_y != keep)] = 0
        new_y[np.where(new_y==keep)]=1
        new_label[np.where(new_label != keep)] = 0
        new_label[np.where(new_label == keep)] = 1

    return new_y, new_label


#
#         index_keep = np.where(new_y == keep)
#         index_other = np.where(new_y != keep)
#         new_y[index_keep]=1
#         new_y[index_other]=0
#
#
#
#
#     return new_y, new_label


def test(model, loader, out_f, out_f_all_acc, out_f_ad, best_mcc, args):

    output=[]
    output_aa=[]
    labels_test=[]
    labels_test_aa=[]

    for i, (input, target) in enumerate(loader):
        target_test = target[:, 0].reshape(target.shape[0])
        target_aa = target[:,1:]

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target_test = target_test.cuda(args.gpu, non_blocking=True)
            target_aa = target_aa.cuda(args.gpu, non_blocking=True)
        else:
            input = input.cuda()
            target_test = target_test.cuda()
            target_aa = target_aa.cuda()

        o1, o_aa= model(input)

        output.extend(o1.cpu().detach().numpy())
        # output_aa.extend(o_aa.cpu().detach().numpy())
        labels_test.extend(target_test.cpu().detach().numpy())
        # labels_test_aa.extend(target_aa.cpu().detach().numpy())

    output = torch.tensor(np.array(output))
    # output_aa = torch.argmax(torch.tensor(np.array(output_aa)), dim=2).reshape(-1, 1)
    labels_test = np.array(labels_test)
    # labels_test_aa = np.array(labels_test_aa).reshape(-1, 1)

    for key in dic.keys():
        for m in metric_basic:
            result = metric(m, output, labels_test, key)
            out_f[key][m].write(str(result) + "\n")
    result = metric("acc", output, labels_test)
    out_f_all_acc.write(str(result) + "\n")

    # # test aa output
    # print("test aa:")
    #
    # indexes_ = torch.where(output_aa == 1)
    # output_aa[indexes_] = 100
    #
    # indexes_1 = torch.where(output_aa == 3)
    # indexes_2 = torch.where(output_aa == 0)
    #
    # output_aa[indexes_1] = 1
    # output_aa[indexes_2] = 1
    #
    # indexes_0 = torch.where(output_aa != 1)
    # output_aa[indexes_0] = 0
    #
    # indexes_ = np.where(labels_test_aa == 1)
    # labels_test_aa[indexes_] = 100
    #
    # indexes_1 = np.where(labels_test_aa == 3)
    # indexes_2 = np.where(labels_test_aa == 0)
    #
    # labels_test_aa[indexes_1] = 1
    # labels_test_aa[indexes_2] = 1
    #
    # indexes_0 = np.where(labels_test_aa != 1)
    # labels_test_aa[indexes_0] = 0
    #
    # y_pred_aa = (output_aa).cpu()
    #
    # for m in metric_ad_aa:
    #     result_ad = metric_advanced(m, y_pred_aa, labels_test_aa)

    print()
    print("test SP:")

    # test SP type output
    for m in metric_ad:
        y_pred = pred(output).cpu()
        result_ad = metric_advanced(m, y_pred, labels_test)
        out_f_ad[m].write(str(result_ad) + "\n")
        if (m == "MCC"):
            if (result_ad > best_mcc):
                best_mcc = result_ad
                save_model(model, epoch+1 , True, args, float('%.4f' % best_mcc))

    print()
    print("BEST MCC:" + str(best_mcc))

    torch.cuda.empty_cache()

    return best_mcc

if __name__ == '__main__':
    print(position_specific_classes_enc.transform(["S", "Z", "T", "C", "K", "O", "I", "L", "E"]))
    # 指定某块GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"
    # 参数定义
    parser = argparse.ArgumentParser(description='PyTorch SignalP Model')
    parser.add_argument('--model_arch', default="CNN-LSTM", type=str, help='the architecture of model')
    parser.add_argument('--loss_type', default="LDAM", type=str, help='loss type')
    parser.add_argument('--train_rule', default="Reweight", type=str, help='train rule')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--attention', default=False, type=bool,
                        help='Use attention or not')

    args = parser.parse_args()
    if args.seed is not None:
        # 设置随机种子，保证每次运行结果一致
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # Embedding
    # 暂时考虑amino acid用26个字母表示+1个为空
    vocab_size = 21
    embedding_size = 20

    # Convolution
    # 第一个CNN Module：filter_length = 10
    filter_length1 = [1, 1, 1, 1, 9, 11, 13, 15]
    pool_length = 2
    feature_size = 30

    # 第一个CNN Module：filter_length = 3
    filter_length2 = 7
    pool_length = 2
    feature_size2 = 128


    # LSTM
    lstm_hidden_dim = 64

    config = {'embedding_size':148, 'vocab_size': 21, 'dropout_rate':0.5,
              'feature_size': feature_size, 'activation_function_type': "Sigmoid", 'kernel_size':10}

    lstm_config = {'dropout_rate': 0.2, 'input_dim': 128+4,
                   'hidden_dim': lstm_hidden_dim, 'output_dim': len(dic.keys()), 'num_layers':1, 'max_text_len': max_len,
                   'classifier': False, 'OnLSTM':True,
                   'use_norm': args.loss_type == 'LDAM', 'use_blstm': True, 'attention':args.attention}

    # 创建模型
    print("Creating the model……")
    if (args.model_arch=="CNN"):
        model = TextCNN(config)
    elif (args.model_arch=="CNN-LSTM"):
        model = TargetModel(config, lstm_config)
    elif (args.model_arch=="ESM"):
        model = ESMModel(lstm_config)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # 创建存放log的文件夹
    make_directory(dic.keys())

    # 加载数据
    print('Loading data...')
    train_set, val_set, cls_num_list = createTrainTestData()
    X_train, Y_train = train_set
    X_val, Y_val = val_set

    X_test, labels_test = createTestData()

    print(X_train[0])
    # 查看
    # print(X_train.shape)
    # print(X_val.shape)

    train_dataset = SPDataset(X_train, Y_train)
    val_dataset = SPDataset(X_val, Y_val)
    test_dataset = SPDataset(X_test, labels_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    labels_test = labels_test[:, 0].reshape(labels_test.shape[0])

    print("Number of test samples:")
    print(len(X_test))
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # 定义损失函数
    cseloss = CSLoss().cuda(args.gpu)

    print("Loss type is:" + args.loss_type)
    print("================================================================")
    out_f = {'SP': {}, 'NO_SP': {},'TAT':{}, 'LIPO':{}}
    out_f_ad = {'F1_score': None, 'MCC': None, "AUC_ROC": None, "Kappa": None}

    for key in dic.keys():
        for m in metric_basic:
            file_name = "./log/%s/model_%s_%s.txt" % (key, args.loss_type, m)
            out_f[key][m] = open(file_name, 'w')

    out_f_all_acc = open("./log/model_%s_%s.txt" % (args.loss_type, "acc"), 'w')

    for key in out_f_ad.keys():
        out_f_ad[key] = open("./log/model_%s_%s.txt" % (args.loss_type, key), 'w')

    best_acc = 0.0
    best_mcc = 0.0

    # 清空内存
    del X_train
    del Y_train
    del X_val
    del Y_val
    del X_test
    del labels_test

    # 开始训练
    for epoch in range(args.epochs):
        # 定义Reweight
        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'Reweight':
            train_sampler = None
            beta = 0.999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

            # effective_num_kingdom = 1.0 - np.power(beta, cls_kingdom_num_list)
            # per_cls_weights_kingdom = (1.0 - beta) / np.array(effective_num_kingdom)
            # per_cls_weights_kingdom = per_cls_weights_kingdom / np.sum(per_cls_weights_kingdom) * len(per_cls_weights_kingdom)
            # per_cls_weights_kingdom = torch.FloatTensor(per_cls_weights_kingdom).cuda(args.gpu)
            # kingdomloss = LDAMLoss(cls_num_list=cls_kingdom_num_list, max_m=0.3, s=150, weight=per_cls_weights_kingdom).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)

        # 定义损失函数
        if args.loss_type == 'LDAM':
            # 给的几个参数：
            # 每个列的sample数量， 每个列的权重，max_m，以及s
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.3, s=150, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'LDAM_CB':
            criterion = LDAMLoss_CB(cls_num_list=cls_num_list, max_m=0.3, s=150).cuda(args.gpu)

        elif args.loss_type == 'Focal':
            criterion = FocalLoss(gamma=2.0).cuda(args.gpu)
        elif args.loss_type == 'Normal':
            criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
        else:
            raise Exception("Wrong loss type!")

        print("================================")
        print("epoch:" + str(epoch + 1))
        # train for one epoch
        train(train_loader, model, criterion, cseloss, optimizer, epoch, args)
        # evaluate on validation set
        best_acc = validate(val_loader, model, criterion, epoch, args, best_acc)

        # test
        model.eval()
        if(epoch+1 > 200):
            if ((epoch + 1) % 1 == 0 ):

                best_mcc = test(model, test_loader, out_f, out_f_all_acc, out_f_ad,
                                                                    best_mcc,
                                                                    args)

        print("================================")

    for key in dic.keys():
        for m in metric_basic:
            out_f[key][m].close()
    out_f_all_acc.close()

    for key in out_f_ad.keys():
        out_f_ad[key].close()

