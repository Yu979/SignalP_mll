from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
from target2 import *

cls_names=['LIPO', 'NO_SP', 'SP', 'TAT']
metrics=['acc', 'F1_score', 'MCC']
loss_names=['LDAM', 'Focal', 'Normal']
metric_ad = ['MCC', 'Kappa', 'balanced_accuracy']
avg = ['micro', 'macro']
metric_ad_aa = ['recall', 'precision']

def precision_recall_curve(Y_test, y_score, n_classes, loss_type):
    """
    :param Y_test: The true labels of test samples, shape = [n_samples]
    :param y_score:  The results of prediction, shape = [n_samples]
    :param n_classes: The total number of classes
    :return:
    """
    print((np.linspace(0,n_classes-1, n_classes).tolist()))
    Y_test = label_binarize(Y_test, classes=(np.linspace(0,n_classes-1, n_classes).tolist()))
    y_score = np.array(y_score)
    print(y_score.shape)
    # %%
    # The average precision score in multi-label settings
    # ....................................................
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        print()
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                         average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    # %%
    # Plot the micro-averaged Precision-Recall curve
    # ...............................................
    #

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))

    # %%
    # Plot Precision-Recall curve for each class and iso-f1 curves
    # .............................................................
    #
    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.savefig('./Results/PR_curve_%s' % (loss_type))
    plt.show()
    plt.close()

def micro_macro():

    for a in avg:
        for metric in metric_ad:

            data_path={'LDAM':'./log/model_%s_%s_%s.txt'%('LDAM', a, metric), 'Focal':'./log/model_%s_%s_%s.txt'%('Focal', a, metric),
                       'Cross_Entropy_Loss':'./log/model_%s_%s_%s.txt'%('Normal', a, metric)}

            # 初始化
            epoch_list = np.linspace(1, 350, 70)
            index=0

            data_list_LDAM=[]
            data_list_Focal=[]
            data_list_CEL=[]

            # 加载数据
            with open(data_path['LDAM'], 'r') as data_file:
                for line in data_file:
                    data_list_LDAM.append(float(line.strip('\n\t')))

            data_file.close()

            with open(data_path['Focal'], 'r') as data_file:
                for line in data_file:
                    data_list_Focal.append(float(line.strip('\n\t')))

            data_file.close()

            with open(data_path['Cross_Entropy_Loss'], 'r') as data_file:
                for line in data_file:
                    data_list_CEL.append(float(line.strip('\n\t')))

            data_file.close()

            plt.plot(epoch_list[55:], data_list_LDAM[55:], label='LDAM')
            plt.plot(epoch_list[55:], data_list_Focal[55:], label='Focal')
            plt.plot(epoch_list[55:], data_list_CEL[55:], label='Cross_entropy_loss')
            plt.xlabel('epoch')
            plt.ylabel(metric)
            plt.title(a+" "+metric+' over all examples')
            plt.legend()

            plt.savefig('./Results/compare_%s_%s.jpg'%(a, metric))
            plt.show()
            plt.close()


def equal(model, X, labels):
    X = torch.tensor(X, dtype=torch.long).cuda()
    output = model(X)
    # 测量各种指标

    all_preds = np.argmax(output.detach().cpu().numpy(), 1)
    result = np.sum(np.equal(np.array(all_preds), np.array(labels)))
    return result


def createTestDataInTest(data_path='./test_data/data_list.txt', label_path="./test_data/target_list.txt",
                    kingdom_path='./test_data/kingdom_list.txt',
                   maxlen=max_len
                   ):
    # 初始化
    data_list = []
    label_list = []
    kingdom_list=[]

    # 加载数据
    with open(data_path, 'r') as data_file:
        for line in data_file:
            data_list.append(np.array(trans_data(line.strip('\n'), maxlen)))

    with open(label_path, 'r') as label_file:
        for line in label_file:
            label_list.append(trans_label(line.strip('\n')))

    with open(kingdom_path, 'r') as kingdom_file:
        for line in kingdom_file:
            kingdom_list.append(np.eye(len(kingdom_dic.keys()))[kingdom_dic[line.strip('\n\t')]])

    data_file.close()
    label_file.close()
    kingdom_file.close()

    X = np.array(data_list)
    labels = np.array(label_list)
    kingdoms= np.array(kingdom_list)

    X = np.concatenate((X,kingdoms), axis=1)
    labels = labels.reshape(labels.shape[0], 1)

    return X, labels

def  loader_test(X_test, labels_test):
    test_dataset = SPDataset(X_test, labels_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    print("Total dataset evaluation:")

    output = []
    labels_test = []

    for i, (input, target) in enumerate(test_loader):
        input = input.cuda()
        target_test = target[:, 0].reshape(target.shape[0]).cuda()

        o1, _ = model(input)
        output.extend(o1.cpu().detach().numpy())
        labels_test.extend(target_test.cpu().detach().numpy())

    output = torch.tensor(np.array(output))
    labels_test = np.array(labels_test)
    y_pred = pred(output).cpu()

    return y_pred, labels_test

def evaluate(X, label):
    test_dataset = SPDataset(X, label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024)

    print("Total dataset evaluation:")

    output = []
    output_aa = []
    labels_test = []
    labels_test_aa = []

    for i, (input, target) in enumerate(test_loader):
        target_test = target[:, 0].reshape(target.shape[0])
        target_aa = target[:, 1:]

        input = input.cuda()
        target_test = target[:, 0].reshape(target.shape[0]).cuda()
        target_aa = target_aa.cuda()

        o1, o_aa= model(input)

        output.extend(o1.cpu().detach().numpy())
        output_aa.extend(o_aa.cpu().detach().numpy())
        labels_test.extend(target_test.cpu().detach().numpy())
        labels_test_aa.extend(target_aa.cpu().detach().numpy())

    output = torch.tensor(np.array(output))
    output_aa = torch.argmax(torch.tensor(np.array(output_aa)), dim=2).reshape(-1, 1)
    labels_test = np.array(labels_test)
    labels_test_aa = np.array(labels_test_aa).reshape(-1, 1)

    return pred(output).cpu(), output_aa, labels_test, labels_test_aa

def aaTest(output_aa_origin, labels_test_aa_origin, labels_test_origin, testType):
    if(testType == "SP"):
        tag=1
    elif(testType == "LIPO"):
        tag=2
    elif (testType == "TAT"):
        tag=3

    labels_test_origin_torch = torch.Tensor(labels_test_origin)
    output_aa = output_aa_origin.reshape(-1, 70).clone()
    labels_test_aa = labels_test_aa_origin.reshape(-1, 70).copy()
    output_aa = output_aa[torch.where(labels_test_origin_torch==tag)].reshape(-1, 1)
    labels_test_aa = labels_test_aa[np.where(labels_test_origin==tag)].reshape(-1, 1)

    print("aa type:"+testType+":================")

    indexes_ = torch.where(output_aa == 1)
    output_aa[indexes_] = 100

    indexes_1 = torch.where(output_aa == 3)
    indexes_2 = torch.where(output_aa == 0)

    output_aa[indexes_1] = 1
    output_aa[indexes_2] = 1

    indexes_0 = torch.where(output_aa != 1)
    output_aa[indexes_0] = 0

    indexes_ = np.where(labels_test_aa == 1)
    labels_test_aa[indexes_] = 100

    indexes_1 = np.where(labels_test_aa == 3)
    indexes_2 = np.where(labels_test_aa == 0)

    labels_test_aa[indexes_1] = 1
    labels_test_aa[indexes_2] = 1

    indexes_0 = np.where(labels_test_aa != 1)
    labels_test_aa[indexes_0] = 0


    y_pred_aa = (output_aa).cpu()

    output_aa_ = output_aa.detach().cpu().numpy()
    indexes_pos = np.where(labels_test_aa == 1)
    p = np.sum(np.equal(output_aa_[indexes_pos], 1))
    s = indexes_pos[0].shape[0]
    print("cleavage site acc: " + str(p / s) + "=" + str(p) + "/" + str(s))

    indexes_neg = np.where(labels_test_aa == 0)
    p = np.sum(np.equal(output_aa_[indexes_neg], 0))
    s = indexes_neg[0].shape[0]
    print("non cleavage site acc: " + str(p / s) + "=" + str(p) + "/" + str(s))

    print("")
    for m in metric_ad_aa:
        result_ad = metric_advanced(m, y_pred_aa, labels_test_aa)

if __name__ == '__main__':
    # loss="LDAM"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
    # 1. 生成按target分类的指标
    # Usual_curves()

    # 2. 生成按生物种类分类的指标

    create_test_files_cls()
    # # 读取模型
    model = torch.load("./checkpoint_LDAM/best_ckpt_cnn_62_0.9566.pth")

    X_test_cls = {'EUKARYA':[],'ARCHAEA':[],'POSITIVE':[],'NEGATIVE':[] }
    labels_test_cls = {'EUKARYA': [], 'ARCHAEA': [], 'POSITIVE': [], 'NEGATIVE': []}
    m="MCC"
    data_files, target_files, kingdom_files, aa_files, feature_files = create_test_files_cls()
    for key in X_test_cls.keys():
        m = "MCC"
        print(key+" MCC:")
        print()
        X_test_cls[key], labels_test_cls[key] = createTestData(data_files[key], target_files[key], kingdom_files[key],
                                                               aa_files[key], test_path=feature_files[key])

        y_pred, output_aa, labels_test, labels_test_aa = evaluate(X_test_cls[key], labels_test_cls[key])

        print()
        print("Test SP Type:")
        print("SP VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 1, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 1, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        print("LIPO VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 2, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 2, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        print("TAT VS NO_SP")
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 3, "part")
        result_ad = metric_advanced(m, y_pred_, labels_test_)
        y_pred_, labels_test_ = relabel(y_pred.clone(), labels_test, 3, "all")
        result_ad = metric_advanced(m, y_pred_, labels_test_)

        print()

        print("Test aa Type:")
        print("test aa:")

        aaTest(output_aa, labels_test_aa, labels_test, "SP")
        if (key != 'EUKARYA'):
            aaTest(output_aa, labels_test_aa, labels_test, "LIPO")
            aaTest(output_aa, labels_test_aa, labels_test, "TAT")

    # 3. 对整体进行测试

    X_test, labels_test=createTestData()

    y_pred, output_aa, labels_test, labels_test_aa = evaluate(X_test, labels_test)
    for m in metric_ad:
        result_ad = metric_advanced(m, y_pred, labels_test)


