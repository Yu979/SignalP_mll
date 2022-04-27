import torch
def trans_data(str1, padding_length):
    # 对氨基酸进行编码转换
    a = []
    dic = {'A':1,'B':22,'U':23,'J':24,'Z':25,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':21}
    for i in range(len(str1)):
        a.append(dic.get(str1[i]))
    while(len(a)<padding_length):
        a.append(0)

    return a

if __name__ == '__main__':
    sequence = input("input the amino sequence")
    sequence=trans_data(sequence, 70)
    # 读取模型
    model = torch.load("./checkpoint_ldam/best_ckpt.pth")

    X = torch.tensor(sequence, dtype=torch.long).cuda().reshape(1, -1)
    output = model(X)
    output = int(torch.argmax(output).detach().cpu())
    print('SP' if output==1 else 'NO_SP')