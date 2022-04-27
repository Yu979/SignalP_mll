# convert fasta files into usable dataset file

# -*- coding: utf-8 -*-

import sys
import time



filename_list=['./data/train_set.fasta', './data/target_list.txt', './data/data_list.txt', './data/kingdom_list.txt']

target_list=[]
data_list=[]
kingdom_list=[]

def usage():
    print('Usage: python script.py [fasta_file] [namelist_file] [outfile_name]')


def main():

    outf1= open(filename_list[1], 'w')
    outf2 = open(filename_list[2], 'w')
    outf3 = open(filename_list[3], 'w')

    count=0

    with open(filename_list[0], 'r') as fastaf:
        for line in fastaf:
            count+=1
            if count==1:
                name = line.split('|')[2]
                kingdom = line.split('|')[1]
                target_list.append(name)
                kingdom_list.append(kingdom)

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

    for kingdom in kingdom_list:
        outf3.write(kingdom)
        outf3.write('\n')

    fastaf.close()
    outf1.close()
    outf2.close()
    outf3.close()

try:
    main()
except IndexError:
    usage()




