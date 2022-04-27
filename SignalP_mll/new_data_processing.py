# convert fasta files into usable dataset file

# -*- coding: utf-8 -*-

import Bio.SeqIO as SeqIO


filename_list=['./data/train_set.fasta', './data/target_list.txt', './data/data_list.txt',
               './data/kingdom_list.txt', './data/aa_list.txt']

target_list=[]
data_list=[]
kingdom_list=[]
max_length = 70 * 2
def usage():
    print('Usage: python script.py [fasta_file] [namelist_file] [outfile_name]')


def main():

    outf1= open(filename_list[1], 'w')
    outf2 = open(filename_list[2], 'w')
    outf3 = open(filename_list[3], 'w')
    outf4 = open(filename_list[4], 'w')

    count=0

    data_list = []
    aa_list=[]
    target_list = []
    kingdom_list = []
    for record in SeqIO.parse(filename_list[0], "fasta"):
        # TODO: skipping non 70 length sequence for now; ISSUE #1
        if int(len(record)) != max_length:
            continue

        half_len = int(len(record) / 2)

        sequence=''
        for a in record[0:half_len]:
            sequence = sequence+str(a)

        ann_sequence=''
        for a in record[half_len : int(len(record))]:
            ann_sequence = ann_sequence+str(a)


        feature_parts = record.id.split("|")
        (uniprot_id, kingdom, sp_type, partition) = feature_parts

        data_list.append((sequence))
        aa_list.append((ann_sequence))
        kingdom_list.append(kingdom)
        target_list.append(sp_type)

    print("Sum:")
    print(len(data_list))
    print(len(aa_list))
    print(len(kingdom_list))
    print(len(target_list))

    for target in target_list:
        outf1.write(target)
        outf1.write('\n')

    for data in data_list:
        outf2.write(data)
        outf2.write('\n')

    for kingdom in kingdom_list:
        outf3.write(kingdom)
        outf3.write('\n')

    for aa in aa_list:
        outf4.write(aa)
        outf4.write('\n')

    outf1.close()
    outf2.close()
    outf3.close()
    outf4.close()

try:
    main()
except IndexError:
    usage()
