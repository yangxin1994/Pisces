#!/usr/bin/env python
# _*_ coding: utf-8 _*_

import csv
import os
import glob
import random
## id,func_name,func_id,block_id_in_func,numeric constants,string constants,No. of transfer instructions,No. of calls,No. of instructinos,No. of arithmetic instructions,No. of logic instructions,No. of offspring,betweenness centrality
#生成训练集中函数block的数量
# block_num_min< block_num <= block_num_max
# if block_num_max = -1, 忽略这一设置,不考虑block数量
block_num_min = 0
block_num_max = -1
PREFIX = ""

#正例负例数量
pos_num = 1
neg_num = 1

train_dataset_num = 100000
test_dataset_num = int(train_dataset_num/10)
vaild_dataset_num = int(train_dataset_num/10)

dir_folder = "./features/"
func_list_file = dir_folder + "function_list"+PREFIX+".csv"
train_file = dir_folder + "train"+PREFIX+".csv"
test_file = dir_folder + "test"+PREFIX+".csv"
vaild_file = dir_folder + "vaild"+PREFIX+".csv"
INDEX = dir_folder+"index.csv"

# function name, program name, block num, version uid list,
func_list_fp = open(func_list_file, "w")

index_uuid = dict()
with open(INDEX, "r") as fp:
    for line in csv.reader(fp):
        index_uuid.setdefault(line[1],line[0])

filters = glob.glob(dir_folder+"*/*/block_info.csv")

for k,v in index_uuid.items():
    if not os.path.exists(dir_folder + v + "/block_info.csv"):
        continue
    with open(dir_folder + v + "/block_info.csv", "r") as fp:
        print dir_folder + v + "/block_info.csv"
        #func_name_set = set()
        block_num = 0
        func_name = ""
        func_uuid = ""
        for line in csv.reader(fp):
            #00x0L,chmod_or_fchmod,00x0L,0,1,0,0,0,9,0,0,2,0.0
            if line[1] == func_name:
                block_num = block_num + 1
            else :
                print "new function : ",line[1],line[2]
                if not func_name == "" :
                    #print "             ",k,func_name,func_uuid
                    #func_name_set.remove(func_name)
                    program = v.split('/')[0]
                    version = v.split('/')[1].split('_')[0]
                    func_list_fp.write(func_name+","+str(block_num)+","+func_uuid+","+k+","+v+","+program+","+version+"\n")
                #func_name_set.add(line[1])
                block_num = 1
                func_name = line[1]
                func_uuid = line[2]
func_list_fp.close()

train_fp = open(train_file, "w")
test_fp = open(test_file, "w")
vaild_fp = open(vaild_file, "w")

func_list_arr = []
func_list_dict = {}
#chmod_or_fchmod,  3 ,  00x0L ,  '1.70 ,coreutils/coreutils6.7_mipsel32_gcc5.5_o2,coreutils,coreutils6.7
with open(func_list_file, "r") as fp:
    for line in csv.reader(fp):
        if line[0] == "":
            continue
        if block_num_max > 0:
            if not ( int(line[1]) > block_num_min and int(line[1]) <= block_num_max ) :
                continue
        if func_list_dict.has_key(line[0]):
            value = func_list_dict.pop(line[0])
            value = value + "," + line[3]+"." + line[2]
            func_list_dict.setdefault(line[0],value)
        else:
            #print line
            value = line[3]+"." + line[2]
            func_list_arr.append(line[0])
            func_list_dict.setdefault(line[0],value)

random.shuffle(func_list_arr)
func_list_test = []
func_list_train = []
func_list_vaild = []
for i in xrange(len(func_list_arr)):
    if i%12==0:
        func_list_test.append(func_list_arr[i])
    elif i%12==1:
        func_list_vaild.append(func_list_arr[i])
    else:
        func_list_train.append(func_list_arr[i])

count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
while count < train_dataset_num:
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_train,1)
        value = func_list_dict.get(random_func[0])
        select_list = value.split(',')
        if(len(select_list)<2):
            continue
        selected_list = random.sample(select_list,2)
        train_fp.write(selected_list[0]+","+selected_list[1]+",1\n")
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_train,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        train_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0

count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
while count < test_dataset_num:
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_test,1)
        value = func_list_dict.get(random_func[0])
        select_list = value.split(',')
        if(len(select_list)<2):
            continue
        selected_list = random.sample(select_list,2)
        test_fp.write(selected_list[0]+","+selected_list[1]+",1\n")
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_test,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        test_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0

count = 0 #记录样本总量
cur_num = 0 #记录当前轮次 正例/负例 数量
while count < vaild_dataset_num:
    # 生成正例
    if cur_num < pos_num:
        random_func = random.sample(func_list_vaild,1)
        value = func_list_dict.get(random_func[0])
        select_list = value.split(',')
        if(len(select_list)<2):
            continue
        selected_list = random.sample(select_list,2)
        vaild_fp.write(selected_list[0]+","+selected_list[1]+",1\n")
    # 生成负例
    elif cur_num < pos_num + neg_num:
        random_func = random.sample(func_list_vaild,2)
        value1 = func_list_dict.get(random_func[0])
        select_list1 = value1.split(',')
        value2 = func_list_dict.get(random_func[1])
        select_list2 = value2.split(',')
        selected_list1 = random.sample(select_list1,1)
        selected_list2 = random.sample(select_list2,1)
        vaild_fp.write(selected_list1[0]+","+selected_list2[0]+",-1\n")
    cur_num += 1
    count += 1
    if cur_num == pos_num+neg_num:
        cur_num=0