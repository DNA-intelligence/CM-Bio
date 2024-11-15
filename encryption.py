# -*- coding:utf-8 -*-

"""
@Filename: encryption.py

@Author: Wang Yu

@Time: 2023/07/24 14:54:12
"""

import os
import math
import argparse
import cv2
import numpy as np
import json

r1_list = [["A","T"],["C","G"]]

r2_list = ["A","C","G","T"]

r3_list = ["AC","AG","TC","TG","CA","CT","GA","GT"]

r4_list = ["ACA","ACT","GAC","GAG","CAC","CAG","TCA", 
           "TCT","AGA","AGT","GTC","GTG","CTC","CTG", 
           "TGA","TGT"
        ]

r5_list = ["ACA","ACT","ACG","CAT","CAG","CAC","TAC", 
           "TAG","GAT","GAC","GAG","AGA","AGC","AGT", 
           "CGA","CGT","TCA","TCT","TCG","GCA","GCT", 
           "ATC","ATG","CTA","CTC","CTG","TGA","TGC", 
           "TGT","GTG","GTA","GTC"
            ]


r6_list = ['ACTC','ACTG','AGAC','AGAG','TCAG','TCAC','CTTC','GATG',
    'TGTC','TGTG','CACA','CACT','CTCA','CTCT','CTTG','GTAG',
    'GACA','GACT','GTCA','GTCT','ACAC','ACAG','GTTC','AGGA',
    'AGTC','AGTG','TCTC','TCTG','TGAG','TGAC','GTTG','AGGT',
    'CAGA','CAGT','CTGA','CTGT','GAGA','GAGT','CATG','TGGA',
    'GTGA','GTGT','ACCA','ACCT','TCCA','TCCT','CTAG','TGGT',
    'AGCT','ACGT','AGCA','ACGA','CAAG','CAAC','GATC','GTAC',
    'GAAC','GAAG','TCGA','TGCA','TGCT','TCGT','CATC','CTAC'
    ]

r7_res = ["ACCA","ACCT","TCCA","TCCT","CAAG","CAAC", 
          "GAAC","GAAG","CTTC","CTTG","GTTC","GTTG", 
          "AGGA","AGGT","TGGA","TGGT","ACCG","TGGC", 
          "CAAT","GTTA"
        ]

xor_rule = [{'A':'T', 'T':'A', 'C':'G', 'G':'C'},
            {'A':'C', 'C':'A', 'T':'G', 'G':'T'},
            {'A':'G', 'G':'A', 'T':'C', 'C':'T'}]

def get_opts():
    opts = argparse.ArgumentParser()
    opts.add_argument('-i', '--input', help='input file', required=True)
    opts.add_argument('-o', '--output', help='output file', required=True)
    opts.add_argument('-c', '--config', help='config file', required=True)
    return opts.parse_args()

def generate_r7():
    '''
    生成r7的对应密码子
    '''
    base = {"A","T","C","G"}
    a_list = []
    for i in base:
        a = i
        for j in base.difference(i):
            tmp1 = a + j
            for k in base.difference(j):
                tmp2 = tmp1 + k
                for l in base.difference(k):
                    tmp3 = tmp2 + l
                    a_list.append(tmp3)

    return a_list+r7_res

def select_table(value: int):
    '''
    选择对应编码表
    '''
    table = {
        2**1: r1_list,
        2**2: r2_list,
        2**3: r3_list,
        2**4: r4_list,
        2**5: r5_list,
        2**6: r6_list,
        2**7: generate_r7,
    }
    return table.get(value, "Wrong Value!")

def logistic_map(u: float,z: float,iters: int) -> list:
    '''
    生成logistic映射
    u: [0,4]
    z: != {0, 0.25, 0.5, 0.75, 1.0}
    '''
    z_list = []
    for i in range(iters):
        z = u * z * (1 - z)
        z_list.append(z)
    return z_list

def sine_map(x0, a, max_g):
    x = x0
    x_list = []
    for i in range(max_g):
        x = (a / 4) * math.sin(math.pi * x)
        x_list.append(x)
    return x_list

def pwlcm(P0, p, T):
    y = []
    for i in range(T):
        if P0 >= 0 and P0 < p:
            P0 = P0 / p
        elif P0 >= p and P0 < 0.5:
            P0 = (P0 - p) / (0.5 - p)
        elif P0 >= 0.5 and P0 < 1 - p:
            P0 = (1 - p - P0) / (0.5 - p)
        elif P0 >= (1 - p) and P0 < 1:
            P0 = (1 - P0) / p
        y.append(P0)
    return y
    
def index_sort(data: list, arr: list) -> list:
    '''
    对序列排序输出排序后的索引
    '''
    new_arr = []
    arr_sort = np.argsort(arr)
    for i in arr_sort:
        new_arr.append(data[i])
    return new_arr

def recover_sort(data: list, arr: list) -> list:
    arr_sort = np.argsort(arr)
    new_data = ['']*len(data)
    for i,j in enumerate(arr_sort):
        new_data[j] = data[i]
    return new_data

def Rossler(T):
    h=0.01
    # 初始化Rossler模型参数
    w=1
    p=0.2
    q=0.2
    r=5.7
    x0=0.923823        # 初始值
    y0=0.892792
    z0=0.1484840
    x=[]
    y=[]
    z=[]
    for t in range(T):
        xt=x0+h*(-w*y0-z0)
        yt=y0+h*(w*x0+p*y0)
        zt=z0+h*(q+z0*(x0-r))

        #x0、y0、z0统一更新
        x0,y0,z0=xt,yt,zt
        x.append(x0)
        y.append(y0)
        z.append(z0)
    return x,y,z

def tentmap(alpha, x0, max_g):
    x = x0
    x_list = []
    for i in range(max_g):
        if x < alpha:
            x = x / alpha
        else:
            x = (1 - x) / (1 - alpha)
        x_list.append(x)
    return x_list

def read_pic(input):
    '''
    读取图片文件
    '''
    img = cv2.imread(input)
    (b,g,r) = cv2.split(img)
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    
    width,height,_ = img.shape
    x,y,z = Rossler(width*height)
    
    new_b = np.array(index_sort(b,x))
    new_g = np.array(index_sort(g,y))
    new_r = np.array(index_sort(r,z))
    
    new_b = new_b.reshape(width,height)
    new_g = new_g.reshape(width,height)
    new_r = new_r.reshape(width,height)
    new_img = cv2.merge([new_b,new_g,new_r])
    return new_img, width, height

def write_pic(img, output):
    img = cv2.imread(input)
    (b,g,r) = cv2.split(img)
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()
    
    width,height,_ = img.shape
    x,y,z = Rossler(width*height)
    
    new_b = np.array(recover_sort(b,x))
    new_g = np.array(recover_sort(g,y))
    new_r = np.array(recover_sort(r,z))
    
    new_b = new_b.reshape(width,height)
    new_g = new_g.reshape(width,height)
    new_r = new_r.reshape(width,height)
    new_img = cv2.merge([new_b,new_g,new_r])
    out = output+'_out.bmp'
    cv2.imwrite('out.bmp',new_img)
    
def encode(input, codon_dict):
    '''
    对输入文件进行编码
    r_dna: 行索引DNA
    c_dna: 列索引DNA
    '''
    r_dna = list()
    c_dna = list()

    input, width, height = read_pic(input)
    input_1d = list(input.flatten())
    for i in input_1d:
        r_dna.append(codon_dict[i][0])
        c_dna.append(codon_dict[i][1])
    return r_dna, c_dna, width, height

def decode_tmp(r_dna, c_dna, condon_dict, width, height, output):
    new_condon = dict()
    for key, value in condon_dict.items():
        key1 = '_'.join(value)
        value1 = key
        new_condon[key1] = value1

    pixes = []
    for i,j in zip(r_dna, c_dna):
        key = i+'_'+j
        pixes.append(new_condon[key])
    img = np.array(pixes).reshape((width, height, 3))
    os.chdir(output)
    out = 'encrypted_' + output + '.png'
    print(out)
    cv2.imwrite(out, img,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])    

if __name__ == '__main__':
    opts = get_opts()
    input = opts.input
    out = opts.output
    
    with open(opts.config, 'r') as f:
        config = json.load(f)
        
    cwd = os.getcwd()
    output = cwd + os.sep + out
    if os.path.exists(output):
        pass
    else:
        os.mkdir(output)

    # 初始化行、列的数目和密码子数量
    # x = int(np.random.random(1)*8)
    r = config['Table_1']
    c = config['Table_2']
    row_nu = 2 ** r
    col_nu = 2 ** c
    cn = 256
    
    # 初始化混沌映射参数u、z
    u = 2.892
    z = 0.8
    limit = cn
        
    # 生成编码表
    print("生成编码表!")
    col_name = select_table(col_nu)
    row_name = select_table(row_nu)
    count = 0
    codon_dict = dict()
    print("col_name:{}, row_name:{}".format(col_name,row_name))
    
    for i in row_name:
        for j in col_name:
            key = count
            codon_dict[key] = [i,j]
            count += 1
    # print(codon_dict)
    
    # 信息编码
    print("开始编码！")
    r_dna, c_dna, width, height = encode(input, codon_dict)

    # logistic函数、sine函数置乱
    log_list = logistic_map(u,z,len(r_dna))
    new_r_dna = index_sort(r_dna,log_list)
    sine_list = sine_map(0.598293, 3.57,len(r_dna))
    new_c_dna = index_sort(c_dna,sine_list)
    
    # 扩散
    p0_r = 0.44453445
    p_r = 0.5423
    p0_c = 0.59223894
    p_c = 0.6323
    pwlcm_list_r = pwlcm(p0_r, p_r, len(new_r_dna))
    pwlcm_list_c = pwlcm(p0_c, p_c, len(new_c_dna))

    ect_c_dna = []
    ect_r_dna = []
    for i,k in enumerate(new_r_dna):
        if int(pwlcm_list_r[i]* (10**12)) % 3  == 0:
            rule = xor_rule[0]
        elif int(pwlcm_list_r[i]* (10**12)) % 3  == 1:
            rule = xor_rule[1]
        else:
            rule = xor_rule[2]
        ss = ''
        for j in k:
            ss += rule[j]
        ect_r_dna.append(ss)
        
    for i,k in enumerate(new_c_dna):
        if int(pwlcm_list_c[i]* (10**12)) % 3  == 0:
            rule = xor_rule[0]
        elif int(pwlcm_list_c[i]* (10**12)) % 3  == 1:
            rule = xor_rule[1]
        else:
            rule = xor_rule[2]
        ss = ''
        for j in k:
            ss += rule[j]
        ect_c_dna.append(ss)

    # 直接解码加密信息
    decode_tmp(ect_r_dna, ect_c_dna, codon_dict, width, height, out)
    r_dna = ''.join(ect_r_dna)
    c_dna = ''.join(ect_c_dna)
    with open(output+os.sep+"r_dna.fa","w") as f:
        f.write(r_dna)
    
    with open(output+os.sep+"c_dna.fa","w") as f:
        f.write(c_dna)

    print("row_dna length: {}\n col_dna length: {}".format(len(r_dna), len(c_dna)))
    print("编码结束！")

