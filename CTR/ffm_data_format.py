#!/usr/bin/env python
# coding=utf-8

'''
libffm 训练数据格式:
<label> <field1>:<feature1>:<value1> <field2>:<feature2>:<value2> ...
==================================
Click  |  Adertiser  | Publisher
----------------------------------
    0  |     Nike    |   CNN
    1  |     ESPN    |   BBC
==================================

fields  : Adertiser and Publisher
features: Advertiser-Nike, Advertiser-ESPN, Publisher-CNN, Publisher-BBC

Usually you will need to build two dictionares, one for field and one for features, like this:
    DictFeature[Advertiser] -> 0
    DictFeature[Publisher]  -> 1
    
    DictFeature[Advertiser-Nike] -> 0
    DictFeature[Publisher-CNN]   -> 1
    DictFeature[Advertiser-ESPN] -> 2
    DictFeature[Publisher-BBC]   -> 3

Then, you can generate FFM format data:

    0 0:0:1 1:1:1
    1 0:2:1 1:3:1
'''

from codecs import open
import pandas as pd
from collections import defaultdict

DictField = {'groupid':0, 'hot':1, 'fresh':2, 'topic':3, 'category':4}
DictFeature = defaultdict(dict)

filter_cols = ['click', 'news_id']
test_data = pd.read_csv('./test_ids.data')
columns = test_data.columns
ulity_cols = list()
col_field  = dict()
for colname in columns:
    if colname in filter_cols:
        continue
    ulity_cols.append(colname)
    if colname.isdigit():
        DictFeature['groupid'][colname] = len(DictFeature['groupid'])
        col_field[colname] = 'groupid'
    elif colname.startswith('topic'):
        DictFeature['topic'][colname] = len(DictFeature['topic'])
        col_field[colname] = 'topic'
    elif colname.startswith('category'):
        DictFeature['category'][colname] = len(DictFeature['category'])
        col_field[colname] = 'category'
    else:
        DictFeature[colname][colname] = len((DictFeature[colname]))
        col_field[colname] = colname

def write(lines):
    print 'start to write libffm file.'
    with open('libffm_file_va.ffm', 'a+', encoding='utf8') as f:
        f.write('\n'.join(lines) + '\n')

lines = list()
for index, row in test_data.iterrows():
    libffm_line = list()
    label = str(int(row['click']))
    libffm_line.append(label)
    for col in ulity_cols:
        field = str(DictField.get(col_field.get(col)))
        feature = str(DictFeature.get(col_field.get(col)).get(col))
        value   = str(row[col])
        ffm_ = ':'.join([field, feature, value])
        libffm_line.append(ffm_)
    libffm_line = ' '.join(libffm_line).rstrip()
    lines.append(libffm_line)
    if len(lines) == 10000:
        write(lines)
        del lines[:]

if lines:
    write(lines)
    del lines[:]
