
# coding: utf-8
from sklearn.linear_model import LinearRegression
import numpy as np
def mean_(column, nan_list):
    column_ = column.copy()
    for i in nan_list:
        del column_[i]
    return sum(list(column_))/len(column_)

def median_(column, nan_list):
    column_ = column.copy()
    for i in nan_list:
        del column_[i]
    sort_column = sorted(list(column_),reverse=True)
    if len(list(sort_column))%2==1:
        return sort_column[len(sort_column)//2]
    else:
        return (sort_column[len(sort_column)//2-1]+sort_column[len(sort_column)//2])/2

def moda_(column, nan_list):
    column_ = column.copy()
    for k in nan_list:
        del column_[k]
    column_=list(column_)
    count_prev = 0
    number = column_[0]
    for i in range(len(column_)):
        count = 1
        for j in column_[i:]:
            if column_[i] == j:
                count+=1
        if count>count_prev:
            number=column_[i]
            count_prev=count   
    return number

def get_nan_list(column):
    from math import isnan
    nan_list = []
    for i in range(len(column)):
        if isnan(column[i]): 
            nan_list.append(i)
    return nan_list

def get_train_test(data, column_name, nan_list):
    all_list = list(range(0, data.shape[0]))
    not_nan_list = list(set(all_list)-set(nan_list))
    data_predict = data.iloc[nan_list, :]
    data_train = data.iloc[not_nan_list, :]
    return data_train, data_predict

def get_columns(data, column_name):
    columns = list(data.columns)
    del columns[columns.index(column_name)]
    return columns
    
def euclidean_distance(node1, node2, shape):
    import math
    distance = 0
    for x in shape:
        distance += pow((node1[x] - node2[x]), 2)
    return math.sqrt(distance)

def get_neighbors(data, column_name, amount_k):
    if data[column_name].dtype=='object':
        raise ValueError
    nan_list = get_nan_list(data[column_name])
    all_list = list(range(0, data.shape[0]))
    not_nan_list = list(set(all_list)-set(nan_list))
    if len(nan_list)==0:
        raise ValueError
    data_train, data_predict = get_train_test(data, column_name, nan_list)        
    columns = get_columns(data, column_name)
    all_neighbors = []
    for y in nan_list:
        distances = []
        for x in not_nan_list:
            dist = euclidean_distance(
                data_predict.loc[y, columns], 
                data_train.loc[x, columns], 
                columns)
            distances.append((x, dist))
        distances.sort(key=lambda x:x[1])
        all_neighbors.append(distances[:amount_k])  
        print(y)  
    return all_neighbors

def get_target(neighbors, target):
    sum_ = 0
    sum_d = 0
    for i in neighbors:
        if i[1]==0:
            return target[i[0]]
        sum_d +=1/i[1]
        sum_+=1/i[1]*target[i[0]]
    return sum_/sum_d


def drop_null(data, axis_, number):
    if axis_==1:
        new_data = data
        for number_ in number:
            new_data = pd.concat([new_data.iloc[:, :number_], 
                                  new_data.iloc[:, number_+1:]], axis=axis_)
    if axis_==0:
        new_data = data
        for number_ in number:
            new_data = pd.concat([new_data.iloc[:number_, :], 
                                  new_data.iloc[number_+1:, :]], axis=axis_, ignore_index=True)
    else:
        raise ValueError
    return new_data


def replace_centr_val(column, method):
    nan_list = get_nan_list(column)
    if method=='mean':
        central =  mean_(column, nan_list)
    if method=='median':
        central = median_(column, nan_list)
    if method=='moda':
        central = moda_(column, nan_list)
    for i in nan_list:
        column[i]=central
    return(column)


def replace_lin_reg(data, column_name):
    if data[column_name].dtype=='object':
        raise ValueError
    nan_list = get_nan_list(data[column_name])
    if len(nan_list)==0:
        raise ValueError
    data_train, data_predict = get_train_test(data, column_name, nan_list)        
    columns = get_columns(data, column_name)
    model = LinearRegression().fit(data_train.loc[:, columns], data_train.loc[:, column_name])
    result = model.predict(data_predict.loc[:, columns])
    data_=data.copy(deep=True)
    for i in range(len(nan_list)):
        data_.loc[nan_list[i], column_name]=result[i]
    return data_


def replace_knn(data, column_name, amount_k):
    if data[column_name].dtype=='object':
        raise ValueError
    nan_list = get_nan_list(data[column_name])
    if len(nan_list)==0:
        raise ValueError
    predictions = []
    neighbors = get_neighbors(data, column_name, amount_k)
    for i in neighbors:
        predictions.append(get_target(i, data[column_name]))
    data_=data.copy(deep=True)
    for i in range(len(nan_list)):
        data_.loc[nan_list[i], column_name]=predictions[i]
    return data_


def standart(column):
    standart_column = []
    for i in column:
        standart_column.append((i-column.mean())/np.std(column, ddof=1))
    return standart_column


def mashtab(column):
    mashtab_column = []
    for i in column:
        mashtab_column.append((i-min(column))/(max(column)-min(column)))
    return mashtab_column

