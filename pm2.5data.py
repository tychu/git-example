#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from PIL import Image
import os, os.path

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator







#### create date
#td = pd.date_range(start='1/1/2018', end='12/31/2018 23:00:00', freq='H')
#date_np = td.values
#date_np = date_np.reshape(1, -1)
#print(date_np.shape)
#print('date data type', type(date_np))
#print(date_np)
#
#reg = np.full([13, 365*24], np.nan)
#print(reg.shape)
#print('reg data type', type(reg))
#print(reg)
#reg_data = np.append(reg, date_np, axis = 1)
#print(reg_data.shape)
#
#
#df = pd.DataFrame(reg_data, columns = ["Date", "PM2.5", "PM10", "SO2", "NOx", "O3", "CO", "NO", "WD_HR", "WS_HR", "RH", "WIND_DIREC", "WIND_SPEED", "AMB_TEMP"])


#df = pd.DataFrame(columns = ["Date", "PM2.5", "PM10", "SO2", "NOx", "O3", "CO", "NO", "WD_HR", "WS_HR", "RH", "WIND_DIREC", "WIND_SPEED", "AMB_TEMP"])
#('PM2.5', 'PM10', 'SO2', 'NOx', 'O3', 'CO', 'NO', 'WD_HR', 'WS_HR', 'RH', 'WIND_DIREC', 'WIND_SPEED', 'AMB_TEMP')
#print(td)



# In[ ]:



import xlrd
#data = xlrd.open_workbook('D:/dawnchu/106_HOUR_00_20180308/106年北部空品區/106年古亭站_20180309.xls')
#datatest = northdata.iloc[:,3:]

#data = pd.read_excel('./106年古亭站_20180309.xls', na_values = ' ')
#data = pd.read_excel('./103_.xls', na_values = ' ')


import os

 

# 指定要查詢的路徑

#yourPath = '/home/data/chudawn/'
#yourPath = '/home/data/dawn/' # .184
# 列出指定路徑底下所有檔案(包含資料夾)
#allFileList = os.listdir(yourPath)
#print(allFileList)
## 逐一查詢檔案清
#for file in allFileList:
##   這邊也可以情況，做檔案的操作(複製、讀取...等)
##   使用isdir檢查是否為目錄
##   使用join的方式把路徑與檔案名稱串起來(等同filePath+fileName)
#  if "中部" in file:
#    print(file)
##  if os.path.isdir(os.path.join(yourPath,file)):
##    print("I'm a directory: " + file)
##   使用isfile判斷是否為檔案
#  elif os.path.isfile(yourPath+file):
#    #if "忠明" in file:
#    print(file)
#  else:
#    print('OH MY GOD !!')


#for file_name in os.list   dir(yourPath):
#    if fnmatch.fnmatch(file_name, '*.txt'):
#        print(file_name)
#for root, dirs, files in os.walk('lang/'):
#    for file in files:
#        filename, extension = os.path.splitext(file)
#        if extension == '.txt':
#            # Do Some Task
#        if 'hello' in filename:
    # Do Some Task
from collections import Counter



def read18file(path):

    data = pd.read_excel(path, na_values = ' ')
    date = np.array(data.iloc[:,0])
    result = Counter(date)
    most_common = result.most_common()
    print(np.array(most_common).shape)
    print(np.array(most_common)[0, 1])
    datatest = np.array(data.iloc[:,3:])
    print(datatest.shape)
    for i in range(6570):
        for j in range(24):
            try:
                float(datatest[i,j])
            except ValueError:
                datatest[i, j]=np.nan
    month_to_data = {}  ## Dictionary (key:month , value:data)
    mon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mc = 0
    sample = np.empty(shape = (18, 365*24))
    for month in range(12):
        for day in range(mon[month]):
            for hour in range(24):
                sample[:,(mc+day) * 24 + hour] = datatest[18 * (mc+ day): 18 * (mc + day + 1),hour]
        mc = mc + mon[month]
        print(month, day, hour)
    print(sample.shape)
    return sample

def read20file(path):
    data = pd.read_excel(path, na_values = ' ')
    datatest = np.array(data.iloc[:,3:])
    print(datatest.shape)
    for i in range(7300):
        for j in range(24):
            try:
                float(datatest[i,j])
            except ValueError:
                datatest[i, j]=np.nan
    month_to_data = {}  ## Dictionary (key:month , value:data)
    mon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    mc = 0
    sample = np.empty(shape = (20, 365*24))
    for month in range(12):
        for day in range(mon[month]):
            for hour in range(24):
                sample[:,(mc+day) * 24 + hour] = datatest[20 * (mc+ day): 20 * (mc + day + 1),hour]
        mc = mc + mon[month]
        print(month, day, hour)
    print(sample.shape)
    return sample




import xlrd
import datetime

def convertdata(DataFrame, year):
    var = ["PM2.5", "PM10", "SO2", "NOx", "O3", "CO", "NO", "WD_HR", "WS_HR", "RH", "WIND_DIREC", "WIND_SPEED", "AMB_TEMP"]
    leng = len(var)
    DataFrame = DataFrame.to_numpy()
    irrey = np.zeros((366*24, 1))
    rey = np.zeros((365*24, 1))
    year = year+1911
    if year % 4 == 0:
        #print('YES 366')
        fix = irrey  ### np.zeros((366*24, 1))
        checkcompleteday = 366
        diffmon = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        fix = rey    ### np.zeros((365*24, 1))
        checkcompleteday = 365
        diffmon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for i in range(leng) :
        print(var[i])
        
        colvar = DataFrame[:, 2] # the column that is data of weather variance 
        vardata = DataFrame[colvar == var[i], :] # take data of only one weather variable 
        
        ### delet variance column ####
        #vardata_del = np.delete(vardata, [1, 2, 3], 1)
        #print(vardata_del[0, :])
        len_vardata = vardata.shape
        print(len_vardata)
        check = len_vardata[0]

        if check == checkcompleteday:
            #print('NO PROBLEM')
            ### delet variance column ####
            vardata_del = np.delete(vardata, [0, 1, 2], 1)
            varvec = vardata_del.reshape(-1, 1)
            #print(varvec.shape)
            #nonlocal irrey
            fix = np.append(fix, varvec, axis = 1)
            print(fix.shape)
            print(varvec.shape)
            #finaldata = irrey

        else:
            print('BIG PROBLEM !!!!!!')
            vardata_del = np.delete(vardata, [1, 2], 1)
            #print(vardata_del[0, :])
            mon = diffmon
            mc = 0
            fixdata = np.zeros((1, 1))

            for month in range(12):
                d = 0
                for day in range(mon[month]):
                    
                    
                    stad = datetime.date(year, (month+1), (day+1) )
                    date_str = vardata_del[mc+d, 0]
                    date_object = datetime.datetime.strptime(date_str, '%Y/%m/%d').date()
                    #print('right time', stad)
                    #print('data time', date_object)
                    if date_object == stad:
                        getdata = vardata_del[mc+day, 1:]
                        #print('GET !!!!!!', getdata)
                        #getdata = np.delete(getdata, 0, 0)
                        #print(getdata.shape)
                        fixdata = np.append(fixdata, getdata)
                        #print(fixdata.shape)
                        d += 1
                    else:
                        print('right time', stad)
                        print('data time', date_object)
                        print('DO NOT GET')
                        fixdata = np.append(fixdata, np.full([1,24], np.nan))
                        print(fixdata.shape)
                        
                mc = mc + d
            
            fixdata = np.delete(fixdata, 0, 0)
            print(fixdata.reshape(-1, 1).shape)
            print(fix.shape)
            fix = np.append(fix, fixdata.reshape(-1, 1), axis = 1)
            #finaldata = irrey
    finaldata = np.delete(fix, 0, 1)
    print(finaldata.shape)
    return finaldata
        #else :
        #    fix = rey
            #print('NO 365')
#            if check == 365:
#                #print('NO PROBLEM')
#                vardata_del = np.delete(vardata, [0, 1, 2], 1)
#                varvec = vardata_del.reshape(-1, 1)
#                print(varvec.shape)
#                #nonlocal rey             
#                fix = np.append(fix, varvec, axis = 1)
#                print(fix.shape)
#                #finaldata = rey
#
#            else:
#                print('BIG PROBLEM !!!!!!')
#                vardata_del = np.delete(vardata, [1, 2], 1)
#                #print(vardata_del[0, :])
#                mon = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#                mc = 0
#                fixdata = np.zeros((1, 1))
#
#                for month in range(12):
#                    d = 0
#                    for day in range(mon[month]):
#                        
#                        
#                        stad = datetime.date(year, (month+1), (day+1) )
#                        date_str = vardata_del[mc+d, 0]
#                        date_object = datetime.datetime.strptime(date_str, '%Y/%m/%d').date()
#                        #print('right time', stad)
#                        #print('data time', date_object)
#                        if date_object == stad:
#                            getdata = vardata_del[mc+day, 1:]
#                            print('GET !!!!!!', getdata)
#                            #getdata = np.delete(getdata, 0, 0)
#                            #print(getdata.shape)
#                            fixdata = np.append(fixdata, getdata)
#                            print(fixdata.shape)
#                            d += 1
#                        else:
#                            print('right time', stad)
#                            print('data time', date_object)
#                            print('DO NOT GET')
#                            fixdata = np.append(fixdata, np.full([1,24], np.nan))
#                            print(fixdata.shape)
#                            
#                    mc = mc + d
#                
#                fixdata = np.delete(fixdata, 0, 0)
#                print(fixdata.shape)
#                fix = np.append(fix, fixdata, axis = 1)
    







#year = ['103', '104', '105', '106', '107', '108']
#all_data = pd.DataFrame()
#all_data = np.array(18)
# 與listdir不同的是，listdir只是將指定路徑底下的目錄和檔案列出來
# walk的方式則會將指定路徑底下所有的目錄與檔案都列出來(包含子目錄以及子目錄底下的檔案)

# 列出所有子目錄與子目錄底下所有的檔案


#for i in range(6):
#    print(i)
def exceldata(region, name):
    d = np.zeros((1, 13))
    yourPath = '/home/data/chudawn/'
    allList = os.walk(yourPath)
    for root, dirs, files in sorted(allList):
        #print(root)
    #列出目前讀取到的路徑
        #print(root)
    # for i in range(6):
    #    print(year[i]);
        
        if region in root: #"中部" in root :
            
        #if year[i] in root and "中部" in root :
            #print("path：", root)
            for file in files:
                filename, extension = os.path.splitext(file)
                #print(filename)
                if name in filename and extension == '.xls': #"大里" in filename and extension == '.xls':
                    print(filename)
                    year = int(filename[0:3])
                    #df = read18file(root+'/'+filename+extension)
                    df = pd.read_excel(root+'/'+filename+extension, na_values = ' ')
                    #print(df.head())
                    #print(df.info)
                    #all_data = all_data.append(df,ignore_index=True)
                    #npdata = np.append(all_data, np.transpose(df))
                    #print(df.to_numpy().shape)
                    
                    newdata = convertdata(df, year)
                    print(newdata.shape)
                    d = np.append(d, newdata, axis = 0)
                    print(d.shape)
    
                    break
            else:
                pass
                print('NOT HERE')
    wholedata = np.delete(d, 0, 0)
    return wholedata

#data = exceldata("中部", "豐原") #  zhongming 忠明 dali 大里 shalu 沙鹿 xitun 西屯 fengyuan 豐原
#np.save('/home/data/chudawn/fengyuan', data)

#################
# interpolation #
#################

def clean_data(data):
    var = ["PM2.5", "PM10", "SO2", "NOx", "O3", "CO", "NO", "WD_HR", "WS_HR", "RH", "WIND_DIREC", "WIND_SPEED", "AMB_TEMP"]
    variable = range(data.shape[1])
    index = range(data.shape[0])
    #print(max(index))
    plusdata = np.zeros((1, 1))

    xend = data.shape[0] #len(data[:, 0])
    qlag = 24
    #xwdss = data[:, col]
    
    nn_in_3dim = np.zeros((xend-qlag,24)).reshape(-1, 24, 1)
    for col in variable:
        print(var[col])
        nn_in = np.zeros(xend-qlag).reshape(-1, 1)
        #print(max(data[:, col]))
        for i in index:
            datatype = type(data[i, col])
            
            if isinstance(data[i, col], str) == True: # data cannnot use
                #print(datatype)
                #print(" WRONG !!!!!!! ")
                #print(data[i, col])
                data[i, col] = np.nan
                #pass
                if var[col] == 'WIND_DIREC':
                    #print("WIND_DIREC")

                    if data[i, col] == 888:
                        print("NO WIND")
                        print(data[i, 10])
                        data[i, col] = 0
                    elif data[i, col] == 999:
                        print("MACHINE MALFUNCTION")
                        print(data[i, 10])
                        data[i, col] = np.nan
                    else:
                        continue
                else:
                    continue

        #print(pd.DataFrame(data[:, col]).isnull().sum() )
        #ndata = pd.DataFrame(data).interpolate(method='linear', limit_direction='forward', axis=0, limit = 3) #4
        #for i in range(13):
        #    print("i : ", i)
        #    pddata = pd.DataFrame(data)
        #    pddata.loc[:, i] = pd.to_numeric(data[:, i], errors='coerce')
        variable = pd.to_numeric(data[:, col], errors='coerce')
        data[:, col] = pd.DataFrame(variable).interpolate(method='linear', limit_direction='forward', axis=0, limit = 5).loc[:, 0] #4


        ##############
        # time lapse #
        ##############
        #xend = len(data[:, col])
        #qlag = 23
        #xwdss = data[:, col]
        #nn_in = np.zeros(xend-(qlag)).reshape(-1, 1)
        #print('nn_in shape : ', nn_in.shape )
        print('MIN : ', min(data[:, col]), 'MAX : ', max(data[:, col]))
        data[:, col] = (data[:, col]-min(data[:, col]))/(max(data[:, col])-min(data[:, col])) 
        print('MIN : ', min(data[:, col]), 'MAX : ', max(data[:, col]))


        for lag in  range(24):#tuple(range(2, qlag)):
            #print("lag : ", lag)
            lagsep = range(lag, xend-(qlag-lag))
            #print('lagsep : ', lagsep.shape)
            tmp = data[lagsep, col]#xwdss[lagsep]
            
            #print('tmp shape : ', tmp.shape)
            #print('tmp type : ', type(tmp))
            nn_in = np.append(nn_in, tmp.reshape(-1, 1), axis = 1)
            #print('nn_in shape : ', nn_in.shape )
        if var[col] == 'PM2.5' :
            lagsep = range(lag+1, xend-(qlag-lag-1))
            y = data[lagsep, col]
        #else:
        #    continue
        nn_in = np.delete(nn_in, 0, 1)
        nn_in_3dim = np.concatenate([nn_in_3dim, nn_in.reshape(-1, 24, 1)], 2)
        #print("nn_in_3dim : ", nn_in_3dim.shape)
    nn_in_3dim = np.delete(nn_in_3dim, 0, 2)
    #print("nn_in_3dim : ", nn_in_3dim.shape)
    return nn_in_3dim, y


def countna(filename):
    root = '/home/data/chudawn/'
    xfiletype = '_x.npy'
    yfiletype = '_y.npy'

    # df = pd.read_excel(root+'/'+filename+extension, na_values = ' ')
    xdata = np.load(root + filename + xfiletype, allow_pickle = True)#place[i] + xfiletype, allow_pickle = True)
    ydata = np.load(root + filename + yfiletype, allow_pickle = True)#place[i] + yfiletype, allow_pickle = True)
    #data = np.append(xdata, ydata.reshape(-1, 1, 1), axis = 2)
    print("x data shape : ", xdata.shape)

    data = xdata
    datanacount = np.zeros((data.shape[0], 1))
    #print("data.shape[0] : ", data.shape[0])
    #print("data.shape[1] : ", data.shape[2])

    #try :
    #    print(data.shape[2])
    #except IndexError:
        #print(" FOUND Y !!!")
        #pass
        #for i in range(data.shape[0]):
    a = ydata.reshape(-1, 1)
        #print(nandata.shape)
        #nancolcount = np.zeros((data.shape[0], 1))        
    datanacount_y = np.count_nonzero((a != a), axis = 1) # count na
    print("datanacount_y : ", datanacount_y.shape)
    #else:
    for i in range(data.shape[2]):
        nandata = data[:, :, i]
        #print(nandata.shape)
        nancolcount = np.zeros((data.shape[0], 1))
        for col in range(len(nandata)):
            #print(len(nandata))
            #np.isnan(nandata[:, col]).sum()
            a = nandata[col, :]
            nancolcount[col, 0] = np.count_nonzero(a != a) # count na 
    
        datanacount = np.append(datanacount, nancolcount, axis = 1)
        #print(datanacount.shape)
    datanacount = np.delete(datanacount, 0, 1)
    print("datanacount : ", datanacount.shape)
    data_na = np.append(datanacount, datanacount_y.reshape(-1, 1), axis = 1)
    print("data_na : ", data_na.shape)
    finalna = np.sum( data_na, axis=1)
    checkna = (finalna == 0)
    print("checkna : ", checkna.shape)

    finaldata_x = data[checkna, :, :]
    finaldata_y = ydata[checkna]

    print("finaldata_x : ", finaldata_x.shape)
    print("finaldata_y : ", finaldata_y.shape)


    return data, ydata, finalna#finaldata_x, finaldata_y, finalna


#############
#           #
# IMPLEMENT #
#           #
#############
#place = ["zhongming", "dali", "shalu", "xitun", "fengyuan"]
#filenum = len(place)
#for i in range(filenum) :
#    root = '/home/data/chudawn/'
#    filetype = '.npy'
#    # df = pd.read_excel(root+'/'+filename+extension, na_values = ' ')
#    data = np.load(root + place[i] + filetype, allow_pickle = True)
#    
#    print("Place  :  ", place[i])
#
#    #dataclean = clean_data(data)
#    dataclean_x, dataclean_y = clean_data(data)
#    #dataclean_x = dataclean[0]
#    #dataclean_y = dataclean[1]
#    np.save( root + place[i] + '_xnorm' , dataclean_x) #'/home/data/chudawn/zhongming_x', dataclean_x)
#    print(" save x : ", dataclean_x.shape)
#    np.save( root + place[i] + '_ynorm' , dataclean_y) #  '/home/data/chudawn/zhongming_y', dataclean_y)
#    print("save y : ", dataclean_y.shape)
#
#    x, y, check = countna(place[i])
#    
#    #np.save( root + place[i] + '_xc' , x)
#    #print("x : ", x.shape())
#    #np.save( root + place[i] + '_yc' , y)
#    np.save( root + place[i] + '_check' , check)
#    #print("check : ", check.shape)


#colname = ["PM2.5", "PM10", "SO2", "NOx", "O3", "CO", "NO", "WD_HR", "WS_HR", "RH", "WIND_DIREC", "WIND_SPEED", "AMB_TEMP"]

#df = pd.DataFrame(data = data, index = list(range(data.shape[0])), columns = colname)

#td = pd.date_range(start='1/1/2014', end='12/31/2019 23:00:00', freq='H')
#date_np = td.values
#
#print(date_np.shape)





zhongming_check = np.load('/home/data/chudawn/zhongming_check.npy', allow_pickle = True)
dali_check = np.load('/home/data/chudawn/dali_check.npy', allow_pickle = True)
shalu_check = np.load('/home/data/chudawn/shalu_check.npy', allow_pickle = True)
xitun_check = np.load('/home/data/chudawn/xitun_check.npy', allow_pickle = True)
fengyuan_check = np.load('/home/data/chudawn/fengyuan_check.npy', allow_pickle = True)


#zhongming_x = np.load('/home/data/chudawn/zhongming_x.npy', allow_pickle = True)



#ckecknp = np.zeros((zhongming_check.shape[0], 1))
total = np.concatenate( (zhongming_check.reshape(-1, 1) 
    , dali_check.reshape(-1, 1), shalu_check.reshape(-1, 1)
    , xitun_check.reshape(-1, 1), fengyuan_check.reshape(-1, 1) ) , axis = 1)

#totalcheck = (((zhongming_check and dali_check) and shalu_check) and xitun_check) and fengyuan_check
print("total check na : ", total.shape)
#print("totalcheck.sum : ", totalcheck.sum())
ckna = np.sum( total, axis=1)
print("sum total na to one column : ", ckna.shape)
checkna = (ckna == 0)

#datatrue = zhongming_x[checkna, :, :]
#print(sum(checkna))
#print("data clean shape : ", datatrue.shape)

place = ["zhongming", "dali", "shalu", "xitun", "fengyuan"]
for i in range(5):
    root = '/home/data/chudawn/'
    xfiletype = '_xnorm.npy'
    yfiletype = '_ynorm.npy'
    xdata = np.load(root + place[i] + xfiletype, allow_pickle = True)#place[i] + xfiletype, allow_pickle = True)
    ydata = np.load(root + place[i] + yfiletype, allow_pickle = True)#place[i] + yfiletype, allow_pickle = True)

    xdatatrue = xdata[checkna, :, :]
    ydatatrue = ydata[checkna]

    print(' x data : ', xdatatrue.shape)
    print(' y data : ', ydatatrue.shape)

    np.save( root + place[i] + '_xcnorm' , xdatatrue)
    np.save( root + place[i] + '_ycnorm' , ydatatrue)

    print(" SAVED !!!! ")

