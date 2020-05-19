# Clean the data set download from Taiwan's Central Weather Bureau

import pandas as pd
import numpy as np
from PIL import Image
import os, os.path
import xlrd
from collections import Counter
import datetime

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator




# read data with 18 variable
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

# read data with 20 variable
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



# convert data into compelte data form
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
        len_vardata = vardata.shape
        print(len_vardata)
        check = len_vardata[0]

        if check == checkcompleteday:
            #print('NO PROBLEM')
            ### delet variance column ####
            vardata_del = np.delete(vardata, [0, 1, 2], 1)
            varvec = vardata_del.reshape(-1, 1)
            fix = np.append(fix, varvec, axis = 1)
            print(fix.shape)
            print(varvec.shape)

        else:
            print('BIG PROBLEM !!!!!!')
            vardata_del = np.delete(vardata, [1, 2], 1)
            mon = diffmon
            mc = 0
            fixdata = np.zeros((1, 1))

            for month in range(12):
                d = 0
                for day in range(mon[month]):
                    
                    
                    stad = datetime.date(year, (month+1), (day+1) )
                    date_str = vardata_del[mc+d, 0]
                    date_object = datetime.datetime.strptime(date_str, '%Y/%m/%d').date()
                    if date_object == stad:
                        getdata = vardata_del[mc+day, 1:]
                        #print('GET !!!!!!', getdata)
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
    finaldata = np.delete(fix, 0, 1)
    print(finaldata.shape)
    return finaldata

    


def exceldata(region, name):
    d = np.zeros((1, 13))
    yourPath = '/home/data/chudawn/'
    allList = os.walk(yourPath)
    for root, dirs, files in sorted(allList):
        #列出目前讀取到的路徑
        #print(root)      
        if region in root: #"中部" in root :
            for file in files:
                filename, extension = os.path.splitext(file)
                #print(filename)
                if name in filename and extension == '.xls': 
                    print(filename)
                    year = int(filename[0:3])
                    df = pd.read_excel(root+'/'+filename+extension, na_values = ' ')
                    
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

data = exceldata("中部", "豐原") #  zhongming 忠明 dali 大里 shalu 沙鹿 xitun 西屯 fengyuan 豐原
np.save('/home/data/chudawn/fengyuan', data)

#################
# interpolation #
#################

def clean_data(data):
    var = ["PM2.5", "PM10", "SO2", "NOx", "O3", "CO", "NO", "WD_HR", "WS_HR", "RH", "WIND_DIREC", "WIND_SPEED", "AMB_TEMP"]
    variable = range(data.shape[1])
    index = range(data.shape[0])
    plusdata = np.zeros((1, 1))

    xend = data.shape[0] #len(data[:, 0])
    qlag = 24
    
    nn_in_3dim = np.zeros((xend-qlag,24)).reshape(-1, 24, 1)
    for col in variable:
        print(var[col])
        nn_in = np.zeros(xend-qlag).reshape(-1, 1)
        for i in index:
            datatype = type(data[i, col])
            
            if isinstance(data[i, col], str) == True: # data cannnot use
                data[i, col] = np.nan
                #pass
                if var[col] == 'WIND_DIREC':
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

        variable = pd.to_numeric(data[:, col], errors='coerce')
        data[:, col] = pd.DataFrame(variable).interpolate(method='linear', limit_direction='forward', axis=0, limit = 5).loc[:, 0] #4


        ##############
        # time lapse #
        ##############
        #print('MIN : ', min(data[:, col]), 'MAX : ', max(data[:, col]))
        data[:, col] = (data[:, col]-min(data[:, col]))/(max(data[:, col])-min(data[:, col])) 
        #print('MIN : ', min(data[:, col]), 'MAX : ', max(data[:, col]))
        for lag in range(24):
            lagsep = range(lag, xend-(qlag-lag))
            tmp = data[lagsep, col]
            nn_in = np.append(nn_in, tmp.reshape(-1, 1), axis = 1)
        if var[col] == 'PM2.5' :
            lagsep = range(lag+1, xend-(qlag-lag-1))
            y = data[lagsep, col]
        nn_in = np.delete(nn_in, 0, 1)
        nn_in_3dim = np.concatenate([nn_in_3dim, nn_in.reshape(-1, 24, 1)], 2)
    nn_in_3dim = np.delete(nn_in_3dim, 0, 2)
    return nn_in_3dim, y


def countna(filename):
    root = '/home/data/chudawn/'
    xfiletype = '_x.npy'
    yfiletype = '_y.npy'
    xdata = np.load(root + filename + xfiletype, allow_pickle = True)
    ydata = np.load(root + filename + yfiletype, allow_pickle = True)
    print("x data shape : ", xdata.shape)

    data = xdata
    datanacount = np.zeros((data.shape[0], 1))

    a = ydata.reshape(-1, 1)      
    datanacount_y = np.count_nonzero((a != a), axis = 1) # count na
    print("datanacount_y : ", datanacount_y.shape)
    for i in range(data.shape[2]):
        nandata = data[:, :, i]
        nancolcount = np.zeros((data.shape[0], 1))
        for col in range(len(nandata)):
            a = nandata[col, :]
            nancolcount[col, 0] = np.count_nonzero(a != a) # count na 
    
        datanacount = np.append(datanacount, nancolcount, axis = 1)

    datanacount = np.delete(datanacount, 0, 1)
    #print("datanacount : ", datanacount.shape)
    data_na = np.append(datanacount, datanacount_y.reshape(-1, 1), axis = 1)
    #print("data_na : ", data_na.shape)
    finalna = np.sum( data_na, axis=1)
    checkna = (finalna == 0)
    #print("checkna : ", checkna.shape)

    finaldata_x = data[checkna, :, :]
    finaldata_y = ydata[checkna]

    #print("finaldata_x : ", finaldata_x.shape)
    #print("finaldata_y : ", finaldata_y.shape)


    return data, ydata, finalna


#############
#           #
# IMPLEMENT #
#           #
#############
place = ["zhongming", "dali", "shalu", "xitun", "fengyuan"]
filenum = len(place)
for i in range(filenum) :
    root = '/home/data/chudawn/'
    filetype = '.npy'

    data = np.load(root + place[i] + filetype, allow_pickle = True)
    #print("Place  :  ", place[i])
    dataclean_x, dataclean_y = clean_data(data)
    np.save( root + place[i] + '_xnorm' , dataclean_x) 
    #print(" save x : ", dataclean_x.shape)
    np.save( root + place[i] + '_ynorm' , dataclean_y) 
    #print("save y : ", dataclean_y.shape)

    x, y, check = countna(place[i])
    
    np.save( root + place[i] + '_xc' , x)
    #print("x : ", x.shape())
    np.save( root + place[i] + '_yc' , y)
    np.save( root + place[i] + '_check' , check)






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

