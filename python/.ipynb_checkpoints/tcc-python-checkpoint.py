'''
    Modelo para análise e predição do valor máximo do ativo PETR4.
'''

import pandas as pandas
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.offline as py
import statsmodels.formula.api as sm
import time

from scipy import stats
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime

################### Análise dos Dados Dollar USDBRL=X.csv #######################
dataFrameDollar = pandas.read_csv ("../dataset/USDBRL=X.csv", sep=',')
print('oi')
print(dataFrameDollar["Close"].count())
print(dataFrameDollar["Open"].count())
dataFrameDollar.head()
dataFrameDollar.isnull().sum()
dataFrameDollar = dataFrameDollar.dropna()
dataFrameDollar.shape
dataFrameDollar.describe()


################### Análise dos Dados Crude Oil Sep 20 (CL=F) #######################
dataFrameOleoCru = pandas.read_csv ("../dataset/CL=F.csv", sep=',')
print('oi')
print(dataFrameOleoCru["Close"].count())
print(dataFrameOleoCru["Open"].count())
dataFrameOleoCru.head()
dataFrameOleoCru.isnull().sum()
dataFrameOleoCru = dataFrameOleoCru.dropna()
dataFrameOleoCru.shape
dataFrameOleoCru.describe()


################### Análise dos Dados ativo Petr4 #######################
dataFramePetr4 = pandas.read_csv ("../dataset/PETR4.SA.csv", sep=',')

dataFramePetr4["clClose"] = dataFrameOleoCru["Close"]

dataFramePetr4.head()
dataFramePetr4.tail()
dataFramePetr4.isnull().sum()
dataFramePetr4 = dataFramePetr4.dropna()
dataFramePetr4.shape
print(dataFramePetr4.describe())

dictionary_plot_values = [
                            ('Open',    dataFramePetr4.Date, dataFramePetr4.Open,   "PETR4 Open",      '#FFFF00', "Ativo Petr4 Open",      '2015-07-15', '2020-07-14'),
                            ('High',    dataFramePetr4.Date, dataFramePetr4.High,   "PETR4 High",      '#17BECF', "Ativo Petr4 High",      '2015-07-15', '2020-07-14'),
                            ('Close',   dataFramePetr4.Date,dataFramePetr4.Close,   "PETR4 Close",     '#ADFF2F', "Ativo Petr4 Close",     '2015-07-15', '2020-07-14'),
                            ('Low',     dataFramePetr4.Date,dataFramePetr4.Low,     "PETR4 Low",       '#FF1493', "Ativo Petr4 Low",       '2015-07-15', '2020-07-14'),
                            ('clClose', dataFramePetr4.Date,dataFramePetr4.clClose, "Preço do Barril", '#FF0000', "Preço do Barril Close", '2015-07-15', '2020-07-14')
                         ]

def plotDataSet(column_date, column_type, name_, color_code, title_name, datefrom, dateUntil):
   
    type_column = go.Scatter( x=column_date, y=column_type, name = name_, line = dict(color = color_code), opacity = 0.8)
    data = [type_column]
    layout = dict(title = title_name,title_x= 0.5,xaxis = dict(range = [datefrom, dateUntil]))
    fig = dict(data=data, layout=layout)
    py.iplot(fig, filename = name_)
   
def convertDateToTimeStamp(dateIn, dataFramePetr4):
   
    dataPetr4 = dateIn
    arrayDate = []
    arrayDateReplace = []
    cont = 0
    for date in dataPetr4:
        dt = datetime.strptime(date, '%Y-%m-%d')
        ts = time.mktime(dt.timetuple())
        arrayDate.insert(cont,ts)
        date = date.replace('-','')
        arrayDateReplace.insert(cont, date)
        cont+=1
       
    dataFramePetr4['timeStamp'] = arrayDate
    print(dataFramePetr4['timeStamp'])
    dataFramePetr4['dateReplace'] = arrayDateReplace
       
def candleSticks(dataFramePetr4):
    datasetUmAno = dataFramePetr4.head(180)
    data = go.Candlestick(x=datasetUmAno.Date, open=datasetUmAno.Open, high=datasetUmAno.High, low=datasetUmAno.Low, close=datasetUmAno.Close)
    data = [data]
    py.offline.iplot(data, filename='Candlestick Petr4')

def linearRegression(dataFramePetr4):
    regression = sm.ols(formula='clClose~Open+High+Low+Close', data=dataFramePetr4).fit()
    print(regression.summary())
    return regression

def areThereCorrelation(dataFramePetr4, dataFrameDollar):
    print('Low   :' + str(dataFramePetr4["Low"].corr(dataFrameDollar["Low"])))  
    print('Open  :' + str(dataFramePetr4["Open"].corr(dataFrameDollar["Open"])))  
    print('High  :' + str(dataFramePetr4["High"].corr(dataFrameDollar["High"])))
    print('Close :' + str(dataFramePetr4["Close"].corr(dataFrameDollar["Close"])))
   
    print('Open sec :' + str(dataFramePetr4["Open"].corr(dataFrameDollar["Close"])))  
    print('High sec :' + str(dataFramePetr4["High"].corr(dataFrameDollar["Low"])))


def correlationChart(x, y):
    #plt.scatter(x,y,color='b')
    plt.plot(x,y,zorder=1)
    plt.scatter(x,y,zorder=2)
    plt.xlabel('Petr4 preço Máximo')
    plt.ylabel('Preco Barril Fechamento')
    plt.axis([min(x),max(x),min(y),max(y)])
    plt.autoscale('False')
    plt.show()

def prepareToTestAndTrainingBases(dataFramePetr4):
    dataTrainingWithoutMax  = dataFramePetr4;
    dataTrainingWithoutMax  = dataTrainingWithoutMax.drop(columns=['High'])
    dataTrainingWithoutMax  = dataTrainingWithoutMax.drop(columns=['Date'])
    print(dataTrainingWithoutMax.describe())

    #Atribuindo apenas o valor máximo
    dataTrainingHigh = dataFramePetr4["High"]

    #Dividindo as bases em teste e treino
    X_dataTrainning, X_dataTest, y_dataTrainning, y_dataTest = train_test_split(dataTrainingWithoutMax, dataTrainingHigh,test_size=0.2, random_state=0)

    print(" ---------------- Treino ------------------ ")
    print(X_dataTrainning)
    print(y_dataTrainning)
    print("Teste")
    print(" ---------------- X_dataTest ------------------ ")
    print(y_dataTest)
    print(X_dataTest)
   
    #Regressão Linear
    linearRegressionModel = LinearRegression()
    linearRegressionModel.fit(X_dataTrainning, y_dataTrainning)
    print(linearRegressionModel.coef_)
   
    #Validação e acertividade do modelo
    RMSE = mean_squared_error(y_dataTest, linearRegressionModel.predict(X_dataTest))**0.5
    print("Acertividade do modelo : " + str(RMSE))
   
    X_predict = linearRegressionModel.predict(X_dataTest)
   
    plotDataSet(dataFramePetr4['Date'][:205], y_dataTest, 'Valores Reais', '#FF5733', 'Valores Reais','2015-07-15', '2016-07-15')
    plotDataSet(dataFramePetr4['Date'][:205], X_predict, 'Valores Preditos', '#3374FF', 'Valores preditos','2015-07-15', '2016-07-15')
   
    return X_predict

if __name__ == '__main__':
   
    #for plot_values in dictionary_plot_values:
        #plotDataSet(plot_values[1], plot_values[2], plot_values[3], plot_values[4], plot_values[5], plot_values[6], plot_values[7])
   
    #convertDateToTimeStamp(dataFramePetr4["Date"], dataFramePetr4)    
    #candleSticks(dataFramePetr4)    
    #candleSticks(dataFrameOleoCru)
    #linearRegression(dataFramePetr4)
    areThereCorrelation(dataFramePetr4,dataFrameDollar)
    areThereCorrelation(dataFramePetr4,dataFrameOleoCru)
    #correlationChart(dataFramePetr4.High[:365], dataFramePetr4.clClose[:365])
    #prepareToTestAndTrainingBases(dataFramePetr4)

