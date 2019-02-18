# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 13:30:47 2018

@author: mfrr
"""
import operator
import functools

from tqdm import tqdm
import pandas as pd
import warnings
import re

import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
import sklearn.utils
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import Memory
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import make_scorer 
import matplotlib.pyplot as plt
from imblearn.under_sampling import EditedNearestNeighbours,CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix

def passarJanela(df,tamanho):
    '''
        O index do DataFrame resultante será o index da primeira linha da janela.
    '''
    df = df.iloc[:len(df)-len(df)%tamanho,:] # deixo so a quantidade que e multiplo do tamanho da janela
    matrizes = [df.iloc[x::tamanho].values for x in range(tamanho)] #salvo as matrizes com o valores a serem somandos
    soma = functools.reduce(operator.add,matrizes) # efetuo a soma
    df = pd.DataFrame(soma,columns = df.columns,index = df.iloc[0::tamanho].index) # crio o novo DataFrame
    return df / tamanho # retorno o novo datafram dividido pelo tamanho da janela

def openCsv(path,index = None):
    df = None
    
    with open(path,'r') as csvFile:
        #read first line to get the sep=,
        matching = re.match(r'sep=(?P<SEP>.)[\n\r]+$',csvFile.readline())
        if matching is not None:
            sep = matching.group('SEP')
        else:
            csvFile.seek(0)
            warnings.warn('O arquivo \'{}\' não possui a definição de separador na primeira linha, assumindo que o separador é \';\'.'.format(path))
            sep = ';'
            
        if index is not None:
            df = pd.read_csv(csvFile, sep=sep,index_col = index ,error_bad_lines=True,dtype=str) #abri como string para nao ter erro com o tipo
        else:
            df = pd.read_csv(csvFile, sep=sep, index_col = False, error_bad_lines=True,dtype=str) #abri como string para nao ter erro com o tipo

    return df

def writeCsv(df,path,sep = ','):
    with open(path,'w') as csvFile:
        print('sep={}'.format(sep), file=csvFile)
    df.to_csv(path, sep = sep, index = True, mode = 'a' )

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def main():    

    # Pipeline
    pipeOrder = [
        ('scaler',MinMaxScaler()),
        ('selection',CondensedNearestNeighbour()), 
        ('reduce_dim',PCA()),
        ('classify',DecisionTreeClassifier())
    ]
    paramGrid = [
        {
            'scaler': [None],
            'selection':[None],
            'reduce_dim':[None],
            'classify': [DecisionTreeClassifier()],
            'classify__min_impurity_decrease': [0],
        },
        {
            'scaler': [None],
            'selection':[None],
            'reduce_dim':[None],
            'classify': [RandomForestClassifier()],
            'classify__n_estimators': [500],
        },
        {
            'scaler': [MinMaxScaler()],
            'selection':[None],
            'reduce_dim':[None],
            'classify': [MLPClassifier()],
            'classify__hidden_layer_sizes': [(neuronios,) for neuronios in range(10,100,10)] + [(neuronios,neuronios) for neuronios in range(10,100,10)],
            'classify__max_iter': [350], 
            'classify__activation': ['tanh', 'logistic'],
            'classify__solver': ['lbfgs', 'adam'],
        },
        {
            'scaler': [MinMaxScaler()],
            'selection':[None],
            'reduce_dim':[PCA(n_components = 0.9, svd_solver = 'full')],
            'classify': [MLPClassifier()],
            'classify__hidden_layer_sizes': [(neuronios,) for neuronios in range(10,100,10)] + [(neuronios,neuronios) for neuronios in range(10,100,10)],
            'classify__max_iter': [350], 
            'classify__activation': ['tanh', 'logistic'],
            'classify__solver': ['lbfgs', 'adam'],
        },
    ]
    
    pathDfBase = 'winequality.csv'    

    dfBase = openCsv(pathDfBase) #abro a base
    
    #A coluna type precisa ser transformada
    dfBase['type'].unique() #Ver valores da colunav (White and Red)
    dfBase.loc[dfBase['type'] == 'White', 'type'] = 1
    dfBase.loc[dfBase['type'] == 'Red', 'type'] = 2        

    dfTemp = pd.DataFrame()
    #Remover Valores com problema
    for index, row in dfBase.iterrows():
        try:
            dfTemp = dfTemp.append(row.astype(float))
        except:
            None
            print('Erro Classe: '+row['quality']+' valor:'+str(row))

    dfBase = dfTemp

    dfBase['quality'].value_counts()

    yLabelEncoder = LabelEncoder()
    y = dfBase.pop('quality').values # classes
    y = yLabelEncoder.fit_transform(y) # transformando classes (de string pra inteiro)
    X = dfBase.values # o que sobrou são os atributos
    X,y = sklearn.utils.shuffle(X,y) # embaralho
    cv = StratifiedKFold(n_splits = 10, shuffle = False)
                
    try:
        # Criando diretorio de caching
        cachedir = mkdtemp()
        memory = Memory(cachedir = cachedir, verbose = 0)
        # Computando o grid
        grid = GridSearchCV(Pipeline(pipeOrder, memory = memory),cv = cv, n_jobs = 10, param_grid = paramGrid, return_train_score = True, verbose = 1, scoring = 'f1_weighted')
        grid.fit(X,y)
                
        # Criando DataFrame para resultados
        dfMean = pd.DataFrame(grid.cv_results_ )
        # Salvando resultados
        print(f'Salvando resultados Wine_results.csv') 
        writeCsv(dfMean, 'Wine_results_2_0_f1_weighted.csv', sep = ',')
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        np.set_printoptions(precision=2)
        
        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title='Confusion matrix, without normalization')
        
        # Plot normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')
        
        plt.show()  
        
    except Exception as e:
        print (e)
    finally:
        rmtree(cachedir)

if __name__ == '__main__':
    main()
