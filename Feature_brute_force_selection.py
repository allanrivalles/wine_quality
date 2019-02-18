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

from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import EditedNearestNeighbours,CondensedNearestNeighbour
from imblearn.over_sampling import SMOTE
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate
import multiprocessing
from sklearn.multioutput import MultiOutputRegressor
import itertools
from joblib import Parallel, delayed

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

def main():    
    
    num_cores = multiprocessing.cpu_count()
    
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
            #print('Erro Classe: '+row['quality'])

    dfBase = dfTemp
    
    dfBase['quality'].value_counts()
    
    yLabelEncoder = LabelEncoder()
    y = dfBase.pop('quality').values # classes
    y = yLabelEncoder.fit_transform(y) # transformando classes (de string pra inteiro)
 
    dfResult = pd.DataFrame()
    
    for siz in range(2,8,1):
        results = Parallel(n_jobs=num_cores)(delayed(Predictor)(e, dfBase[list(e)], y) for e in tqdm(combinador(dfBase.columns, siz)))        
        dfResult = dfResult.append(results)

    print(f'Salvando resultados Wine_results.csv') 
    writeCsv(dfResult, 'Wine_featureselection_results_1_0_f1_score_weighted.csv', sep = ',')

def Predictor(e, df, y):
    
    X = df.values # o que sobrou são os atributos
    X,y = sklearn.utils.shuffle(X,y) # embaralho
    
    clf = RandomForestClassifier(n_estimators = 500)
    score = np.mean(cross_validate(clf, X, y, n_jobs=1, verbose=0,cv=5, scoring = 'f1_weighted')['test_score'])
    result = {'config':str(e),'score':score}
    return result

def combinador(cols, siz):
    for e in itertools.combinations(cols, siz):
        yield e

if __name__ == '__main__':
    main()
