# coding: utf-8
   
import timeit
import numpy as np
import random
import pandas as pd
#from sklearn import datasets, linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
#import os

class individual(object):
    def __init__(self):
        self.mse = None
        self.rss = None
        self.coefs = None
        self.features = []

class ag(object):
    def __init__(self,popSize,cromoSize,maxGen,nExec):
        self.__popSize = popSize
        self.__cromoSize = cromoSize
        self.__maxGen = maxGen
        self.__nExec = nExec
        self.__pop = []
        self.__popTemp = []
        self.__df = None
        self.__columns_train = None
        self.__columns_target = None
        self.loadDataSet()
        

    def loadDataSet(self):
        self.__df = pd.read_csv("student-performance.csv", delimiter=';')
        self.__columns_train = self.__df.columns.difference(['G3'])
        self.__columns_target = self.__df.columns[-1]
        x = self.__df[self.__columns_train]
        label = LabelEncoder()
        cat_Columns = x.dtypes.pipe(lambda x: x[x=='object']).index
        for col in cat_Columns:
            self.__df[col] = label.fit_transform(x[col])

    
    def setPop(self):
        for i in range(self.__popSize):
            ind = individual()
            ind.features = np.random.choice([0,1] , size = self.__cromoSize).tolist()
            ind = self.fitness(ind)
            self.__pop.append(ind)
            
    def fitness(self, ind):
        columns_train = []
        for i in range(self.__cromoSize):
            if ind.features[i] == 1:
                columns_train.append(self.__df.columns[i]) 
        x = self.__df[columns_train]
        y = self.__df[self.__columns_target]
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=5)
        reg = LinearRegression()
        
        reg.fit(x_train, y_train)
        y_pred = reg.predict(x_test)
        ind.mse = mean_squared_error(y_test, y_pred)
        ind.rss = ((reg.predict(x_test) - y_test) ** 2).sum()
        ind.coefs = reg.coef_
        return ind

    def mutate(self, filho):
        for i in range(self.__cromoSize):
            if random.random() < 0.05:
                if filho.features[i] == 0:
                    filho.features[i] = 1
                else:
                    filho.features[i] = 0
                
        return filho
    
    
    #Sorteia dois pais diferentes para crossover uniforme.
    #Gera dois filhos, aplica, ou não, mutação
    #Calcula fitness e os adiciona à população
    def crossUniforme(self):
        filho1 = individual()
        filho2 = individual()
        
        while True:
            p1 = np.random.randint(self.__popSize)
            p2 = np.random.randint(self.__popSize)
            if p1 != p2:
                break
                
        for i in range(self.__cromoSize):
            if random.random() > 0.5:
                filho1.features.append(self.__pop[p2].features[i])
                filho2.features.append(self.__pop[p1].features[i])
            else:
                filho1.features.append(self.__pop[p1].features[i])
                filho2.features.append(self.__pop[p2].features[i])
                
        if random.random() < 0.10:
            filho1 = self.mutate(filho1)
            filho2 = self.mutate(filho2)
       
        filho1 = self.fitness(filho1)
        filho2 = self.fitness(filho2)
        
        self.__pop.append(filho1)
        self.__pop.append(filho2)
           
    def generation(self):
        for j in range(self.__popSize // 2):
                self.crossUniforme()
     
    #Responsável por ajustes necessários entre as gerações
    #Faz o ranqueamento da população e preserva os indivíduos de melhor fitness 
    #Obs.: Durante a transição, após o ranqueamento o melhor indivíduo sempre estará na primeira posição         
    def transition(self):
        self.__pop = sorted(self.__pop, key = lambda ind: ind.mse)
        del self.__pop[self.__popSize - 1:self.__popSize * 2 - 1]
        print("Melhor solução: ", self.__pop[0].mse)
    
    #Responsável por executar o algoritmo por n gerações
    def execution(self):
        for i in range(self.__maxGen):            
            print("Geração ", i)
            self.generation()
            self.transition()
            
    #Responsável por executar o algoritmo por n execuções previamente indicadas, trials
    def trials(self):        
        for i in range(self.__nExec):
            #print('Trial %i' % i)
            self.execution() 
         
    #Apresenta as melhores soluções (primeiro indivíduo da população)
    #Em seguida verifica quais características foram selecionadas por ele e apresenta-as
    def bestSols(self):
        print("MSE: ", self.__pop[0].mse)
        print("RSS: ", self.__pop[0].rss)
        print("Coeficientes: ", self.__pop[0].coefs)
        columns_train = []
        for i in range(self.__cromoSize):
            if self.__pop[0].features[i] == 1:
                columns_train.append(self.__df.columns[i]) 
        print("Features: ", columns_train)
         


ga = ag(200,32,100,1)
ga.setPop()
ini = timeit.default_timer()
ga.trials()
fim = timeit.default_timer()
print('Duração: %f s' % (fim - ini))
ga.bestSols()

df = pd.read_csv("student-performance.csv", delimiter=';')
df.head()
