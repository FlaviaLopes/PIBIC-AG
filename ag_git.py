# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import random

class ag(object):
    def __init__(self,popSize,cromoSize,maxGen,nExec):
        self.__popSize = popSize
        self.__cromoSize = cromoSize
        self.__maxGen = maxGen
        self.__nExec = nExec        
        self.__popTemp = np.array([])        
        self.__pop = np.array([])
        self.__minFeatures = np.array([])
        self.__avgFeatures = np.array([])
        self.__maxFeatures = np.array([])        
   
    def setPop(self):  
        for i in np.arange(self.__popSize):
            self.__pop = np.append(self.__pop, np.random.choice([0,1] , size = self.__cromoSize))
        self.__pop = self.__pop.reshape(self.__popSize,self.__cromoSize)
        #print(self.__pop)
    
    def crossUniforme(self):        
        while True:
            p1,p2 = np.random.randint(self.__popSize), np.random.randint(self.__popSize)
            #print(p1,p2)
            if p1 != p2:
                break
        filho1,filho2 = self.__pop[p1], self.__pop[p2]
        #print("antes do crossover")
        #print(filho1)
        #print(filho2)
        for i in np.arange(self.__cromoSize):
            if random.random() > 0.5:
                filho1[i], filho2[i] = filho2[i].copy(), filho1[i].copy() 
            else:
                filho1[i], filho2[i] = filho1[i].copy(), filho2[i].copy() 
        #print("depois do crossover")
        #print(filho1)
        #print(filho2)
        self.__popTemp = np.append(self.__popTemp, filho1)
        self.__popTemp = np.append(self.__popTemp, filho2)        
    
    def generation(self):
        for i in np.arange(self.__popSize // 2):
            self.crossUniforme()
            
    def transition(self):
        self.__popTemp = self.__popTemp.reshape(self.__popSize,self.__cromoSize)
        self.__pop = self.__popTemp.copy()
        self.__popTemp = np.array([])
        self.statistics()  
        
    def execution(self):
        for i in np.arange(self.__maxGen):            
            self.generation()
            self.transition()
    
    def trials(self):        
        for i in np.arange(self.__nExec):
            print('Trial %i' % i)
            self.execution()
                
    def statistics(self):        
        minFeatures, avgFeatures, maxFeatures = self.__cromoSize, 0, 0        
        count = np.array([])
        
        for i in np.arange(self.__popSize):            
            count = np.append(count,np.count_nonzero(self.__pop[i]))
            
        minFeatures, avgFeatures, maxFeatures = np.min(count), np.mean(count), np.max(count)
         
        self.__maxFeatures = np.append(self.__maxFeatures, maxFeatures)
        self.__minFeatures = np.append(self.__minFeatures, minFeatures)
        self.__avgFeatures = np.append(self.__avgFeatures, avgFeatures)
    
    def histograma(self):
        #CALCULA A MÉDIA DAS EXECUÇÕES PARA CADA ARRAY DE ESTATÍSTICAS
        self.__minFeatures = self.__minFeatures.reshape(self.__nExec,self.__maxGen).mean(axis=0)
        self.__avgFeatures = self.__avgFeatures.reshape(self.__nExec,self.__maxGen).mean(axis=0)
        self.__maxFeatures = self.__maxFeatures.reshape(self.__nExec,self.__maxGen).mean(axis=0)
                
        plt.plot(list(range(self.__maxGen)), self.__maxFeatures, linestyle='-', color='g', linewidth=1.0, label='Máximo')
        plt.plot(list(range(self.__maxGen)), self.__avgFeatures, linestyle='-', color='b', linewidth=1.0, label='Média')
        plt.plot(list(range(self.__maxGen)), self.__minFeatures, linestyle='-', color='r', linewidth=1.0, label='Mínimo')
        #plt.axis([0.0,float(self.__maxGen),float(self.__cromoSize),float(self.__cromoSize)])
        plt.axis([0.0,float(self.__maxGen),float(min(self.__minFeatures) - 1 ),float(max(self.__maxFeatures) + 1)])
        plt.title("Características Selecionadas")
        plt.grid(True)
        plt.rcParams['figure.figsize'] = (20,8)
        plt.xlabel("Gerações")
        plt.ylabel("Características")
        plt.legend()
        plt.show()
        
import timeit
#popSize,cromoSize,maxGen,nExec
ga = ag(500,500,5000,10)
ga.setPop()
ini = timeit.default_timer()
ga.trials()
fim = timeit.default_timer()
print('Duração: %f s' % (fim - ini))
ga.histograma()
