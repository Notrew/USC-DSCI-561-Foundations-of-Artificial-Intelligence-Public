#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import pandas as pd
import math
import copy
import time


# In[2]:


fpath="input.txt"
# fpath="./test4.txt"
f=open(fpath,"r")
f=f.readlines()
city_size=int(f[0].replace("\n",""))
cities={}

for i in range(len(f[1:])):
    xyz_list=(f[1:][i].replace("\n","").split(" "))    
    xyz=tuple(int(i) for i in xyz_list)
    cities[i]=xyz

# nodes=list(cities.keys())


# In[3]:


# cities


# ## initialization

# In[8]:


def initializePopulation(pop_size,city_size,dist_matrix):
    initial_pop = []
    for i in range(pop_size):
        path = generate_path(city_size,dist_matrix)
        initial_pop.append(path)
    return initial_pop

def generate_path(city_size,dist_matrix):
    path = []
    person = random.randint(0,city_size-1)
    path.append(person)
    pop_dist = 0
    for _ in range(0,city_size-1):
        k =0
        dist_tmp = float("inf")
        while k < city_size:
            if k not in path and dist_matrix[k][person] < dist_tmp:
                j = k
                dist_tmp = dist_matrix[k][person]
            k += 1
        path.append(j)
        person = j
        pop_dist += dist_tmp
    return path 

def cityInPath(path,city):
    for i in path:
        if i == city:
            return True
    return False

# def pathInPop(pop,path):
#     for i in pop:
#         if i==path:
#             return True
#     return False

def factorial(n):
    res=1
    for i in range(2,n+1):
        res*=i
    return res

def generate_tours(initial_pop):
    tours=copy.deepcopy(initial_pop)
    for tour in tours:
        start=tour[0]
        tour.append(start)
    return tours


# ## distance & fitness value

# In[5]:


def cal_dis_matrix(cities):
    cities_df=pd.DataFrame(cities)
    cities_v=np.array(cities_df).astype(int)
    cities_d=cities_v
    dist = np.zeros((cities_v.shape[1],cities_d.shape[1])) #distance matrix
    for i in range(cities_v.shape[1]):
        for j in range(cities_d.shape[1]):
            distance=math.sqrt(np.sum((cities_v[:,i]-cities_d[:,j])**2))
            if distance==0:
                distance=None
            dist[i,j]=distance
    return dist

def cal_tour_dis(tour,dist_matrix):
    distance=0
    for i in range(len(tour)-1):
        distance += dist_matrix[tour[i],tour[i+1]]
    return distance

def fitness(pop_size,ini_tours,dist_matrix):
    values=[]
    for i in range(pop_size):
        tour=ini_tours[i]
        value=cal_tour_dis(tour,dist_matrix)
        values.append(1000.0/value) #the lower the distance, the higher value for that tour
        if values[i]<0:
            values[i]=0
    return values


# ## roulette slection & crossover & mutation

# In[6]:


def roulette_slection(pop,pop_size,values):
    prob=[]
    for i in range(len(values)):
        prob.append(values[i]/sum(values)) 

    pop_new=[] #select new population
    lower_bound=1/pop_size/2
    higher_bound=1/pop_size
    for i in range(len(values)):
        rand=np.random.uniform(0, higher_bound)
        for j in range(len(values)): 
            if rand<=prob[j] and pop[j] not in pop_new: #if its possibility > the random number,take it
                pop_new.append(pop[j])
    
    return pop_new

def crossover(pop,crossover_rate):
    offspring = []
    cut_point = int(len(pop)/random.randint(2,4))
    father = pop[:cut_point]
    mother = pop[cut_point:]
    np.random.shuffle(father)
    np.random.shuffle(mother)
    son = []
    daughter = []

    for i in range(cut_point):
        if np.random.uniform(0,1) <= crossover_rate:
            cross_pos = np.random.randint(1,math.ceil(city_size/2))
            father_cros1 = father[i][:cross_pos]
            mother_cros1 = mother[i][cross_pos:]
            if len(set(father_cros1)&set(mother_cros1)) == 0:
                son = father_cros1+mother_cros1
                daughter = mother[i][:cross_pos]+father[i][cross_pos:]
        else:
            son = father[i]
            daughter = mother[i]
        
        if len(son)&len(daughter) != 0:
            offspring.append(son)
            offspring.append(daughter)
    
    return offspring

def mutation(offspring,mutation_rate):
    for i in range(len(offspring)):
        if np.random.uniform(0, 1) <= mutation_rate:
            mutation_pos1 = np.random.randint(0,len(offspring[i]))
            mutation_pos2 = np.random.randint(0,len(offspring[i]))
            offspring[i][mutation_pos1],offspring[i][mutation_pos2] = offspring[i][mutation_pos2],offspring[i][mutation_pos1]
    
    return offspring


# In[9]:


pop_size = 20
shortest_tours = []
t_dist=1000000000
#calculate distance matrix
dist=cal_dis_matrix(cities)
for i in range(100): #number of initial populations
    #initialize population
    initial_pop=initializePopulation(pop_size,city_size,dist)
    ini_tours=generate_tours(initial_pop)
    #
    candidates_pop = []
    crossover_rate = 0.65
    mutation_rate = 0.5
    #GA
    #select
    values = fitness(pop_size,ini_tours,dist)
#     pop_new = roulette_slection(initial_pop,pop_size,values)
    for j in range(30):
        #crossover
        c_offspring = crossover(initial_pop,crossover_rate)
        #mutation
        m_offspring = mutation(c_offspring,mutation_rate)
        #select shortest tour for this initial population
        all_pop = initial_pop+c_offspring+m_offspring
        candidates_pop += all_pop
    
    candidates_pop = [list(t) for t in set(tuple(_) for _ in candidates_pop)]
#     print(len(candidates_pop))    
    candidates_tour = generate_tours(candidates_pop)
    for t in candidates_tour:
        if cal_tour_dis(t,dist)<t_dist:
            t_dist = cal_tour_dis(t,dist)
            shortest_tour = t

    if shortest_tour not in shortest_tours:
        shortest_tours.append(shortest_tour)

#for t in shortest_tours:
for tour in shortest_tours:
    if cal_tour_dis(tour,dist)<=t_dist:
        t_dist = cal_tour_dis(tour,dist)
        res = tour


# In[10]:


#output file
f = open('output.txt','w')
for i in res:
    output=""
    for j in range(0,3):
        output += str(cities[i][j])
        if j != 2:
            output += " "
    print(output,file=f)
f.close()
