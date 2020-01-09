import logging

import numpy as np
import random
import ga
import time 

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter

# Import SMAC-utilities
from smac.scenario.scenario import Scenario
from smac.facade.smac_bo_facade import SMAC4BO


def gawd(x):

    num_generations = x["num_generations"]
    number_chromosomes = x["number_chromosomes"]
    num_parents_mating = x["num_parents_mating"]
    mutation_prob = x["mutation_prob"]

    """ Input data """
    number_bidders = 42;
    number_objects = 26;
    number_bids = 5;


    offers = [[40, [10, 21, 2, 20], 20],
    [5, [10, 21, 2, 20], 20],
    [7, [10, 21, 2, 20], 20],
    [16, [26, 7, 24, 19, 9, 5, 25, 8, 11, 12, 18, 6, 16, 17, 14], 14],
    [18, [26, 7, 24, 19, 9, 5, 25, 8, 11, 12, 18, 6, 16, 17, 14], 14]];


    """ Generate individuals """
    new_population = []
    for index_i in range(int(number_chromosomes)):
        current_individual = np.random.randint(2, size=number_bids)
        all_ones = [i for i, e in enumerate(current_individual) if e == 1]
        for el in all_ones:
            temp = []
            for jel in all_ones:
                if el != jel and len(list(set(offers[el][1]) & set(offers[jel][1]))) > 0:                                  
                    current_individual[jel] = 0             
                    temp.append(jel)
            for el in temp:
                all_ones.remove(el)                 
        new_population.append(current_individual)
    
    new_population = np.asarray(new_population)

    for generation in range(int(num_generations)):              
        """ Generate fitnesses """
        fitness = 0
        all_fitness = []
        for el in new_population:
            fitness = 0
            for index_i in range(len(el)):
                fitness += el[index_i]*offers[index_i][2]        
            all_fitness.append(fitness)

        fitness = all_fitness

        """ Selection mating parents """
        new_population = np.asarray(new_population)
        fitness = np.asarray(fitness)
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents = np.empty((num_parents_mating, new_population.shape[1]))
    
        for parent_num in range(int(num_parents_mating)):
            max_fitness_idx = np.where(fitness == np.max(fitness))
            max_fitness_idx = max_fitness_idx[0][0]
            parents[parent_num, :] = new_population[max_fitness_idx, :]
            fitness[max_fitness_idx] = -999 

        
        offspring_size=(int(number_chromosomes) - parents.shape[0], number_bids)
        parents = np.asarray(parents)
        offspring = np.empty(offspring_size) 
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

            # Validation that an object can not be provided two times
            all_ones = [i for i, e in enumerate(offspring[k]) if e == 1]
            for el in all_ones:
                temp = []
                for jel in all_ones:
                    if el != jel and len(list(set(offers[el][1]) & set(offers[jel][1]))) > 0:                                  
                        offspring[k][jel] = 0               
                        temp.append(jel)
                for el in temp:
                    all_ones.remove(el)                 

        offspring_crossover = offspring
        
        # Mutation changes a single gene in each offspring randomly.
        # For each individual in the current population
        for idx in range(offspring_crossover.shape[0]):
            current_individual = offspring_crossover[idx]       
            all_ones = [i for i, e in enumerate(current_individual) if e == 1]      
            odds = random.uniform(0, 1)     
            chances = 0
            ok = 0
            if odds >= mutation_prob:
                while ok == 0 and chances < 10:
                    ok = 1
                    chances += 1
                    x = random.randrange(0, number_bids)
                    for el in all_ones:
                        if len(list(set(offers[x][1]) & set(offers[el][1]))) > 0: 
                            ok = 0
           
            if ok == 1:
                if current_individual[x] == 1:
                    current_individual[x] = 0
                else:
                    current_individual[x] = 1

        offspring_mutation = offspring_crossover       

        new_population[0:parents.shape[0], :] = parents
        new_population[parents.shape[0]:, :] = offspring_mutation
        new_population = np.asarray(new_population)

        
    # Getting the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    fitness = 0
    all_fitness = []
    for el in new_population:
        fitness = 0
        for index_i in range(len(el)):
            fitness += el[index_i]*offers[index_i][2]
        
        all_fitness.append(fitness)
    fitness = all_fitness
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(fitness == np.max(fitness))
    
    return fitness[int(best_match_idx[0][0])]


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

# Build Configuration Space which defines all parameters and their ranges
cs = ConfigurationSpace()
num_generations = UniformFloatHyperparameter("num_generations", 50, 250, default_value=100)
number_chromosomes = UniformFloatHyperparameter("number_chromosomes", 10, 100, default_value=20)
num_parents_mating = CategoricalHyperparameter("num_parents_mating", [2, 4], default_value=4)
mutation_prob = UniformFloatHyperparameter("mutation_prob", 0.1, 0.9, default_value=0.5)
cs.add_hyperparameters([num_generations, number_chromosomes, num_parents_mating, mutation_prob])

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 10,   # max. number of function evaluations; for this example set to a low number
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = gawd(cs.get_default_configuration())

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4BO(scenario=scenario,
               rng=np.random.RandomState(42),
               tae_runner=gawd,
               )

smac.optimize()
