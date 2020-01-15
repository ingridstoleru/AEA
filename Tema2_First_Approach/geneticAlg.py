#The population size is the number of bidders - n
#Each individual would be made up of an array of 0 and 1. All possible subsets (1 if there is an accepted bid on it/ 0 if not) - excepting the empty subset
#So each individual will have the size: 2^m - 1.
#Fitness function for an individual: sum of (price[i]*x[i]) - where x[i] is 0 or 1.
#Selection -> Tournament selection
#Crossover -> for each gene, a random value between 0 and 1 is generated. If the value is less than 0.5, the gene is inherited from the first parent, otherwise from the second parent.
#Second offspring is created using inverse mapping.
#Mutation occures with a certain probability. (bit flip)
#Replacement

import random
import numpy
import ga
import time	
import statistics	

""" Aux data for algorithm """
num_generations = 100
best_outputs = []
number_chromosomes = 20
num_parents_mating = 4

""" Input data """
number_bidders = 42;
number_objects = 21;
number_bids = 4;

"""
offers = [[1, [1, 2, 3, 4, 5], 173],
          [1, [1, 2, 4, 5], 173],
          [1, [1, 2, 5], 59],
          [1, [1, 3, 4, 5], 130],
          [1, [1, 2, 3], 55],
          [2, [1, 2, 3, 4, 5], 300],
          [3, [1, 2, 3, 4, 5], 200],
          [2, [1, 2, 5], 161],
          [2, [3, 4, 5], 181],
          [4, [1, 4], 142],
          [4, [1, 4, 5], 223],
          [2, [3], 52]]; 
"""      

"""offers = [[40, [10, 21, 2, 20], 20],
[5, [10, 21, 2, 20], 20],
[7, [10, 21, 2, 20], 20],
[16, [26, 7, 24, 19, 9, 5, 25, 8, 11, 12, 18, 6, 16, 17, 14], 14],
[18, [26, 7, 24, 19, 9, 5, 25, 8, 11, 12, 18, 6, 16, 17, 14], 14]];"""

"""
offers = [[5, [24, 30, 40, 17, 8, 33, 31, 3, 10, 15, 48, 36, 34, 41, 2, 18, 25, 21, 46, 20, 49], 49],
[4, [17, 11, 42, 1, 23, 50, 36, 39, 3, 29, 27, 22, 38, 19, 20, 40, 41, 35, 8, 14], 14],
[2, [46, 49, 31, 39, 42, 33, 50, 17, 12, 6, 16, 9, 32, 41, 48, 22, 36, 23, 19, 43, 28, 40, 45, 34, 10, 37, 2, 47, 35, 4, 13, 7, 25, 5, 30, 21, 27, 15], 15],
[1, [46, 49, 31, 39, 42, 33, 50, 17, 12, 6, 16, 9, 32, 41, 48, 22, 36, 23, 19, 43, 28, 40, 45, 34, 10, 37, 2, 47, 35, 4, 13, 7, 25, 5, 30, 21, 27, 15], 15],
[3, [2, 26, 23, 45, 35, 20, 18, 3, 24, 42, 48, 38, 37, 7, 19, 34, 14, 21, 11, 15, 27, 4, 5, 17, 31, 12, 6, 8, 36], 36],
[3, [17, 11, 42, 1, 23, 50, 36, 39, 3, 29, 27, 22, 38, 19, 20, 40, 41, 35, 8, 14], 14],
[4, [17, 11, 42, 1, 23, 50, 36, 39, 3, 29, 27, 22, 38, 19, 20, 40, 41, 35, 8, 14], 14]];
"""

offers = [[6, [1, 17, 3, 4, 20, 15, 12, 9, 2, 19, 8, 7, 13, 10, 18, 16, 5, 21], 21],
[22, [1, 17, 3, 4, 20, 15, 12, 9, 2, 19, 8, 7, 13, 10, 18, 16, 5, 21], 21],
[24, [14, 17, 11, 15, 13, 5, 8, 2, 10, 4, 21, 18, 19, 12, 6, 20, 7, 16, 3, 1, 9], 9],
[21, [7, 20, 9, 15], 15]];

# Python program to illustrate the intersection 
# of two lists in most simple way 
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def individual_generation():		
	new_population = []
	for index_i in range(number_chromosomes):
		current_individual = numpy.random.randint(2, size=number_bids)		
		all_ones = [i for i, e in enumerate(current_individual) if e == 1]
		for el in all_ones:
			temp = []
			for jel in all_ones:
				if el != jel and len(intersection(offers[el][1], offers[jel][1])) > 0:									
					current_individual[jel] = 0				
					temp.append(jel)
			for el in temp:
				all_ones.remove(el)					
		new_population.append(current_individual)

	return new_population	



def select_mating_pool(pop, fitness, num_parents):	
	pop = numpy.asarray(pop)
	fitness = numpy.asarray(fitness)
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
	parents = numpy.empty((num_parents, pop.shape[1]))
	
	for parent_num in range(num_parents):
		max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
		max_fitness_idx = max_fitness_idx[0][0]
		parents[parent_num, :] = pop[max_fitness_idx, :]
		fitness[max_fitness_idx] = -999	
	
	return parents

def cal_pop_fitness(pop):     
	fitness = 0
	all_fitness = []
	for el in pop:
		fitness = 0
		for index_i in range(len(el)):
			fitness += el[index_i]*offers[index_i][2]
		
		all_fitness.append(fitness)

	return all_fitness

def crossover(parents, offspring_size):	
	parents = numpy.asarray(parents)
	offspring = numpy.empty(offspring_size)	
	# The point at which crossover takes place between two parents. Usually, it is at the center.
	crossover_point = numpy.uint8(offspring_size[1]/2)

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
				if el != jel and len(intersection(offers[el][1], offers[jel][1])) > 0:									
					offspring[k][jel] = 0				
					temp.append(jel)
			for el in temp:
				all_ones.remove(el)					

	return offspring
     

def mutation(offspring_crossover):	
	# Mutation changes a single gene in each offspring randomly.
	# For each individual in the current population
	for idx in range(offspring_crossover.shape[0]):
		current_individual = offspring_crossover[idx]		
		all_ones = [i for i, e in enumerate(current_individual) if e == 1]		
		odds = random.uniform(0, 1)		
		chances = 0
		ok = 0
		if odds >= 0.5:
			while ok == 0 and chances < 10:
				ok = 1
				chances += 1
				x = random.randrange(0, number_bids)
				for el in all_ones:
					if len(intersection(offers[x][1], offers[el][1])) > 0: 
						ok = 0
       
		if ok == 1:
			if current_individual[x] == 1:
				current_individual[x] = 0
			else:
				current_individual[x] = 1

	return offspring_crossover

def gawd():
	new_population = individual_generation()
	new_population = numpy.asarray(new_population)
	for generation in range(num_generations):		
		fitness = cal_pop_fitness(new_population)
		parents = select_mating_pool(new_population, fitness, num_parents_mating)
		offspring_crossover = crossover(parents, offspring_size=(number_chromosomes - parents.shape[0], number_bids))
		# Adding some variations to the offsrping using mutation.
		offspring_mutation = mutation(offspring_crossover)

		new_population[0:parents.shape[0], :] = parents
		new_population[parents.shape[0]:, :] = offspring_mutation
		new_population = numpy.asarray(new_population)

		# The best result in the current iteration.
		# print("Best result at iteration {0}: {1}".format(generation, numpy.max(fitness)))
		
	# Getting the best solution after iterating finishing all generations.
	# At first, the fitness is calculated for each solution in the final generation.
	fitness = cal_pop_fitness(new_population)
	# Then return the index of that solution corresponding to the best fitness.
	best_match_idx = numpy.where(fitness == numpy.max(fitness))


	print("Best solution : ", new_population[best_match_idx[0][0], :])
	print("Best solution fitness : ", fitness[best_match_idx[0][0]])
	return fitness[best_match_idx[0][0]]
 
if __name__ == '__main__':
    times = []
    fitnesses = []
    for el in range(1):
        start_time = time.time()
        x = gawd()
        print("--- %s seconds ---" % (time.time() - start_time))
        times.append((time.time() - start_time))
        fitnesses.append(x)

    print(min(times))
    print(max(times))
    print(statistics.mean(times))

    print(min(fitnesses))
    print(max(fitnesses))
    print(statistics.mean(fitnesses))
    print(statistics.stdev(fitnesses))
    print(fitnesses.count(max(fitnesses)))