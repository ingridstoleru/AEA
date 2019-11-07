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

""" Aux data for algorithm """
num_generations = 100
best_outputs = []
number_chromosomes = 20
num_parents_mating = 4

""" Input data """
number_bidders = 4;
number_objects = 5;
number_bids = 12;

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
	
if __name__ == '__main__':
	gawd()

