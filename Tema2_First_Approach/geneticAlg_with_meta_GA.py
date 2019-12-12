#The population size is the number of bidders - n
#Each individual would be made up of an array of 0 and 1. All possible subsets (1 if there is an accepted bid on it/ 0 if not) - excepting the empty subset
#So each individual will have the size: 2^m - 1.
#Fitness function for an individual: sum of (price[i]*x[i]) - where x[i] is 0 or 1.
#Selection -> Tournament selection
#Crossover -> for each gene, a random value between 0 and 1 is generated. If the value is less than 0.5, the gene is inherited from the first parent, otherwise from the second parent.
#Second offspring is created using inverse mapping.
#Mutation occures with a certain probability. (bit flip)
#Replacement

""" Aux data for algorithm """
best_outputs, number_chromosomes, num_parents_mating = [], 20, 4

""" Input data """
number_bidders, number_objects, number_bids = 42, 26, 5

""" Information for meta algorithm """
number_indiv_meta, number_bits_meta, num_generations_meta, prob_mutation_meta, prob_cross_meta = 20, 20, 20, 0.3, 0.2
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

offers = [[40, [10, 21, 2, 20], 20],
[5, [10, 21, 2, 20], 20],
[7, [10, 21, 2, 20], 20],
[16, [26, 7, 24, 19, 9, 5, 25, 8, 11, 12, 18, 6, 16, 17, 14], 14],
[18, [26, 7, 24, 19, 9, 5, 25, 8, 11, 12, 18, 6, 16, 17, 14], 14]];


# Python program to illustrate the intersection 
# of two lists in most simple way 
def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


# The meta algorithm will have 20 individuals, each of them represented on 20 bits (first 6 for mutation probability, next 6 for 
# crossover probability, last 8 for number of generations )
def individual_generation_meta():
	new_meta_pop = []
	for index_i in xrange(number_indiv_meta):
		current_individual = numpy.random.randint(2, size=number_bits_meta)	
		new_meta_pop.append(current_individual)

	return new_meta_pop

def select_parents_meta(fitness, num_parents):
	parents = []
	for i in xrange(num_parents):
		parents.append(fitness.pop()[0])			
	
	return parents

def gawd_meta():
	new_population = individual_generation_meta()
	new_population = numpy.asarray(new_population)
	for generation in xrange(num_generations_meta):
		all_fitnesses_indiv = []		
		for individual in new_population:
			prob_mutation, prob_crossover, generationss = 0, 0, 0
			prob_mutation_bits = individual[:6]
			prob_crossover_bits = individual[6:12]
			num_gen = individual[12:18]

			for i in range(6):
				prob_mutation += prob_mutation_bits[i] * pow(2, i)
				prob_crossover += prob_crossover_bits[i] * pow(2, i)
				generationss += num_gen[i] * pow(2, i)

			prob_mutation = prob_mutation/ 100
			prob_crossover = prob_crossover/ 100 	

			fitness = gawd(prob_mutation, prob_crossover, generationss)
			all_fitnesses_indiv.append([individual, fitness])
		
		""" Sort all after best fitness to choose parents """
		all_fitnesses_indiv.sort(key=lambda all_fitnesses_indiv: all_fitnesses_indiv[1])
		""" Select parents by elitism """
		parents = select_parents_meta(all_fitnesses_indiv, 4)
		
		crossover_individuals = []
		new = []
		for i in xrange(len(new_population) - len(parents)):
			new = []
			parent1 = random.choice([0,1,2,3]) 					
			parent2 = random.choice([0,1,2,3]) 					
			crossover_point = random.randint(0, len(parents)) 

			for i in range(crossover_point):
				new.append(parents[parent1][i])
			for i in range(crossover_point, len(parents[parent1])):
				new.append(parents[parent2][i])

			crossover_individuals.append(new)	
		
		for el in parents:
			crossover_individuals.append(el)

		for el in crossover_individuals:
			odds = random.uniform(0, 1)		
			if odds > prob_mutation_meta:
				k = random.randint(0, len(el)-1)
				if el[k] == 1:
					el[k] = 0
				else:
					el[k] = 1
			
	for el in crossover_individuals:
			prob_mutation, prob_crossover, generationss = 0, 0, 0
			prob_mutation_bits = individual[:6]
			prob_crossover_bits = individual[6:12]
			num_gen = individual[12:18]

			for i in range(6):
				prob_mutation += prob_mutation_bits[i] * pow(2, i)
				prob_crossover += prob_crossover_bits[i] * pow(2, i)
				generationss += num_gen[i] * pow(2, i)

			prob_mutation = prob_mutation/ 100
			prob_crossover = prob_crossover/ 100 	
				
			fitness = gawd(prob_mutation, prob_crossover, generationss)
			all_fitnesses_indiv.append([(prob_mutation, prob_crossover, generationss), fitness])
		
	all_fitnesses_indiv.sort(key=lambda all_fitnesses_indiv: all_fitnesses_indiv[1])

	print("Best elements: fitness {0}, prob_mutation: {1}, prob_crossover: {2}, number_generations: {3}".format(all_fitnesses_indiv[-1][1],
	all_fitnesses_indiv[-1][0][0], all_fitnesses_indiv[-1][0][1], all_fitnesses_indiv[-1][0][2]))

import numpy
from past.builtins import xrange

def individual_generation():		
	new_population = []
	for index_i in xrange(number_chromosomes):
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
	
	for parent_num in xrange(num_parents):
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
		for index_i in xrange(len(el)):
			fitness += el[index_i]*offers[index_i][2]
		
		all_fitness.append(fitness)

	return all_fitness

def crossover(parents, offspring_size):	
	parents = numpy.asarray(parents)
	offspring = numpy.empty(offspring_size)	
	# The point at which crossover takes place between two parents. Usually, it is at the center.
	crossover_point = numpy.uint8(offspring_size[1]/2)

	for k in xrange(offspring_size[0]):
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
     

import random

def mutation(offspring_crossover, prob_mutation):	
	# Mutation changes a single gene in each offspring randomly.
	# For each individual in the current population
	for idx in xrange(offspring_crossover.shape[0]):
		current_individual = offspring_crossover[idx]		
		all_ones = [i for i, e in enumerate(current_individual) if e == 1]		
		odds = random.uniform(0, 1)		
		chances = 0
		ok = 0
		if odds >= prob_mutation:
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

def gawd(prob_mutation, prob_cross, num_generations):
	new_population = individual_generation()
	new_population = numpy.asarray(new_population)
	for generation in xrange(num_generations):		
		fitness = cal_pop_fitness(new_population)
		parents = select_mating_pool(new_population, fitness, num_parents_mating)

		odds = random.uniform(0, 1)				
		if odds > prob_cross:
			offspring_crossover = crossover(parents, offspring_size=(number_chromosomes - parents.shape[0], number_bids))
			# Adding some variations to the offsrping using mutation.
			offspring_mutation = mutation(offspring_crossover, prob_mutation)
			new_population[0:parents.shape[0], :] = parents
			new_population[parents.shape[0]:, :] = offspring_mutation
			new_population = numpy.asarray(new_population)

		else:
			offspring_mutation = mutation(new_population, prob_mutation)
			new_population = offspring_mutation

		# The best result in the current iteration.
		# print("Best result at iteration {0}: {1}".format(generation, numpy.max(fitness)))
		
	# Getting the best solution after iterating finishing all generations.
	# At first, the fitness is calculated for each solution in the final generation.
	fitness = cal_pop_fitness(new_population)
	# Then return the index of that solution corresponding to the best fitness.
	best_match_idx = numpy.where(fitness == numpy.max(fitness))


	#print("Best solution : ", new_population[best_match_idx[0][0], :])
	#print("Best solution fitness : ", fitness[best_match_idx[0][0]])

	return fitness[best_match_idx[0][0]]
	
import time	
import statistics
if __name__ == '__main__':
    times = []
    fitnesses = []
    for el in range(50):
        start_time = time.time()
        x = gawd(0.51, 0.11, 28)
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
