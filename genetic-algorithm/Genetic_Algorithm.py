# Adaptation of the knapsack problem  https://deap.readthedocs.io/en/master/examples/ga_knapsack.html
# Import Libraries 
import random

import numpy
from random import randint

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

# Set initial values
IND_INIT_SIZE = 8
MAX_ITEM = 50
MAX_COST = 100
NBR_ITEMS = 20

# To assure reproductibility, the RNG seed is set prior to the items
# dict initialization. It is also seeded in main().
random.seed(128)

# Create individuals
# In this problem the fitness is the minimization of the first objective (the cost of the food)
# And the maximization of the second objective (protein percentage)
creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", set, fitness=creator.Fitness)

# Create the item dictionary
dog_food_names = ['Chicken & Duck Grain Free Dry Food - Lilys Kitchen', 'Salmon & Trout Dry Food for Senior Dogs - Lilys Kitchen',
                  'Venison & Duck Dry Food - Lilys Kitchen', 'Organic Chicken & Vegetable Dry Food - Lilys Kitchen',
                  'Breakfast Crunch - Lilys Kitchen', 'Chicken & Turkey Casserole - Lilys Kitchen', 'Cottage Pie - Lilys Kitchen',
                  'Lamb Hotpot - Lilys Kitchen', 'Chomp-Away Chicken Bites - Lilys Kitchen', 'The best ever beef mini burgers - Lilys Kitchen',
                  'Scrumptious Duck & Venison Sausages - Lilys Kitchen', 'Light Large Breed Adult Dog Food with Chicken - Hills Science Plan',
                  'Large Breed Adult Dog Food with Lamb & Rice - Hills Science Plan', 'Healthy Mobility Large Breed Adult Dog Food with Chicken - Hills Science Plan',
                  'Perfect Weight Large Breed Adult Dog Food with Chicken - Hills Science Plan', 'Large Breed Adult Dog Food with Chicken - Hills Science Plan',
                  'Adult Dog Food with Turkey - Hills Science Plan', 'Light Adult Dog Food - Hills Science Plan', 
                  'Maxi Adult - Royal Canin - Hills Science Plan', 'Adult Turkey and Rice - James Wellbeloved']

dog_food_cost = [18, 18, 18, 19, 27.34,
                 18.33, 14.84, 14.84, 84.82, 84.82,
                 84.82, 18.01, 7.68, 8.04, 8.33,
                 16.29, 12.9, 14.07, 11.24, 11.24]

dog_food_proteinpctg = [22, 25, 24, 22, 21,
                        10.5, 10.5, 10.4, 36, 43,
                        32, 24.4, 23.1, 20.3, 27.8,
                        22.8, 5.97, 6.35, 26, 22]

dog_food_vals = {}
for i in range(NBR_ITEMS):
    dog_food_vals[i] = (dog_food_cost[i], dog_food_proteinpctg[i])
    

toolbox = base.Toolbox() # for initializing population and individuals in it

# Attribute generator
toolbox.register("attr_item", random.randrange, NBR_ITEMS)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_item, IND_INIT_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define evaluation function
def evalFunc(individual):
    cost = 0.0
    protein_pctg = 0.0
    for dog_food_val in individual:
        cost += dog_food_vals[dog_food_val][0]
        protein_pctg += dog_food_vals[dog_food_val][1]
    if len(individual) > MAX_ITEM or cost > MAX_COST:
        return 10000, 0             # Ensure the cost excess are dominated
    return cost, protein_pctg

# Define crossover and mutation operators to be applied

# crossover, producing two children from two parents, could be that the first child 
# is the intersection of the two sets and the second child their absolute difference.
def cxSet(ind1, ind2):
    """Apply a crossover operation on input sets. The first child is the
    intersection of the two sets, the second child is the difference of the
    two sets.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2

# mutation operator could randomly add or remove an element from the set input individual.
def mutSet(individual):
    """Mutation that pops or add an element."""
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.randrange(NBR_ITEMS))
    return individual,

# Since it is a multi-objective problem, we have selected the NSGA-II selection scheme : selNSGA2()

toolbox.register("evaluate", evalFunc)
toolbox.register("mate", cxSet)
toolbox.register("mutate", mutSet)
toolbox.register("select", tools.selNSGA2)


# Finally apply the algorithm

number_gen = 50
individuals_nextgen = 50
child_per_gen = 100
crossover_prob = 0.6
mutation_prob = 0.2
    
population = toolbox.population(n=individuals_nextgen)
# hall of fame object that contains the best individuals
hof = tools.ParetoFront()

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)
 
algo = algorithms.eaMuPlusLambda(population, toolbox, individuals_nextgen, child_per_gen, crossover_prob,
                          mutation_prob, number_gen, stats, halloffame=hof)

best_dog_foods = []

def best_foods():
    final_pop = algo[0]
    individual_num = randint(0,50)
    final_individual = final_pop[individual_num]
    
    for dog_food_id in final_individual:
        best_dog_foods.append(dog_food_names[dog_food_id])
    dog_food_list_to_string = ', '.join(best_dog_foods)    
    dog_food = 'The best dog foods are: ' + dog_food_list_to_string + '.'
    return dog_food

        
