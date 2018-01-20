import math
import csv
import itertools
import random
import time
import numpy

class Travelling_Salesman:
  num_cities = None
  cities_list = None
  dist_table = None

  # constructor
  def __init__(self, num_cities):
      self.num_cities = num_cities
      self.cities_list, self.dist_table = self.read_file("european_cities.csv")


  #-----------------------GENERAL FUNCTIONS-----------------------#

  # reads data from file
  def read_file(self, file_name):
      with open(file_name) as csv_table:
          reader = csv.reader(csv_table, delimiter = ';')
          cities = reader.next()
          distances = []
          for row in reader:
              distances.append(row)

      for i in range(len(distances)):
          for j in range(len(distances)):
              distances[i][j] = float(distances[i][j])
      return cities, distances

  # calculates tour length of a given permutation of cities
  def calc_distance(self, permutation, dist_table):
      total = 0
      for i in range(len(permutation)):
          if i < len(permutation)-1:
              total += dist_table[permutation[i]][permutation[i+1]]
          else:
              total += dist_table[permutation[i]][0]
      return total

  # returns a stream with all possible permutations of cities
  def get_permutations(self):
      indices = [i for i in range(self.num_cities)] #list with ints from 0 to N
      permutations = itertools.permutations(indices)
      return permutations


  #-----------------------TASK 1: EXHAUSTIVE SEARCH-----------------------#

  # performs exhaustive search for the shortest tour
  def exhaustive_search(self):
      start_time = time.time()

      permutations = self.get_permutations()
      min_distance = float("inf") # starting point is infinity
      best_solution = None
      for current in permutations:
          total_distance = self.calc_distance(current, self.dist_table)
          if total_distance < min_distance:
              best_solution = current
              min_distance = total_distance
      tour = []
      for i in best_solution:
          tour.append(self.cities_list[i])
      print(("Order of cities: %s") % (tour))
      print(("Total distance: %0.f kilometers.") % (min_distance))

      print("Exhaustive search execution time: %.3f seconds. \n" % (time.time() - start_time))


  #-----------------------TASK 2: HILL CLIMBING-----------------------#

  # simple hill climbing algorithm: swaps two random cities in a permutation,
  # saves the solution if it is the best so far, stops if gets stuck for a
  # given number of iterations (~65,600 for 10 cities, ~238,000 for 24 cities)
  def hill_climbing(self):
      start_time = time.time()

      best_solution = [i for i in range(self.num_cities)]
      random.shuffle(best_solution)
      stuck_count = 0
      stuck_limit = math.factorial(self.num_cities-1)
      if stuck_limit > ((10**4)*math.log10(math.factorial(self.num_cities))):
          stuck_limit = (10**4)*math.log10(math.factorial(self.num_cities))
      min_distance = float("inf")
      while stuck_count < stuck_limit:
          current = best_solution
          rand_1 = random.randint(0,self.num_cities-1)
          rand_2 = random.randint(0,self.num_cities-1)
          #swap two random indices in the solution
          current[rand_1], current[rand_2] = current[rand_2], current[rand_1]
          if self.calc_distance(current, self.dist_table) < min_distance:
              best_solution = current
              min_distance = self.calc_distance(current, self.dist_table)
              stuck_count = 0
          else:
              stuck_count += 1
      tour = []
      for i in best_solution:
          tour.append(self.cities_list[i])
      print(("\nOrder of cities: %s") % (tour))
      print(("Total distance: %.0f kilometers.") % (min_distance))

      print("Hill-climbing algorithm execution time: %.3f seconds.\n" % (time.time() - start_time))
      return min_distance


  #-----------------------TASK 3: GENETIC ALGORITHM-----------------------#

  # creates a population of solutions with a given size
  def init_population(self, size):
      population = []
      for _ in range(size):
          indices = [i for i in range(self.num_cities)]
          random.shuffle(indices)
          population.append(indices)
      return population

  # calculates fitness of a given individual
  def calc_fitness(self, specimen):
      return self.num_cities*10**6/self.calc_distance(specimen, self.dist_table)

  # finds the minimum fitness in a given population
  def min_fitness(self, population):
      min_fit = float("inf")
      for specimen in population:
          fitness = self.calc_fitness(specimen)
          if fitness < min_fit:
              min_fit = fitness
      return min_fit

  # finds the shortest tour length in a given population
  def min_distance(self, population):
      shortest = float("inf")
      for specimen in population:
          distance = self.calc_distance(specimen, self.dist_table)
          if distance < shortest:
              shortest = distance
      return shortest

  # finds maximum fitness in a given population
  def max_fitness(self, population):
      max_fit = 0
      for specimen in population:
          fitness = self.calc_fitness(specimen)
          if fitness > max_fit:
              max_fit = fitness
      return max_fit

  # finds the longest tour length in a given population
  def max_distance(self, population):
      longest = 0
      for specimen in population:
          distance = self.calc_distance(specimen, self.dist_table)
          if distance > longest:
              longest = distance
      return longest

  # calculates the average fitness in a given population
  def avg_fitness(self, population):
      sum_fit = 0
      for specimen in population:
          sum_fit += self.calc_fitness(specimen)
      return sum_fit/len(population)

  # calculates the average tour length in a given population
  def avg_distance(self, population):
      sum_dist = 0
      for specimen in population:
          sum_dist += self.calc_distance(specimen, self.dist_table)
      return sum_dist/len(population)

  # calculates the standard deviation of fitnesses in a given population
  def deviation_fitness(self, population):
      fitnesses = []
      for specimen in population:
          fitness = self.calc_fitness(specimen)
          fitnesses.append(fitness)
      return numpy.std(fitnesses)

  # calculates the standard deviation of tour lengths in a given population
  def deviation_distance(self, population):
      lengths = []
      for specimen in population:
          length = self.calc_distance(specimen, self.dist_table)
          lengths.append(length)
      return numpy.std(lengths)

  # mating selection algorithm, returns a subset of a population
  def parent_selection(self, population):
      parents = []
      max_fit = self.max_fitness(population)
      for specimen in population:
          fitness = self.calc_fitness(specimen)
          selection_threshold = 1.0-(0.5*fitness/max_fit)
          if (random.random() > selection_threshold) or (fitness == max_fit):
              parents.append(specimen)
      return parents

  # partially mapped crossover
  def pm_crossover(self, parent1, parent2, start, stop):
    child = [None] * len(parent1)
    child[start:stop] = parent1[start:stop]
    for index, value in enumerate(parent2[start:stop]):
      i = index + start
      if value not in child:
          while child[i] != None:
              i = parent2.index(parent1[i])
          child[i] = value
    for index, value in enumerate(child):
        if value == None:
          child[index] = parent2[index]
    return child

  # calls pmx function with random parameters
  # to produce two children from two parents
  def get_offspring(self, parent1, parent2):
      xover_length = random.randint(1, self.num_cities//2+1)
      start = random.randint(0, self.num_cities - xover_length)
      stop = start + xover_length
      child1 = self.pm_crossover(parent1, parent2, start, stop)
      child2 = self.pm_crossover(parent2, parent1, start, stop)
      offspring = [child1, child2]
      return offspring

  # mutates (swaps to elements in) an individual with a given probability
  def mutate(self, specimen, probability):
      if (random.random() < probability):
          rand_1 = random.randint(0,self.num_cities-1)
          rand_2 = random.randint(0,self.num_cities-1)
          #swap two random indices in the permutation
          specimen[rand_1], specimen[rand_2] = specimen[rand_2], specimen[rand_1]
      return specimen

  # survivor selection algorithm
  def survivor_selection(self, population, limit):
    while(len(population) > limit):
      victim = random.randint(0, len(population)-1)
      fitness = self.calc_fitness(population[victim])
      max_fit = self.max_fitness(population)
      survival_threshold = 1.0 - 0.95*fitness/max_fit
      if (random.random() < survival_threshold) and (len(population) > limit):
        population.pop(victim)
    return population

  # emulates passage of one generation by calling the helper functions above
  def advance_generation(self, population):
      parents = self.parent_selection(population)
      children = []
      for i in range(0, len(parents)):
          partner = random.randint(0, len(parents)-(i+1))
          offspring = self.get_offspring(parents[i], parents[partner])
          for specimen in offspring:
              self.mutate(specimen, 0.33)
              children.append(specimen)
      #for specimen in population:
          #self.mutate(specimen, 0.01) # old individuals can mutate too
      generation = population + children
      self.survivor_selection(generation, len(population))
      return generation

  # creates a population of given size and follows its development until termination
  # terminates after 8*num_cities generations
  # returns the distance of the shortes tour and best fitnesses in each generation
  def genetic_algorithm(self, size):
      start_time = time.time()

      print("\nINITIALIZING POPULATION OF SIZE %d." % (size))
      pop = self.init_population(size)
      gen_count = 1
      fitnesses = [] # stores best fitness of each generation
      while(gen_count <= 8*self.num_cities):
          pop = self.advance_generation(pop)
          max_fit = self.max_fitness(pop)
          fitnesses.append(max_fit)
          print("Max fitness in generation %d: %d" % (gen_count, max_fit))
          gen_count += 1

      print("\nTOUR LENGTH STATISTICS")
      print("Shortest: %.0f kilometers" % (self.min_distance(pop)))
      print("Average: %.0f kilometers" % (self.avg_distance(pop)))
      print("Longest: %.0f kilometers" % (self.max_distance(pop)))
      print("Deviation: %.0f kilometers" % (self.deviation_distance(pop)))

      print("PROGRAM EXECUTION TIME: %.3f SECONDS\n" % (time.time() - start_time))
      return self.min_distance(pop), fitnesses


  #-----------------------EXTRA: FUNCTIONS FOR DATA GATHERING-----------------------#

  # collects various data from 20 runs of hill climbing algorithm
  def hill_data(self):
    lengths = []
    times = []
    print("Running the hillclimbing algorithm 20 times.")
    for i in range(20):
        start_time = time.time()
        lengths.append(self.hill_climbing())
        times.append(time.time() - start_time)
        print("%d/20 RUNS COMPLETE." % (i+1))
    print("\nThe shortest tour length: %.0f kilometers." % (min(lengths)))
    print("The average tour length: %.0f kilometers." % (numpy.mean(lengths)))
    print("The longest tour length: %.0f kilometers." % (max(lengths)))
    print("Standard deviation of tour lengths: %.0f kilometers." % (numpy.std(lengths)))
    print("Mean execution time: %.1f seconds." % (numpy.mean(times)))

  # collects various data from 20 runs of genetic algorithm
  # returns average fitnesses for each generation
  def genetic_data(self, size):
    lengths = [None]*20 # stores the shortest tour length in every run
    fitnesses = [None]*20 # stores the best fitness for each generations in every run
    avg_fitnesses = [] # stores the average best fitness for each generation across runs
    times = [] # stores the execution times
    print("Running the genetic algorithm 20 times.")
    for i in range(20):
      start_time = time.time()
      lengths[i], fitnesses[i] = self.genetic_algorithm(size) # runs the algorithm, stores data
      times.append(time.time() - start_time)
      print("%d/20 RUNS COMPLETE." % (i+1))
    fitnesses = numpy.asarray(fitnesses).T.tolist() # transposes the list "fitnesses"
    for gen_fitness in fitnesses:
        avg_fitnesses.append(numpy.mean(gen_fitness))
    print("\nThe shortest tour length: %d kilometers." % (min(lengths)))
    print("The average tour length: %d kilometers." % (numpy.mean(lengths)))
    print("The longest tour length: %d kilometers." % (max(lengths)))
    print("Standard deviation of tour lengths: %d kilometers." % (numpy.std(lengths)))
    print("Mean execution time: %.1f seconds." % (numpy.mean(times)))
    print("Avg best fitness in each generation:")
    return avg_fitnesses


#-------COMMAND LOOP/ MAIN--------#
def command_loop():
    print("This program presents various solutions of the Travelling Salesman problem.")
    num_cities = input("Enter the number of cities to run the program with (3-24): ")
    if num_cities > 3 and num_cities <= 24:
        num_cities = int(num_cities)
        salesman = Travelling_Salesman(num_cities)
        print("Program initialized with %d cities.\n" % (num_cities))

        while True:
            print("1) Exhaustive search.")
            print("2) Hill-climbing algorithm.")
            print("3) Genetic Algorithm.")
            option = input("Enter option (0 to exit): ")
            if option == 0:
                return
            elif option == 1:
                print("Executing exhaustive search.\n")
                salesman.exhaustive_search()
            elif option == 2:
                print("Executing hill-climbing algorithm.\n")
                salesman.hill_climbing()
            elif option == 3:
                print("Executing genetic algorithm.")
                pop_size = input("Enter population size: ")
                salesman.genetic_algorithm(pop_size)
            else:
                print("Invalid option.")
    else:
        print("Invalid number. Terminating program.")

if __name__ == "__main__":
    command_loop()
