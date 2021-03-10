from functools import reduce
from operator import add
import random
from network import Network
import logging


class Optimizer:
    """Class that implements genetic algorithm for MLP optimization"""

    def __init__(self, param_choices, retain=0.3,
                 random_select=0.3, mutate_chance=0.5):
        """
        create an optimizer

        :param nn_param_choices: Possible network parameter
        :param retain: percentage of population to retain after
            each generation
        :param random_select: Probability of a rejected network
            remaining in the population
        :param mutate_chance: Probability a network will be randomly mutated
        """

        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.param_choices = param_choices
        self.mutated = []

    def create_population(self, count):
        """
        Create population of random network

        :param count: Number of networks to generate, aka the
            size of the population
        :return: population of network objects
        """

        pop = []
        for _ in range(0, count):
            network = Network(self.param_choices)
            network.create_random()

            # add the network to our population
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        return network.accuracy

    def grade(self, pop):
        """
        Find average fitness for a population

        :param pop:  The population of networks
        :return: the average accuracy of the population
        """
        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """
        Make two children as part of their parents

        :param mother: Network parameter
        :param father: Network parameter
        :return: Two network objects
        """
        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameter and pick params for the kid
            for param in self.param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object
            network = Network(self.param_choices)
            network.create_set(child)

            # Randomly mutate some of children
            if self.mutate_chance > random.random():
                network = self.mutate(network)
                self.mutated.append(network)
                logging.info("Network mutated")

            children.append(network)

        return children

    def mutate(self, network):
        cov_layer = []
        fully_layer = []

        mutation = random.choice(list(self.param_choices))

        if mutation == 'cov2d_layers':
            for _ in range(network.network['convolution']):
                x = random.choice(self.param_choices['cov2d_layers'])
                cov_layer.append(x)

            network.network['cov2d_layers'] = cov_layer

        elif mutation == 'convolution':
            network.network[mutation] = random.choice(self.param_choices[mutation])
            for _ in range(network.network['convolution'] - len(network.network['cov2d_layers'])):
                x = random.choice(self.param_choices['cov2d_layers'])
                network.network['cov2d_layers'].append(x)

        elif mutation == 'fc_layers':
            network.network[mutation] = random.choice(self.param_choices[mutation])
            for _ in range(network.network['fc_layers'] - len(network.network['neurons'])):
                x = random.choice(self.param_choices['neurons'])
                network.network['neurons'].append(x)

        elif mutation == 'neurons':
            for _ in range(network.network['fc_layers']):
                x = random.choice(self.param_choices['neurons'])
                fully_layer.append(x)

            network.network['cov2d_layers'] = fully_layer

        else:
            network.network[mutation] = random.choice(self.param_choices[mutation])

        return network

    def evolve(self, pop):
        """
        Evolve a population of network

        :param pop: A list of network parameters
        :return: The evolved population
        """
        # Get Score for each network
        graded = [(self.fitness(network), network) for network in pop]

        # sort on the scores
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # get the number we want to keep for the next gen
        retain_length = int(len(graded) * self.retain)

        # the parents are every network we want to keep
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # add children, which are bred from two remaining networks
        while len(children) < desired_length:

            # Get random parent
            male = random.randint(0, parents_length - 1)
            female = random.randint(0, parents_length - 1)

            # assuming they aren't the same network
            if male != female:
                male = parents[male]
                female = parents[female]

                # breed them
                babies = self.breed(male, female)

                # Add children one at a time
                for baby in babies:
                    if len(children) < desired_length:
                        if baby not in children and self.mutated and parents:
                            children.append(baby)

        parents.extend(children)

        return parents
