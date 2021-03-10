import random
import logging
from train import train_and_score

class Network:
    """
    Represent a network
    """

    def __init__(self, param_choices=None):
        """
        Initialize the network
        :param param_choices: Parameter of the network
        """

        self.accuracy = 0
        self.param_choices = param_choices
        self.network = {}
        self.cov_layer = []
        self.fully_layer = []

    def create_random(self):
        for key in self.param_choices:
            self.network[key] = random.choice(self.param_choices[key])

        for _ in range(self.network['convolution']):
            x = random.choice(self.param_choices['cov2d_layers'])
            self.cov_layer.append(x)

        for _ in range(self.network['fc_layers']):
            j = random.choice(self.param_choices['neurons'])
            self.fully_layer.append(j)

        self.network['cov2d_layers'] = self.cov_layer
        self.network['neurons'] = self.fully_layer

    def create_set(self, network):
        self.network = network

    def train(self):
        """
        Train the network and record the accuracy.
        :return:
        """

        if self.accuracy == 0.:
            self.accuracy = train_and_score(self.network)

    def print_network(self):
        """Print out a network"""

        logging.info(self.network)
        logging.info("Network accuracy: %.2f%%" % (self.accuracy * 100))