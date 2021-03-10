import logging
from optimizer import Optimizer
from tqdm import tqdm

# setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m%d%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='log5.txt'
)


def train_networks(networks):
    """
    Train each network

    :param networks: Current population of network
    """
    pbar = tqdm(total=len(networks))
    for network in networks:
        network.train()
        pbar.update(1)
    pbar.close()


def get_average_accuracy(networks):
    """
    Get accuracy for a group of network
    :param networks: list of networks
    :return:
        float : the accuracy of a population of networks
    """
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy

    return total_accuracy / len(networks)


def generate(generations, population, param_choices):
    """
    Generate a network with the genetic algorithm
    :param generations: Number times to evolve the population
    :param population: Number of network in each generation
    :param param_choices: parameter vhoices for networks
    :return:
    """
    optimizer = Optimizer(param_choices)
    networks = optimizer.create_population(population)

    # evolve the generation
    for i in range(generations):
        logging.info("***Doing generation %d of %d***" %
                     (i + 1, generations))

        # train and get accuracy for networks.
        train_networks(networks)
        print_networks(networks)

        # get the average accuracy for this generation
        average_accuracy = get_average_accuracy(networks)

        # print out the average accuracy for each generation
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-' * 80)

        # evolve, except on the last iteration
        if i != generations - 1:
            networks = optimizer.evolve(networks)

    # Sort our final population
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks.
    print_networks(networks[:5])


def print_networks(networks):
    """
    Print a list of networks

    :param networks: The population of network
    :return:
    """
    logging.info('-' * 80)
    for network in networks:
        network.print_network()


def main():
    """Evolve a network"""
    generations = 16
    population = 10

    param_choices = {
        'convolution': [1, 2, 3, 4],
        'cov2d_layers': [32, 64, 128, 256, 512],
        'fc_layers': [1, 2, 3],
        'neurons': [64, 128, 256, 512, 1024],
        'optimizer': ['rmsprop', 'adam', 'nadam', 'sgd']
    }

    logging.info("***Evolving %d generations with population %d***" %
                 (generations, population))

    generate(generations, population, param_choices)


if __name__ == '__main__':
    main()
