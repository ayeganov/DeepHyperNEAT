'''
Set of functions for reporting status of an evolutionary run for the rectangle center prediction task.
'''
from itertools import islice
from deep_hyperneat.util import iteritems, itervalues, iterkeys
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN
from deep_hyperneat.decode import decode
from rectangle_loader import PCATransform
import matplotlib.pyplot as plt
import numpy as np
import torch

# These should match your main script
image_width, image_height = 40, 40
sub_in_dims = [1, image_height * image_width]
sub_sh_dims = [1, 3]
sub_o_dims = 2
pca_comps = 5
pca = PCATransform(pca_comps)


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def report_output(pop, data_loader):
    '''
    Reports the output of the current champion for the rectangle center prediction task.

    pop -- population to be reported
    data_loader -- DataLoader containing the rectangle dataset
    '''
    genome = pop.best_genome
    cppn = CPPN.create(genome)
    substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims)
    
    print("\n=================================================")
    print("\tChampion Output at Generation: {}".format(pop.current_gen))
    print("=================================================")
    
    # Use a small subset of the data for reporting
    sample_data = list(islice(data_loader, 1))
    
    total_error = 0.0
    for images, labels, paths in sample_data:
        for image, label, path in zip(images, labels, paths):
#            compressed = pca(image, path).reshape(-1)
#            compressed = np.append(compressed, 1.0)
#            outputs = substrate.activate(compressed)
            inputs = image.view(-1).numpy()
            inputs = np.append(inputs, 1.0)
            outputs = substrate.activate(inputs)
            
            predicted_center = (
                int(outputs[0] * image_width),
                int(outputs[1] * image_height)
            )
            
            actual_center = (
                label[0].item() + label[2].item() / 2,
                label[1].item() + label[3].item() / 2
            )
            
            error = calculate_distance(predicted_center, actual_center)
            total_error += error
            print(f"Image: {path}")
            print(f"Predicted center: {predicted_center}")
            print(f"Actual center: {actual_center}")
            print(f"Error: {error:.2f}\n")
    
    avg_error = total_error / (len(sample_data) * data_loader.batch_size)
    print(f"Average Error: {avg_error:.2f}")

def report_fitness(pop):
    '''
    Report average, min, and max fitness of a population

    pop -- population to be reported
    '''
    avg_fitness = sum(genome.fitness for genome in itervalues(pop.population)) / pop.size
    
    print("\n=================================================")
    print(f"\t\tGeneration: {pop.current_gen}")
    print("=================================================")
    print("Best Fitness \t Avg Fitness \t Champion")
    print("============ \t =========== \t ========")
    print(f"{pop.best_genome.fitness:.2f} \t\t {avg_fitness:.2f} \t\t {pop.best_genome.key}")
    print("=================================================")
    print("Max Complexity \t Avg Complexity")
    print("============ \t =========== \t ========")
    print(f"{pop.max_complexity} \t\t {pop.avg_complexity}")


def report_species(species_set, generation):
    '''
    Reports species statistics

    species_set -- set containing the species
    generation  -- current generation
    '''
    print("\nSpecies Key \t Fitness Mean/Max \t Sp. Size")
    print("=========== \t ================ \t ========")
    for sid, species in species_set.species.items():
        try:
            fitness = species.fitness if hasattr(species, 'fitness') else 'N/A'
            max_fitness = species.max_fitness if hasattr(species, 'max_fitness') else 'N/A'
            size = len(species.members) if hasattr(species, 'members') else 'N/A'
            
            print(f"{sid} \t\t {fitness} / {max_fitness} \t\t {size}")
        except AttributeError as e:
            print(f"Error reporting species {sid}: {str(e)}")

def plot_fitness(x, y):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.ylabel("Fitness")
    plt.xlabel("Generation")
    plt.title("Fitness over Generations")
    plt.tight_layout()
    plt.savefig("reports/fitness_plot.png")
    plt.close()

def visualize_prediction(image, predicted_center, actual_center, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(image.squeeze(), cmap='gray')
    plt.plot(predicted_center[0], predicted_center[1], 'ro', markersize=10, label='Predicted')
    plt.plot(actual_center[0], actual_center[1], 'bo', markersize=10, label='Actual')
    plt.legend()
    plt.title("Rectangle Center Prediction")
    plt.savefig(filename)
    plt.close()

def report_best_individual(pop, data_loader, generation):
    '''
    Reports and visualizes the best individual's performance

    pop -- population to be reported
    data_loader -- DataLoader containing the rectangle dataset
    generation -- current generation number
    '''
    genome = pop.best_genome
    cppn = CPPN.create(genome)
    substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims)
    
    # Get a sample image
    sample_image, sample_label = next(iter(data_loader))
    sample_image = sample_image[0]
    sample_label = sample_label[0]
    
    inputs = sample_image.view(-1).numpy()
    outputs = substrate.activate(inputs)
    
    predicted_center = (
        int(outputs[0] * image_width),
        int(outputs[1] * image_height)
    )
    
    actual_center = (
        sample_label[0].item() + sample_label[2].item() / 2,
        sample_label[1].item() + sample_label[3].item() / 2
    )
    
    error = calculate_distance(predicted_center, actual_center)
    
    print(f"\nBest Individual at Generation {generation}")
    print(f"Fitness: {genome.fitness:.4f}")
    print(f"Predicted center: {predicted_center}")
    print(f"Actual center: {actual_center}")
    print(f"Error: {error:.2f}")
    
    visualize_prediction(sample_image, predicted_center, actual_center, f"reports/best_individual_gen_{generation}.png")

