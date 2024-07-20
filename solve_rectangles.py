import numpy as np
from deep_hyperneat.population import Population
from deep_hyperneat.phenomes import FeedForwardCPPN as CPPN
from deep_hyperneat.decode import decode
from deep_hyperneat.visualize import draw_net
from tqdm import tqdm
from rectangle_loader import get_rectangle_data_loader, PCATransform

# Substrate parameters
pca_comps = 5
image_width, image_height = 12, 12
sub_in_dims = [1, image_height * image_width]  # 1 x flattened image
sub_sh_dims = [1, 3]
sub_o_dims = 144  # 2 outputs (x, y) for the center of the large rectangle

# Evolutionary parameters
pop_key = 0
pop_size = 50
pop_elitism = 2
num_generations = 20
pca = PCATransform(pca_comps)


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def rectangle_proximity(genomes, data_loader):
    max_distance = np.sqrt(image_width**2 + image_height**2)

    # Progress bar for genomes
    for genome_key, genome in tqdm(genomes, desc="Evaluating genomes", leave=False):
        cppn = CPPN.create(genome)
        substrate = decode(cppn, sub_in_dims, sub_o_dims, sub_sh_dims)

        total_fitness = 0
        # Progress bar for dataset
        for images, labels, paths in tqdm(data_loader, desc=f"Genome {genome_key}", leave=False):
            for image, label, path in zip(images, labels, paths):
                # Flatten the image for substrate input
#                compressed = pca(image, path).reshape(-1)
#                compressed = np.append(compressed, 1.0)
                inputs = image.view(-1).numpy()
                inputs = np.append(inputs, 1.0)

                # Generate rectangle center using the substrate
                outputs = substrate.activate(inputs)

                # Interpret outputs as the center of the large rectangle
#                predicted_center = (
#                    int(outputs[0] * image_width),
#                    int(outputs[1] * image_height)
#                )

                max_idx = np.argmax(outputs)
                # Extract the center of the main rectangle from the label
                actual_center = (
                    label[0].item() + label[2].item() / 2,
                    label[1].item() + label[3].item() / 2
                )
                predicted_center = np.unravel_index(max_idx, (image_height, image_width))
                # Calculate fitness based on the distance between predicted and actual centers
                distance = calculate_distance(predicted_center, actual_center)
                fitness = 1 - (distance / max_distance)  # Normalize to [0, 1]

                total_fitness += fitness

        # Average fitness across all images in the dataset
        genome.fitness = total_fitness


# Load the dataset
data_dir = input("Enter the directory containing the image and JSON data: ")
data_loader = get_rectangle_data_loader(data_dir, batch_size=32, shuffle=True)

# Set goal fitness to 98%
goal_fitness = len(data_loader) * 32 * 0.97
print("Goal fitness set to 97% of the dataset size = ", goal_fitness)

# Initialize population
pop = Population(pop_key, pop_size, pop_elitism, data_loader)

# Run population on the defined task for the specified number of generations
def run_generation(genomes):
    rectangle_proximity(genomes, data_loader)

winner_genome = pop.run(run_generation, goal_fitness, num_generations)
winner_genome.save_to_json("reports/champion_genomes/rectangle_proximity_winner.json")

# Decode winner genome into CPPN representation
cppn = CPPN.create(winner_genome)

# Decode Substrate from CPPN
substrate = decode(cppn, sub_in_dims, sub_o_dims)

# Visualize networks of CPPN and Substrate
draw_net(cppn, filename="reports/champion_images/rectangle_proximity_cppn")
#draw_net(substrate, filename="reports/champion_images/rectangle_proximity_substrate")

# Run winning genome on the task again and display results
print("\nChampion Genome: {} with Fitness {}\n".format(winner_genome.key, winner_genome.fitness))
rectangle_proximity([(winner_genome.key, winner_genome)], data_loader)

# Generate and display the final rectangle center for a sample image
sample_image, sample_label, path = next(iter(data_loader))
sample_image = sample_image[0]
sample_label = sample_label[0]

#compressed = pca(sample_image, path).reshape(-1)
#compressed = np.append(compressed, 1.0)
inputs = sample_image.view(-1).numpy()
inputs = np.append(inputs, 1.0)
outputs = substrate.activate(inputs)

predicted_center = (
    int(outputs[0] * image_width),
    int(outputs[1] * image_height)
)

actual_center = (
    sample_label[0].item() + sample_label[2].item() / 2,
    sample_label[1].item() + sample_label[3].item() / 2
)

print("Final Rectangle Center Prediction for sample image:")
print("Predicted center:", predicted_center)
print("Actual center:", actual_center)

# Visualize the predicted and actual centers on the sample image
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.imshow(sample_image.squeeze(), cmap='gray')
ax.plot(predicted_center[0], predicted_center[1], 'ro', markersize=10, label='Predicted')
ax.plot(actual_center[0], actual_center[1], 'bo', markersize=10, label='Actual')
ax.legend()
plt.savefig("reports/champion_images/final_centers_on_sample.png")
plt.close()

print("Final centers visualization saved as 'reports/champion_images/final_centers_on_sample.png'")
