from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load CIFAR-100 dataset

from gensim.models import KeyedVectors

# Load the pre-trained Word2Vec model
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)  # Loading only the first 500,000 words to save memory
# Get the list of super categories and their corresponding classes

super_categories_to_classes = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea']
}



# Initialize a 2D matrix for storing the similarities
similarity_matrix = np.zeros((len(super_categories_to_classes), len(super_categories_to_classes)))

# Compute the average vectors for each super category
avg_vectors = []
for category, items in super_categories_to_classes.items():
    avg_vector = np.zeros(word_vectors.vector_size)
    count = 0
    for item in items:
        if item in word_vectors:
            avg_vector += word_vectors[item]
            count += 1
    if count > 0:
        avg_vector /= count
    avg_vectors.append(avg_vector)

# Compute the semantic similarity matrix
for i in range(len(super_categories_to_classes)):
    for j in range(len(super_categories_to_classes)):
        similarity_matrix[i][j] = np.dot(avg_vectors[i], avg_vectors[j]) / (np.linalg.norm(avg_vectors[i]) * np.linalg.norm(avg_vectors[j]))

print(similarity_matrix)


# Save the distance matrix to a file
np.savetxt('word2vec_distance_matrix.txt', similarity_matrix)


# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=super_categories_to_classes.keys(),
            yticklabels=super_categories_to_classes.keys())
plt.xlabel('Super Categories')
plt.ylabel('Super Categories')
plt.title('Wasserstein Distance Matrix between Super Categories in CIFAR-100')
plt.show()
