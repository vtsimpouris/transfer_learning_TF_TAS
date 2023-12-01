import torchvision
import torchvision.transforms as transforms
import numpy as np
import ot
import seaborn as sns
import matplotlib.pyplot as plt

# Load CIFAR-100 dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

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

# Initialize an empty Wasserstein distance matrix
num_categories = len(super_categories_to_classes)
wasserstein_distance_matrix = np.zeros((num_categories, num_categories))

# Compute Wasserstein distance for all pairs of super categories
for i, (super_category_1, classes_1) in enumerate(super_categories_to_classes.items()):
    for j, (super_category_2, classes_2) in enumerate(super_categories_to_classes.items()):
        if i != j:
            class_1_images = []
            class_2_images = []
            for k in range(len(trainset)):
                image, label = trainset[k]
                label_name = trainset.classes[label].lower()
                if label_name in [c.lower() for c in classes_1]:
                    class_1_images.append(np.array(image))
                elif label_name in [c.lower() for c in classes_2]:
                    class_2_images.append(np.array(image))

                if len(class_1_images) >= 2500 and len(class_2_images) >= 2500:
                    break

            if len(class_1_images) == 0 or len(class_2_images) == 0:
                print(f"No images found for {super_category_1} and {super_category_2}.")
                continue

            class_1_images_np = np.array(class_1_images)
            class_2_images_np = np.array(class_2_images)

            class_1_images_flat = class_1_images_np.reshape(class_1_images_np.shape[0], -1)
            class_2_images_flat = class_2_images_np.reshape(class_2_images_np.shape[0], -1)

            M = ot.dist(class_1_images_flat, class_2_images_flat)
            P = ot.emd2([], [], M)
            wasserstein_distance = np.sum(P * M)
            print(wasserstein_distance)
            wasserstein_distance_matrix[i, j] = wasserstein_distance

np.savetxt('poc/wasserstein_distances.txt', wasserstein_distance_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(wasserstein_distance_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
            xticklabels=super_categories_to_classes.keys(),
            yticklabels=super_categories_to_classes.keys())
plt.xlabel('Super Categories')
plt.ylabel('Super Categories')
plt.title('Wasserstein Distance Matrix between Super Categories in CIFAR-100')
plt.show()
