import math

def calculate_proximity_fitness(location1, location2, image_width, image_height):
    """
    Calculate fitness based on the proximity of two locations.
    
    Args:
    location1 (tuple): (x, y) coordinates of the first location
    location2 (tuple): (x, y) coordinates of the second location
    image_width (int): Width of the image
    image_height (int): Height of the image
    
    Returns:
    float: A fitness value between 0 and 1, where 1 indicates the locations are identical,
           and values approach 0 as the locations get farther apart.
    """
    x1, y1 = location1
    x2, y2 = location2
    
    # Calculate Euclidean distance
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Calculate maximum possible distance in the image
    max_distance = math.sqrt(image_width**2 + image_height**2)
    
    # Calculate fitness (1 for identical locations, approaching 0 for far apart locations)
    fitness = 1 - (distance / max_distance)
    
    return fitness


def calculate_rectangle_proximity_fitness(rect1, rect2, image_width, image_height):
    """
    Calculate fitness based on the proximity of two rectangles.
    
    Args:
    rect1 (dict): Dictionary containing 'x', 'y', 'width', and 'height' of the first rectangle
    rect2 (dict): Dictionary containing 'x', 'y', 'width', and 'height' of the second rectangle
    image_width (int): Width of the image
    image_height (int): Height of the image
    
    Returns:
    float: A fitness value between 0 and 1, where higher values indicate closer proximity
    """
    # Calculate center points of rectangles
    center1 = (rect1['x'] + rect1['width']/2, rect1['y'] + rect1['height']/2)
    center2 = (rect2['x'] + rect2['width']/2, rect2['y'] + rect2['height']/2)
    
    return calculate_proximity_fitness(center1, center2, image_width, image_height)


if __name__ == "__main__":
    # Example rectangle data
    rect1 = {'x': 10, 'y': 10, 'width': 20, 'height': 10}
    rect2 = {'x': 50, 'y': 50, 'width': 10, 'height': 5}
    
    # Image dimensions
    image_width, image_height = 100, 75
    
    # Calculate fitness
    fitness = calculate_rectangle_proximity_fitness(rect1, rect2, image_width, image_height)
    
    print(f"Fitness value: {fitness:.4f}")
    
    # Test with identical locations
    rect3 = {'x': 10, 'y': 10, 'width': 20, 'height': 10}
    fitness_identical = calculate_rectangle_proximity_fitness(rect1, rect3, image_width, image_height)
    print(f"Fitness value for identical rectangles: {fitness_identical:.4f}")
    
    # Test with far apart locations
    rect4 = {'x': 0, 'y': 0, 'width': 10, 'height': 5}
    rect5 = {'x': 90, 'y': 70, 'width': 10, 'height': 5}
    fitness_far = calculate_rectangle_proximity_fitness(rect4, rect5, image_width, image_height)
    print(f"Fitness value for far apart rectangles: {fitness_far:.4f}")
