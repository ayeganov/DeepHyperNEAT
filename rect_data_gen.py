import argparse
import json
import os
import random

import cv2
import numpy as np

def draw_rectangle(image, location, width, height, color=(255, 255, 255), thickness=-1):
    x, y = location
    cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)

def get_user_input():
    key = cv2.waitKey(0)
    if key == ord('w'):
        return 'up'
    elif key == ord('s'):
        return 'down'
    elif key == ord('a'):
        return 'left'
    elif key == ord('d'):
        return 'right'
    elif key == ord('q'):
        return 'quit'
    else:
        return None

def move_rectangle(image, location, width, height, direction):
    x, y = location
    if direction == 'up':
        y = max(0, y - 1)
    elif direction == 'down':
        y = min(image.shape[0] - height, y + 1)
    elif direction == 'left':
        x = max(0, x - 1)
    elif direction == 'right':
        x = min(image.shape[1] - width, x + 1)
    return (x, y)

def random_walk(image, location, width, height, steps):
    path = [location]
    x, y = location
    
    for _ in range(steps):
        direction = random.choice(['up', 'down', 'left', 'right'])
        x, y = move_rectangle(image, (x, y), width, height, direction)
        path.append((x, y))
    
    return path

def get_non_overlapping_location(image, main_rect, width, height):
    while True:
        x = random.randint(0, image.shape[1] - width)
        y = random.randint(0, image.shape[0] - height)
        if not rectangles_overlap(main_rect, (x, y, width, height)):
            return (x, y)

def rectangles_overlap(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

def save_grayscale_image(gray_image, directory, step):
    filename = f"step_{step:03d}.png"
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, gray_image)
    return filename

def save_rectangle_data(data, directory):
    filepath = os.path.join(directory, "rectangle_data.json")
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def run_random_walk(image, rect_location, rect_width, rect_height, small_rect_width, small_rect_height, steps, save_directory):
    walk_path = random_walk(image, rect_location, rect_width, rect_height, steps)
    rectangle_data = []

    for step, location in enumerate(walk_path):
        image.fill(0)
        draw_rectangle(image, location, rect_width, rect_height, color=(255, 255, 255))
        small_rect_location = get_non_overlapping_location(image, (*location, rect_width, rect_height), small_rect_width, small_rect_height)
        draw_rectangle(image, small_rect_location, small_rect_width, small_rect_height, color=(255, 255, 255))

        filename = save_grayscale_image(image, save_directory, step)

        rectangle_data.append({
            "filename": filename,
            "main_rectangle": {
                "x": location[0],
                "y": location[1],
                "width": rect_width,
                "height": rect_height
            },
            "small_rectangle": {
                "x": small_rect_location[0],
                "y": small_rect_location[1],
                "width": small_rect_width,
                "height": small_rect_height
            }
        })

        cv2.imshow('Non-overlapping Rectangles', image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    # Ensure the last frame stays visible for a moment
    cv2.waitKey(500)

    return rectangle_data, walk_path[-1]


def run_manual_movement(image, rect_location, rect_width, rect_height, small_rect_width, small_rect_height, save_directory, start_step=0):
    rectangle_data = []
    step = start_step

    # Display initial image
    image.fill(0)
    draw_rectangle(image, rect_location, rect_width, rect_height, color=(255, 255, 255))
    small_rect_location = get_non_overlapping_location(image, (*rect_location, rect_width, rect_height), small_rect_width, small_rect_height)
    draw_rectangle(image, small_rect_location, small_rect_width, small_rect_height, color=(255, 255, 255))
    cv2.imshow('Non-overlapping Rectangles', image)

    while True:
        # Display the image and wait for user input
        cv2.imshow('Non-overlapping Rectangles', image)
        direction = get_user_input()

        if direction == 'quit':
            break
        elif direction is not None:
            step += 1
            rect_location = list(move_rectangle(image, rect_location, rect_width, rect_height, direction))

            image.fill(0)
            draw_rectangle(image, rect_location, rect_width, rect_height, color=(255, 255, 255))
            small_rect_location = get_non_overlapping_location(image, (*rect_location, rect_width, rect_height), small_rect_width, small_rect_height)
            draw_rectangle(image, small_rect_location, small_rect_width, small_rect_height, color=(255, 255, 255))

            filename = save_grayscale_image(image, save_directory, step)

            rectangle_data.append({
                "filename": filename,
                "main_rectangle": {
                    "x": rect_location[0],
                    "y": rect_location[1],
                    "width": rect_width,
                    "height": rect_height
                },
                "small_rectangle": {
                    "x": small_rect_location[0],
                    "y": small_rect_location[1],
                    "width": small_rect_width,
                    "height": small_rect_height
                }
            })

    return rectangle_data


def main():
    parser = argparse.ArgumentParser(description="Rectangle Movement Simulation")
    parser.add_argument("--mode", choices=["both", "random", "manual"], default="both", help="Simulation mode")
    parser.add_argument("--directory", required=True, help="Directory to save images and data")
    parser.add_argument("--steps", type=int, default=50, help="Number of steps for random walk")
    parser.add_argument("--resolution", default="60x60", help="Image resolution in format 'AAAxBBB'")
    parser.add_argument("--rect_size", default="15x10", help="Main rectangle size in format 'AAAxBBB'")

    args = parser.parse_args()

    os.makedirs(args.directory, exist_ok=True)

    width, height = map(int, args.resolution.split('x'))
    image = np.zeros((height, width, 1), dtype=np.uint8)

    rect_width, rect_height = map(int, args.rect_size.split('x'))
    rect_location = [width // 2, height // 2]  # Start at the center

    small_rect_width = int(rect_width * 0.6)
    small_rect_height = int(rect_height * 0.6)

    rectangle_data = []

    if args.mode in ["both", "random"]:
        random_data, last_location = run_random_walk(image, rect_location, rect_width, rect_height, 
                                                     small_rect_width, small_rect_height, args.steps, args.directory)
        rectangle_data.extend(random_data)
        rect_location = last_location

    if args.mode in ["both", "manual"]:
        if args.mode == "both":
            print("Random walk completed. Use 'w', 'a', 's', 'd' to move, 'q' to quit.")
        manual_data = run_manual_movement(image, rect_location, rect_width, rect_height, 
                                          small_rect_width, small_rect_height, args.directory, 
                                          start_step=len(rectangle_data))
        rectangle_data.extend(manual_data)

    save_rectangle_data(rectangle_data, args.directory)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
