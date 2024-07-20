import argparse
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



class PCATransform:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.original_shape = None
        self.cache = {}

    def __call__(self, img, path):
        if self.original_shape is None:
            self.original_shape = img.shape

        if isinstance(img, torch.Tensor):
            img_np = img.numpy().reshape((-1, img.shape[1])).astype(np.float32)
        else:
            img_np = img.reshape((-1, img.shape[1])).astype(np.float32)

        if path in self.cache:
            return self.cache[path]

        img_transformed = self.pca.fit_transform(img_np)
        self.cache[path] = img_transformed
        return img_transformed

    def inverse_transform(self, img_compressed):
        if isinstance(img_compressed, torch.Tensor):
            img_compressed = img_compressed.numpy()

        img_reconstructed = self.pca.inverse_transform(img_compressed)
        return img_reconstructed.reshape(self.original_shape)


class RectangleDataset(Dataset):
    def __init__(self, data_dir, transform=None, use_cache=True):
        self.data_dir = data_dir
        self.transform = transform
        self.use_cache = use_cache
        self.data = self.load_data()
        self.image_cache = {}

    def load_data(self):
        with open(os.path.join(self.data_dir, 'rectangle_data.json'), 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        img_path = os.path.join(self.data_dir, item['filename'])

        if self.use_cache:
            if img_path not in self.image_cache:
                self.image_cache[img_path] = Image.open(img_path).convert('L')
            image = self.image_cache[img_path].copy()
        else:
            image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        main_rect = item['main_rectangle']
        small_rect = item['small_rectangle']

        labels = torch.tensor([
            main_rect['x'], main_rect['y'], main_rect['width'], main_rect['height'],
            small_rect['x'], small_rect['y'], small_rect['width'], small_rect['height']
        ], dtype=torch.float32)

        return image, labels, img_path


def get_rectangle_data_loader(data_dir, batch_size=32, shuffle=False, num_workers=4):
    transform_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]

    dataset = RectangleDataset(data_dir, transform=transforms.Compose(transform_list))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def main():
    parser = argparse.ArgumentParser(description="Rectangle Loader")
    parser.add_argument("-p", "--pca", help="Use PCA", action="store_true")
    parser.add_argument("-d", "--directory", required=True, help="Directory to load images from")
    parser.add_argument("-c", "--comps", type=int, default=5, help="Number of PCA components to use (default: 5)")

    args = parser.parse_args()

    pca_transform = None
    if args.pca:
        pca_transform = PCATransform(args.comps)

    data_loader = get_rectangle_data_loader(args.directory)

    if pca_transform is not None:
        sample_image, _, path = data_loader.dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        compressed = pca_transform(sample_image, path)


    print(f"Dataset size: {len(data_loader.dataset)}")

    for i, (images, labels, paths) in enumerate(data_loader):
        print(f"Batch {i+1}:")
        print(f"  Image shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  First item labels: {labels[0]}")
        print(f"  Path: {paths[0]}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        if pca_transform is not None:
            # Original image
            ax1.imshow(images[0].squeeze().numpy(), cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')

            compressed = pca_transform(images[0], paths[0])
            print(f"Compressed shape: {compressed.shape}")
            print(f"Compressed values: {compressed}")
            reconstructed_image = pca_transform.inverse_transform(compressed)
            ax2.imshow(reconstructed_image.squeeze(), cmap='gray')
            ax2.set_title('Reconstructed Image')
        else:
            inputs = images[0].view(-1).numpy()
            inputs = np.clip(inputs, 0, 1)
            print(f"{inputs=}")
            ax2.imshow(images[0].squeeze().numpy(), cmap='gray')
            ax2.set_title('Original Image (No PCA)')
        ax2.axis('off')

        plt.suptitle(f'Image: {paths[0]}')
        plt.tight_layout()
        plt.show()

        if i == 2:
            break


if __name__ == "__main__":
    main()
