import os
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE
from lightning.pytorch.callbacks import Callback
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import re

class TsneGeneratorCallback(Callback):
    def __init__(self, image_save_dir, latent_dim=100):
        super().__init__()
        self.image_save_dir = image_save_dir
        self.latent_dim = latent_dim
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            batch_size = 1000
            z = torch.randn(batch_size, self.latent_dim, device=pl_module.device)
            generated_images = pl_module.gen(z)
            generated_images = generated_images.view(generated_images.size(0), -1).cpu().numpy()

            tsne = TSNE(n_components=2, random_state=42)
            tsne_results = tsne.fit_transform(generated_images)

            plt.figure(figsize=(8, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=1, alpha=0.5)
            plt.title(f't-SNE of Generated Images - Epoch {trainer.current_epoch}')

            tsne_image_path = os.path.join(self.image_save_dir, f'tsne_epoch_{trainer.current_epoch}.png')
            plt.savefig(tsne_image_path)
            plt.close()

    def on_train_end(self, trainer, pl_module):
        image_files = sorted(os.listdir(self.image_save_dir), key=numerical_sort)
        image_paths = [os.path.join(self.image_save_dir, file_name) for file_name in image_files if file_name.endswith('.png')]

        create_gif(image_paths, self.image_save_dir, duration=200)
class TsneEncoderCallback(Callback):
    def __init__(self, image_save_dir, random_state=42, points=1000):
        super().__init__()
        self.image_save_dir = image_save_dir
        self.random_state = random_state
        self.points = points
        self.SAC = ['Sit', 'Stand', 'Walk', 'Upstairs', 'Downstairs', 'Run']
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            val_loader = trainer.datamodule.val_dataloader()
            all_samples = []
            all_labels = []
            for batch in val_loader:
                x, labels = batch  # Assuming batch = (data, labels)
                all_samples.append(x.cpu().numpy())  # Store numpy arrays
                all_labels.append(labels.cpu().numpy())  # Store numpy arrays
                if len(all_samples) * x.shape[0] >= self.points:  # Stop when reaching limit
                    break

            # Convert to numpy arrays
            all_samples = np.vstack(all_samples)  # Shape: (N, features)
            all_labels = np.hstack(all_labels)  # Shape: (N,)

            # Convert to tensors for model processing
            all_samples = torch.from_numpy(all_samples).to(pl_module.device)

            # Extract features using the discriminator's backbone
            features = pl_module.dis.backbone(all_samples).cpu().detach().numpy()

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=self.random_state)
            transformed = tsne.fit_transform(features)

            cmap = cm.get_cmap("viridis", len(self.SAC))
            norm = plt.Normalize(vmin=min(all_labels), vmax=max(all_labels))

            # Plot results
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=all_labels, cmap="viridis", alpha=0.7)
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title(f't-SNE of Generated Images - Epoch {trainer.current_epoch}')

            handles = [mpatches.Patch(color=cmap(norm(i)), label=label) for i, label in enumerate(self.SAC)]

            plt.legend(handles=handles, title="Class Labels", loc='lower right')

            # Save image
            tsne_image_path = os.path.join(self.image_save_dir, f'tsne_epoch_{trainer.current_epoch}.png')
            plt.savefig(tsne_image_path)
            plt.close()

    def on_train_end(self, trainer, pl_module):
        image_files = sorted(os.listdir(self.image_save_dir), key=numerical_sort)
        image_paths = [os.path.join(self.image_save_dir, file_name) for file_name in image_files if file_name.endswith('.png')]

        create_gif(image_paths, self.image_save_dir, duration=200)

        
def numerical_sort(value):
    """
    Custom sort function to sort filenames with numerical values correctly.
    """
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def create_gif(image_paths, output_gif_path, duration=500):
    images = [Image.open(image_path) for image_path in image_paths]
    # Save as GIF
    images[0].save(
        f'{output_gif_path}/tsne.gif',
        save_all=True,
        append_images=images[1:],            
        duration=duration,
        loop=0  # 0 means infinite loop
    )    
class KNNValidationCallback(Callback):
    def __init__(self, k: int=10, flatten: bool=False):
        super().__init__()
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.flatten = flatten

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loader = trainer.datamodule.val_dataloader()
        accuracy_list = []
        for batch in val_loader:
            accuracy_list.append(self.knn_validation(pl_module, batch))
        knn_accuracy = np.array(accuracy_list).mean()
        pl_module.log("knn_accuracy", knn_accuracy)

    def knn_validation(self, pl_module, batch):
        x, y = batch
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        features = pl_module.dis.backbone(x)
        
        if self.flatten:
            features = features.reshape(features.size(0), -1)
        
        self.knn.fit(features.cpu().detach().numpy(), y.cpu().numpy())
        predictions = self.knn.predict(features.cpu().detach().numpy())
        knn_accuracy = accuracy_score(y.cpu().numpy(), predictions)

        return knn_accuracy

class MyPrintingCallback(Callback):
    #Test callback just to do a start/end test for our training
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
