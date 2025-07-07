import os
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
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

###################################################################################################################333

    def on_train_epoch_end(self, trainer, pl_module):





            # Create custom legend with assigned colors
            handles = [mpatches.Patch(color=self.colors[i], label=self.SAC[i]) for i in range(len(self.SAC))]
            plt.legend(handles=handles, title="Class Labels", loc='lower right')

            # Save image
            tsne_image_path = os.path.join(self.image_save_dir, f'tsne_epoch_{trainer.current_epoch}.png')
            plt.savefig(tsne_image_path)
            plt.close()
class TsneEncoderCallback(Callback):
    def __init__(self, image_save_dir, random_state=42, points=1000, flatten=False):
        super().__init__()
        self.image_save_dir = image_save_dir
        self.random_state = random_state
        self.points = points
        self.flatten = flatten
        self.SAC = ['Sit', 'Stand', 'Walk', 'Upstairs', 'Downstairs', 'Run']
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

        self.colors = ['red', 'blue', 'green', 'orange', 'pink', 'brown']
        self.cmap = ListedColormap(self.colors)

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

            if self.flatten:
                features = features.reshape(features.shape[0], -1)

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=self.random_state)
            transformed = tsne.fit_transform(features)

            # Get unique labels and assign index-based colors
            unique_labels = np.unique(all_labels)
            label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
            color_indices = np.array([label_to_index[label] for label in all_labels])

            # Plot results
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(transformed[:, 0], transformed[:, 1], c=color_indices, cmap=self.cmap, alpha=0.7)
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.title(f't-SNE of Generated Images - Epoch {trainer.current_epoch}')

            # Create custom legend with assigned colors
            handles = [mpatches.Patch(color=self.colors[i], label=self.SAC[i]) for i in range(len(self.SAC))]
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
    def __init__(self, k: int = 5, flatten: bool = False, feature_extractor_path: str = "dis.backbone"):
        """
        Args:
            k (int): Número de vizinhos no KNN.
            flatten (bool): Se deve achatar as features.
            feature_extractor_path (str): Caminho para acessar o extrator de features a partir do pl_module,
                                          por exemplo: "dis.backbone", "backbone", "encoder", etc.
        """

        super().__init__()
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.flatten = flatten
        self.feature_extractor_path = feature_extractor_path

    def on_fit_start(self, trainer, pl_module):
        self._evaluate_knn(trainer, pl_module, prefix="knn")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._evaluate_knn(trainer, pl_module, prefix="knn")

    def _evaluate_knn(self, trainer, pl_module, prefix="knn"):
        val_loader = trainer.datamodule.val_dataloader()
        accuracy_list = []
        for batch in val_loader:
            accuracy_list.append(self.knn_validation(pl_module, batch))
        knn_accuracy = np.array(accuracy_list).mean()
        pl_module.log(f"{prefix}_accuracy", knn_accuracy, prog_bar=True)

    def knn_validation(self, pl_module, batch):
        x, y = batch
        x, y = x.to(pl_module.device), y.to(pl_module.device)
        
        # Obtem o extrator de features a partir do caminho especificado
        feature_extractor = self._get_feature_extractor(pl_module)
        features = feature_extractor(x)

        if self.flatten:
            features = features.reshape(features.size(0), -1)

        features_np = features.cpu().detach().numpy()
        labels_np = y.cpu().numpy()

        self.knn.fit(features_np, labels_np)
        predictions = self.knn.predict(features_np)
        knn_accuracy = accuracy_score(labels_np, predictions)

        return knn_accuracy

    def _get_feature_extractor(self, pl_module):
        """Acessa o extrator de features a partir de um caminho string como 'dis.backbone'."""
        obj = pl_module
        for attr in self.feature_extractor_path.split("."):
            obj = getattr(obj, attr)
        return obj


class MyPrintingCallback(Callback):
    #Test callback just to do a start/end test for our training
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")
