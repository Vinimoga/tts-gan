from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from minerva.models.ssl.cpc import CPC
from minerva.models.nets.cpc_networks import HARCPCAutoregressive
from minerva.data.data_modules.har_rodrigues_24 import HARDataModuleCPC
from minerva.pipelines.lightning_pipeline import SimpleLightningPipeline
from minerva.models.nets.base import SimpleSupervisedModel

from minerva.models.nets.tnc import TSEncoder
import torchmetrics

from minerva.data.data_modules.har import MultiModalHARSeriesDataModule
from minerva.models.loaders import FromPretrained
from minerva.models.nets.base import SimpleSupervisedModel
from minerva.models.nets.mlp import MLP
from minerva.analysis.metrics.balanced_accuracy import BalancedAccuracy
from minerva.analysis.model_analysis import TSNEAnalysis

from minerva.models.nets.time_series.gans import TTSGAN_Encoder, GAN, TTSGAN_Generator, TTSGAN_Discriminator
import os

#############################################################################


def complete_TTSGAN_encoder_evaluation(data_path: str,
                             checkpoint_path: str,
                             root_log_dir: str,
                             execution_id: int,
                             is_basegan: int,
                             appendice: str = '',
                             num_classes: int = 6,
                             max_epochs: int = 100):
    
    log_dir = root_log_dir + f"/{os.path.basename(data_path)}/{execution_id}" 

    data_module = MultiModalHARSeriesDataModule(
        data_path=data_path,
        feature_prefixes=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
        label="standard activity code",
        features_as_channels=True,
        cast_to="float32",
        batch_size=64,
        num_workers=5)

    ckpt = torch.load(f=checkpoint_path + appendice)

    # Get state dict
    if is_basegan:
        state_dict = ckpt['state_dict']
        # Separate GAN generator and discriminatorstate_dict
        gen_state_dict = {key: value for key, value in state_dict.items() if key.startswith('gen')}
        dis_state_dict = {key: value for key, value in state_dict.items() if key.startswith('dis')}

        # Remove prefix 'gen.' e 'dis.' from keys
        gen_state_dict = {key[len('gen.') :]: value for key, value in gen_state_dict.items()}
        dis_state_dict = {key[len('dis.') :]: value for key, value in dis_state_dict.items()}

    else:
        gen_state_dict = ckpt['gen_state_dict']
        dis_state_dict = ckpt['dis_state_dict']

    #save dictionary
    save_dict = {'gen_state_dict': gen_state_dict, 'dis_state_dict': dis_state_dict}
    torch.save(save_dict, "gan_dict.pth")
    
    backbone = TTSGAN_Encoder(in_channels=6, seq_len=60)

    # Loading encoder from checkpoint
    backbone = FromPretrained(
        model=backbone,
        ckpt_path='gan_dict.pth',
        strict=False,
        ckpt_key='dis_state_dict')

    data_module.setup("fit")
    train_data_loader = data_module.train_dataloader()

    # Obtem o primeiro batch de treino (64 amostras de 6x60)
    first_batch = next(iter(train_data_loader))
    
    #print(f'first batch: {first_batch}')
    X, y = first_batch

    embeddings = backbone(X)
    mlp_input_shape = embeddings.shape[1] * embeddings.shape[2]

    head = MLP([mlp_input_shape, 128, num_classes])

    model = SimpleSupervisedModel(
        backbone=backbone,
        fc=head,
        loss_fn=torch.nn.CrossEntropyLoss(),
        flatten=True,
        train_metrics={
            "acc": torchmetrics.Accuracy(task="multiclass", num_classes=6)},
        val_metrics={
            "acc": torchmetrics.Accuracy(task="multiclass", num_classes=6)})

    ## Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        monitor='val_loss',
        mode='min',
        save_last=True)

    ## Logger
    logger = CSVLogger(save_dir=log_dir, name='tts-finetune', version=execution_id)

    ## Trainer
    trainer = L.Trainer(
        # Maximum number of epochs to train
        max_epochs=max_epochs,
        # Training on GPU
        accelerator="gpu",
        # We will train using 1 gpu
        devices=1,
        # Logger for logging
        logger=logger,
        # List of callbacks
        callbacks=[checkpoint_callback],
        # Only for testing. Remove for production. We will only train using 1 batch of training and validation
        #limit_train_batches=1,
        #limit_val_batches=1,
    )

    train_pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
        seed=42)
    
    train_pipeline.run(data_module, task="fit")

    test_pipeline = SimpleLightningPipeline(
        model=model,
        trainer=trainer,
        log_dir=log_dir,
        save_run_status=True,
        seed=42,
        classification_metrics={
            "accuracy": torchmetrics.Accuracy(num_classes=6, task="multiclass"),
            "f1": torchmetrics.F1Score(num_classes=6, task="multiclass"),
            "precision": torchmetrics.Precision(num_classes=6, task="multiclass"),
            "recall": torchmetrics.Recall(num_classes=6, task="multiclass"),
            "balanced_accuracy": BalancedAccuracy(num_classes=6, task="multiclass"),
        },
        apply_metrics_per_sample=False,
        model_analysis={
            "tsne": TSNEAnalysis(
                height=800,
                width=800,
                legend_title="Activity",
                title="t-SNE of CPC Finetuned on KuHar",
                output_filename="tsne_cpc_finetuned_kuhar.pdf",
                label_names={
                    0: "sit",
                    1: "stand",
                    2: "walk",
                    3: "stair up",
                    4: "stair down",
                    5: "run",
                    6: "stair up and down",
                },
            )
        },
    )

    d = test_pipeline.run(data_module, task="evaluate", ckpt_path=checkpoint_callback.best_model_path)
    return d








