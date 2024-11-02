############################################################
#                          train.py                        #
#                   Training C2SDI model                   #
############################################################


import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pypots.optim import Adam
from pypots.imputation import CSDI


def train(
    training_dataset, validation_dataset, n_features, saving_path, 
    model_epochs, batch_size, patience, inference_mode, existed_model_path, learning_rate, n_layers
    ):

    # initialize the model
    csdi = CSDI(
        n_steps=None,
        n_features=n_features,
        n_layers=n_layers,
        n_heads=4,
        n_channels=64,
        d_time_embedding=128,
        d_feature_embedding=64,
        d_diffusion_embedding=128,
        target_strategy="random",
        n_diffusion_steps=50,
        batch_size=batch_size,
        epochs=model_epochs,
        patience=patience,
        optimizer=Adam(lr=learning_rate),

        num_workers=0,
        device="cuda",
        saving_path=saving_path,
        model_saving_strategy="best",

        d_class_embedding=32,
        n_classes=2,
        w=3.0,
    )

    if not inference_mode:
        csdi.fit(train_set=training_dataset, val_set=validation_dataset)
    elif inference_mode:
        csdi.load(existed_model_path)

    return csdi