############################################################
#                          train.py                        #
#                   Training C2SDI model                   #
############################################################

from pypots.optim import Adam
from pypots.imputation import CSDI

def train(training_dataset, validation_dataset, n_features, saving_path):
  # initialize the model
  csdi = CSDI(
    n_steps=300,
    n_features=n_features,
    n_layers=8,
    n_heads=4,
    n_channels=64,
    d_time_embedding=128,
    d_feature_embedding=64,
    d_diffusion_embedding=128,
    target_strategy="random",
    n_diffusion_steps=50,
    batch_size=32,
    epochs=20,
    patience=10,
    optimizer=Adam(lr=5e-5),

    num_workers=0,
    device="cuda:0",
    saving_path=saving_path,
    model_saving_strategy="best",

    d_class_embedding=32,
    n_classes=2,
    w=3.0,
  )

  csdi.fit(train_set=training_dataset, val_set=validation_dataset)

  return csdi