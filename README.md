# Seisbench Training : A Configurable Seismic Phase Picking Pipeline

PhasePicker is a fully **configurable, Hydra and Mlflow driven pipeline** for seismic phase picking built on top of **SeisBench**. It supports flexible dataset loading, augmentation, model configuration, not only training workflows but also the evaluation workflow using **Mlflow**.

## 🚀 Features

* **Hydra Configuration** for every component:

  * Dataset (name, component_orders, dimension_orders, samplng_rate)
  * Augmentations (probabilitic_labeller, normalize, randow_window)
  * Training (lr, epochs, batch_size, n_workers, optimizer)
  * Model (model_name)
  * 
* **Reproducible experiments** with Hydra logging and config saving
  
* **MLflow Pipeline** for evaluation and saving best model.

   * Setup Mlflow callback that evaluates model metrics during training from validation dataset and save metrics and plots in mlflow.
   * Once the training is finished it saves the best_model as artifact with the details of the model and the config that it was trained on.
  


  


