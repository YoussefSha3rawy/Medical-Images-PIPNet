import wandb
import datetime
import numpy as np

class WandbLogger:
    step = 0

    def __init__(self, config, logger_name='logger', project='inm706'):
        logger_name = f'{logger_name}-{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}'
        logger = wandb.init(project=project, name=logger_name, config=config)
        self.logger = logger

    def log(self, data):
        self.logger.log(data)

    def log_confusion_matrix(self, y_true, y_pred):
        """
        Logs a confusion matrix to W&B and returns the confusion matrix.

        Parameters:
        y_true (list or np.array): True labels.
        y_pred (list or np.array): Predicted labels.

        Returns:
        np.array: Confusion matrix.
        """
        # Infer class names
        class_names = [str(cl) for cl in np.unique(np.concatenate((y_true, y_pred)))]

        # Log the confusion matrix plot to W&B
        self.logger.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true,
            preds=y_pred,
            class_names=class_names
        )})


    def watch(self, model):
        self.logger.watch(model, log='all')

    def log_artifact(self, artifact):
        self.logger.log_artifact(artifact)
