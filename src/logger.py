from . import config

if config.USE_WANDB:
    try:
        import wandb
        from wandb.integration.keras import WandbMetricsLogger
    except ImportError:
        print("Warning: USE_WANDB is True but 'wandb' is not installed. Disabling logging.")
        config.USE_WANDB = False

class ExperimentLogger:
    @staticmethod
    def init():
        """Starts a new experiment run if enabled."""
        if not config.USE_WANDB:
            print("Running in OFFLINE mode (W&B disabled).")
            return None
        
        return wandb.init(
            project=config.WANDB_PROJECT,
            config={
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "weight_decay": config.WEIGHT_DECAY,
            "architecture": "InceptionV3",
            "dataset": "classes"
            }
        )

    @staticmethod
    def get_train_callbacks():
        """Returns W&B callbacks ONLY if enabled, otherwise returns an empty list."""
        if not config.USE_WANDB:
            return []
        
        return [WandbMetricsLogger(log_freq=5)]

    @staticmethod
    def upload_model(file_path):
        """Uploads the final best model to W&B as an artifact if enabled."""
        if not config.USE_WANDB or not wandb.run:
            return

        print(f"Uploading final best model to W&B: {file_path}")
        # Create and upload a W&B artifact
        artifact = wandb.Artifact(
            name=f"model_{wandb.run.id}", 
            type="model",
            description="Best model saved during training"
        )
        artifact.add_file(file_path)
        wandb.log_artifact(artifact)

    @staticmethod
    def log_evaluation(y_true, y_pred, class_names):
        """Logs evaluation metrics ONLY if enabled."""
        if not config.USE_WANDB or not wandb.run:
            return

        # Log interactive confusion matrix
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true, 
                preds=y_pred,
                class_names=class_names
            )
        })

        # Log classification report as a table
        from sklearn.metrics import classification_report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        table_data = [[lbl, m['precision'], m['recall']] for lbl, m in report.items() if isinstance(m, dict)]
        wandb.log({"eval_table": wandb.Table(data=table_data, columns=["class", "precision", "recall"])})

    @staticmethod
    def finish():
        """Closes the run if enabled."""
        if config.USE_WANDB and wandb.run:
            wandb.finish()