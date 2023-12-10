# Third-party libraries
import mlflow
import wandb

# Local application/modules
from src.constants.types import Optional, Any, Dict


class CustomLogger:
    def __init__(self, backend: str, project_name: str = None, exp_name: str = None) -> None:
        """
        Initialize the CustomLogger.

        :param backend: The logging backend to use ("mlflow" or "wandb").
        """
        if backend == "mlflow":
            mlflow.start_run()
        elif backend == "wandb":
            wandb.init(project=project_name, name=exp_name)

        self.backend = backend

    def log_metric(self, key: str, value: float, step: Optional[int] = None, step_name: str ='epoch') -> None:
        """
        Log a metric value.

        :param key: The name of the metric.
        :param value: The value of the metric.
        :param step: The step or epoch for the logged metric. Default is None.
        """
        if self.backend == "mlflow":
            mlflow.log_metric(key, value, step=step)
        elif self.backend == "wandb":
            wandb.log({key: value, step_name: step})

    def log_image(self, image_key: str, image: Any, caption: Optional[str] = None, file_path: Optional[str] = None) -> None:
        """
        Log an image.

        :param image_key: The key or name for the logged image.
        :param image: The image to log.
        :param caption: Caption for the image. Default is None.
        """
        if self.backend == "mlflow":
            mlflow.log_artifact(file_path)
        elif self.backend == "wandb":
            wandb.log({image_key: [wandb.Image(image, caption=caption)]})

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters or configurations.

        :param params: Dictionary containing the parameters or configurations to log.
        """
        if self.backend == "mlflow":
            mlflow.log_params(params)
        elif self.backend == "wandb":
            wandb.config.update(params)

    def log_artifact(self, file_path: str) -> None:
        """
        Log an artifact, typically a file.

        :param file_path: Path to the file to be logged.
        """
        if self.backend == "mlflow":
            mlflow.log_artifact(file_path)
        elif self.backend == "wandb":
            pass
    
    def finish(self) -> None:
        """
        Finish experiment logging.
        """
        if self.backend == "mlflow":
            mlflow.finish()
        elif self.backend == "wandb":
            wandb.finish()
