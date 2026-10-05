
from datetime import datetime
from backend.inference.model_cache import model_cache
from backend.task_manager.task import FAILURE_TYPE_GENERAL, Task
from backend.training.engine import BaseTrainingEngine
from util.logger import get_logger

logger = get_logger(__name__)

class TrainingTask(Task):

    def __init__(self, task_id: str, training_engine: BaseTrainingEngine):
        super().__init__(task_id=task_id)
        self.training_engine = training_engine

    @classmethod
    def from_engine(cls, task_id: str, training_engine: BaseTrainingEngine) -> 'Task':
        return cls(task_id, training_engine)

    def run(self):
        model_cache.evict(reason="training needs the GPU")
        self.training_engine.train()

    def task_failure(self, error_msg: str, failure_type: str = FAILURE_TYPE_GENERAL):
        # let trainer object handle failure
        self.training_engine.visit_training_failed(datetime.now().timestamp(), error_msg, failure_type=failure_type)

    def task_success(self, message: str):
        # Handle training success logic here
        pass

    def unwrap(self) -> BaseTrainingEngine:
        return self.training_engine

    def task_appended(self, message: str):
        pass

    def _task_finalize(self):
        try:
            self.training_engine.finalize()
        except Exception as e:
            logger.warning(f"Ignore error with the training task finalize failed for task {self.task_id}:", exc_info=e)
            pass
