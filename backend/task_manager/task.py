import threading

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from backend.inference.model_cache import model_cache
from util.logger import get_logger

logger = get_logger(__name__)

FAILURE_TYPE_GENERAL = "general"
FAILURE_TYPE_OOM = "out_of_memory"

# Upper bound between idle checks of the model cache; new tasks wake the loop immediately
TASK_IDLE_WAIT_SECONDS = 1.0

class Task(ABC):

    def __init__(self, task_id: str = None):
        self.task_id = task_id

    def id(self) -> str:
        return self.task_id

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def task_failure(self, error_msg: str, failure_type: str = FAILURE_TYPE_GENERAL):
        pass

    @abstractmethod
    def task_success(self, message: str):
        pass

    @abstractmethod
    def task_appended(self, message: str):
        pass

    @abstractmethod
    def unwrap(self) -> Any:
        pass

    def task_finalize(self):
        try:
            self._task_finalize()
        except Exception as e:
            logger.warning(f"Task finalized failed {self.task_id} finalize failed:", exc_info=e)
        pass

    @abstractmethod
    def _task_finalize(self):
        pass


class Manager:
    def __init__(self):
        self.task: Task = None
        self._wakeup = threading.Event()
        self._task_done = threading.Condition()

    def task_run_loop(self):
        while True:
            # Cleared before reading self.task so an add_task racing with this check still wakes the wait
            self._wakeup.clear()
            task = self.task
            if task is None:
                self._evict_model(idle=True)
                self._wakeup.wait(TASK_IDLE_WAIT_SECONDS)
                continue
            self._run_task(task)

    def _run_task(self, task: Task):
        try:
            task.run()
            task.task_success("Task run completed successfully.")
        # OutOfMemoryError subclasses RuntimeError, so it must be caught before Exception
        except torch.cuda.OutOfMemoryError as e:
            logger.error("TaskManager task_run_loop CUDA OOM error:", exc_info=e)
            self._evict_model("out of memory")
            self._report_failure(task, "CUDA Out of Memory Error", FAILURE_TYPE_OOM)
        except Exception as e:
            logger.error("TaskManager task_run_loop error:", exc_info=e)
            # A task that died mid-forward can leave offloaded layers on the wrong device
            self._evict_model("task failed")
            self._report_failure(task, str(e), FAILURE_TYPE_GENERAL)
        finally:
            task.task_finalize()
            model_cache.touch()
            with self._task_done:
                self.task = None
                self._task_done.notify_all()

    def _report_failure(self, task: Task, error_msg: str, failure_type: str):
        try:
            task.task_failure(error_msg, failure_type=failure_type)
        except Exception as e:
            logger.error(f"Failed to record the failure of task {task.id()}:", exc_info=e)

    def _evict_model(self, reason: str = "", idle: bool = False):
        try:
            if idle:
                model_cache.evict_if_idle()
            else:
                model_cache.evict(reason=reason)
        except Exception as e:
            logger.warning("Failed to unload the cached model:", exc_info=e)

    def add_task(self, task: Task) -> bool:
        # Concurrent requests must not both claim the single slot
        with self._task_done:
            if self.task is not None:
                logger.warning(f"Cannot add task id{task.id()}, another "
                               f"inference {self.task.id()} is in progress")
                return False
            task.task_appended("Task added to the manager.")
            self.task = task
        logger.info(f"Added task id{task.id()} to current inference")
        self._wakeup.set()
        return True

    def get_current_task(self) -> Task:
        if self.task is not None:
            return self.task
        return None

    def has_task(self) -> bool:
        return self.task is not None

    def wait_for_task(self, task_id: str, timeout: float) -> bool:
        """Block until task_id is no longer running; returns False on timeout."""
        with self._task_done:
            return self._task_done.wait_for(
                lambda: self.task is None or self.task.id() != task_id, timeout)


gm = Manager()
threading.Thread(target=gm.task_run_loop, daemon=True).start()
