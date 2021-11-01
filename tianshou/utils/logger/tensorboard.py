import warnings
import os
from typing import Any, Callable, Optional, Tuple

from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils.logger.base import LOG_DATA_TYPE, BaseLogger


class TensorboardLogger(BaseLogger):
    """A logger that relies on tensorboard SummaryWriter by default to visualize \
    and log statistics.

    :param SummaryWriter writer: the writer to log data.
    :param int train_interval: the log interval in log_train_data(). Default to 1000.
    :param int test_interval: the log interval in log_test_data(). Default to 1.
    :param int update_interval: the log interval in log_update_data(). Default to 1000.
    :param int save_interval: the save interval in save_data(). Default to 1 (save at
        the end of each epoch).
    """

    def __init__(
        self,
        writer: SummaryWriter,
        train_interval: int = 1000,
        test_interval: int = 1,
        update_interval: int = 1000,
        save_interval: int = 1,
    ) -> None:
        super().__init__(train_interval, test_interval, update_interval)
        self.save_interval = save_interval
        self.last_save_step = -1
        self.writer = writer
        logdir = self.writer.log_dir
        self.train_csv = CSVOutputFormat(os.path.join(logdir, "train.csv"))
        self.test_csv = CSVOutputFormat(os.path.join(logdir, "test.csv"))

    def write(self, step_type: str, step: int, data: LOG_DATA_TYPE) -> None:
        if "test" in step_type:
            self.test_csv.writekvs(data)
        if "train" in step_type:
            self.train_csv.writekvs(data)
        for k, v in data.items():
            self.writer.add_scalar(k, v, global_step=step)

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Optional[Callable[[int, int, int], None]] = None,
    ) -> None:
        if save_checkpoint_fn and epoch - self.last_save_step >= self.save_interval:
            self.last_save_step = epoch
            save_checkpoint_fn(epoch, env_step, gradient_step)
            self.write("save/epoch", epoch, {"save/epoch": epoch})
            self.write("save/env_step", env_step, {"save/env_step": env_step})
            self.write(
                "save/gradient_step", gradient_step,
                {"save/gradient_step": gradient_step}
            )

    def restore_data(self) -> Tuple[int, int, int]:
        ea = event_accumulator.EventAccumulator(self.writer.log_dir)
        ea.Reload()

        try:  # epoch / gradient_step
            epoch = ea.scalars.Items("save/epoch")[-1].step
            self.last_save_step = self.last_log_test_step = epoch
            gradient_step = ea.scalars.Items("save/gradient_step")[-1].step
            self.last_log_update_step = gradient_step
        except KeyError:
            epoch, gradient_step = 0, 0
        try:  # offline trainer doesn't have env_step
            env_step = ea.scalars.Items("save/env_step")[-1].step
            self.last_log_train_step = env_step
        except KeyError:
            env_step = 0

        return epoch, env_step, gradient_step


class BasicLogger(TensorboardLogger):
    """BasicLogger has changed its name to TensorboardLogger in #427.

    This class is for compatibility.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "Deprecated soon: BasicLogger has renamed to TensorboardLogger in #427."
        )
        super().__init__(*args, **kwargs)


class CSVOutputFormat:
    def __init__(self, filename):
        self.file = open(filename, 'w+t')
        self.keys = []
        self.sep = '\t'

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(kvs.keys() - self.keys)
        extra_keys.sort()
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write('\t')
                self.file.write(k)
            self.file.write('\n')
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write('\n')
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write('\t')
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write('\n')
        self.file.flush()

    def close(self):
        self.file.close()
