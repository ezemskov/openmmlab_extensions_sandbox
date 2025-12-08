from collections import deque
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MovingAvgLossHook(Hook):
    """
    Logs a moving average of the training loss so ClearML can display it.
    """

    def __init__(self, window_size=100, key='loss', out_key='loss_avg'):
        self.window_size = window_size
        self.key = key
        self.out_key = out_key
        self.buffer = deque(maxlen=window_size)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # Latest logged values for the "train" group
        log_vars = runner.message_hub.log_scalars()

        if self.key in log_vars:
            val = float(log_vars[self.key])
            self.buffer.append(val)

            avg_val = sum(self.buffer) / len(self.buffer)

            # Send smoothed loss metric into MMEngine's scalar system
            runner.message_hub.update_scalar(self.out_key, avg_val)
