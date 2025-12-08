from collections import deque
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MovingAvgLossHook(Hook):
    """
    Logs a moving average of the training loss so ClearML can display it.
    """

    def __init__(self, key, out_key, window_size=100):
        self.window_size = window_size
        self.key = key
        self.out_key = out_key
        self.buffer = deque(maxlen=window_size)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        valBuf = runner.message_hub.get_scalar(self.key)
        currVal = float(valBuf.current())
        self.buffer.append(currVal)
        avg_val = sum(self.buffer) / len(self.buffer)

        # Send smoothed loss metric into MMEngine's scalar system
        runner.message_hub.update_scalar(self.out_key, avg_val)
"""
"""        