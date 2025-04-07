import time
import warnings
from torch.utils.data import DataLoader

class TimedDataLoader(DataLoader):
    def __iter__(self):
        iterator = super().__iter__()
        while True:
            start_time = time.time()
            item = next(iterator)
            elapsed_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            if elapsed_time > 250:
                warnings.warn(f"DataLoader yielding blocked for {elapsed_time:.2f}ms")
            yield item

