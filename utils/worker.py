from PyQt6.QtCore import QObject, pyqtSignal


class Worker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._func(*self._args, **self._kwargs)
            self.finished.emit(result)
        except MemoryError:
            self.error.emit("Out of memory. Try a smaller kernel size or image.")
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
