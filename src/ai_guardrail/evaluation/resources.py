from __future__ import annotations

import threading


class PeakRssSampler:
    def __init__(
        self,
        interval_seconds: float = 0.01,
        process_id: int | None = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("sample interval must be positive")
        try:
            import psutil
        except ImportError:
            raise RuntimeError(
                "peak RSS sampling requires the ml optional dependencies"
            ) from None
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(process_id)
        self._process_error = psutil.Error
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read_rss(self) -> int:
        try:
            return self.process.memory_info().rss
        except self._process_error:
            return 0

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(
                self.peak_bytes,
                self._read_rss(),
            )
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> PeakRssSampler:
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.peak_bytes = max(
            self.peak_bytes,
            self._read_rss(),
        )

    @property
    def peak_mebibytes(self) -> float:
        return self.peak_bytes / 1024 / 1024
