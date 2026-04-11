from __future__ import annotations

import bisect


class BinnedCalibrator:
    """Equal-frequency binning calibrator with Laplace smoothing."""

    def __init__(self, n_bins: int = 15, laplace: float = 1.0) -> None:
        self.n_bins = max(2, n_bins)
        self.laplace = laplace
        self.boundaries: list[float] = []
        self.bin_rates: list[float] = []
        self.global_rate: float = 0.5

    def fit(self, confidences: list[float], labels: list[int]) -> None:
        if not confidences:
            raise ValueError("No confidences provided")
        if len(confidences) != len(labels):
            raise ValueError("Length mismatch between confidences and labels")

        pairs = sorted(zip(confidences, labels), key=lambda x: x[0])
        sorted_conf = [p[0] for p in pairs]

        self.boundaries = []
        for i in range(1, self.n_bins):
            index = int(i * len(sorted_conf) / self.n_bins)
            index = min(max(index, 0), len(sorted_conf) - 1)
            self.boundaries.append(sorted_conf[index])

        # Keep increasing unique boundaries only.
        unique: list[float] = []
        for value in self.boundaries:
            if not unique or value > unique[-1]:
                unique.append(value)
        self.boundaries = unique

        bins_correct = [0] * (len(self.boundaries) + 1)
        bins_count = [0] * (len(self.boundaries) + 1)

        for conf, label in pairs:
            idx = bisect.bisect_right(self.boundaries, conf)
            bins_correct[idx] += int(label)
            bins_count[idx] += 1

        self.global_rate = sum(labels) / len(labels)
        self.bin_rates = []
        for correct, count in zip(bins_correct, bins_count):
            # Laplace smoothing to avoid degenerate 0/1 estimates.
            rate = (correct + self.laplace) / (count + 2.0 * self.laplace)
            self.bin_rates.append(rate)

    def predict_one(self, confidence: float) -> float:
        if not self.bin_rates:
            return self._clamp(confidence)
        idx = bisect.bisect_right(self.boundaries, confidence)
        if idx >= len(self.bin_rates):
            return self.global_rate
        return self._clamp(self.bin_rates[idx])

    def predict_many(self, confidences: list[float]) -> list[float]:
        return [self.predict_one(c) for c in confidences]

    @staticmethod
    def _clamp(value: float) -> float:
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value
