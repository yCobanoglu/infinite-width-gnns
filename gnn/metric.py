from typing import Sequence

import numpy as np
import torch
from tabulate import tabulate


def aggregate_list_of_metric(list_of_metric):
    start = list_of_metric[0]
    for metric in list_of_metric[1:]:
        start.merge(metric)
    return start


def aggregate_list_of_metric_lists(list_of_metric_lists):
    aggregated = np.array([[metric for metric in metric_list.metrics] for metric_list in list_of_metric_lists])
    cols = aggregated.T.tolist()
    return MetricList([aggregate_list_of_metric(metrics) for metrics in cols])


class Metric:
    def __init__(self, stage: str, performance: dict):
        self.stage = stage
        self.performance = {k: round((v.item() if torch.is_tensor(v) else v), 3) for k, v in performance.items()}

    def aggregate_single(self):
        self.performance = {k: [v] for k, v in self.performance.items()}
        return self

    @property
    def is_accumulated(self):
        return all([isinstance(v, list) for v in self.performance.values()])

    def _merge_performance(self, dict):
        for k, v in dict.items():
            if isinstance(v, list):
                self.performance[k].extend(v)
            else:
                self.performance[k].append(v)

    def merge(self, element: "Metric"):
        if self.is_accumulated:
            self._merge_performance(element.performance)
        else:
            self.performance = {k: [v] for k, v in self.performance.items()}
            self.merge(element)

    def __str__(self):
        aggr = ""
        for k, v in self.performance.items():
            aggr += f"{self.stage}.{k}: {round(v, 3)} "
        return aggr

    def to_dict(self):
        return {"stage": self.stage, "performance": self.performance}


class MetricList:
    def __init__(self, metrics: Sequence = tuple()):
        self.metrics = list(metrics)

    def add(self, performance: Metric):
        self.metrics.append(performance)

    @classmethod
    def create(cls, metric: Metric):
        return cls([metric])

    def to_dict(self):
        return {m.stage: m.performance for m in self.metrics}

    def __str__(self):
        return self.to_table()

    def to_table(self, is_concise=False):
        def format_metric_row(metric_stage, key, v):
            def r(number):
                return round(number, 3)

            if "params" in metric_stage:
                if isinstance(v, list):
                    return ["params", v[0]]
                else:
                    return ["params", v]

            name = f"{metric_stage}.{key}"
            avg = f"{r(np.mean(v))}±{r(np.std(v))}"
            min_max = f"{r(np.min(v))},{r(np.max(v))}"

            if is_concise and ("test" in metric.stage or "val" in metric.stage) and ("loss" not in k):
                return [name, avg]
            if not is_concise:
                return [name, avg, min_max]
            return None

        rows = []
        for metric in self.metrics:
            for k, v in metric.performance.items():
                row = format_metric_row(metric.stage, k, v)
                if row is not None:
                    rows.append(format_metric_row(metric.stage, k, v))

        return tabulate(rows, tablefmt="plain")

    def __repr__(self):
        return self.__str__()
