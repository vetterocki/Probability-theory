import math
import statistics

import numpy as np
from scipy.stats import chi2
from scipy.stats import t
from tabulate import tabulate


def t_distribution(percent, size):
    return abs(t.ppf(percent, size - 1))


class DescriptiveStatistics:
    def __init__(self, loc, scale, size, percent):
        self.loc = loc
        self.scale = scale
        self.size = size
        self.percent = percent
        self.data = np.random.normal(loc, math.sqrt(scale), size)

    @property
    def mean(self):
        return 1 / len(self.data) * sum(self.data)

    @property
    def deviation(self):
        return math.sqrt(sum(map(lambda elem: (elem - self.mean) ** 2, self.data)) / (len(self.data) - 1))

    @property
    def confidence_interval_expected_value(self):
        t_d = t_distribution((1 - self.percent) / 2, self.size - 1)
        d = self.deviation
        return self.mean - (t_d * d / math.sqrt(len(self.data))), self.mean + (t_d * d / math.sqrt(len(self.data)))

    @property
    def confidence_interval_deviation(self):
        length = len(self.data)
        lower_critical_value = chi2.ppf((1 - self.percent) / 2, length - 1)
        upper_critical_value = chi2.ppf((1 + self.percent) / 2, length - 1)
        return np.sqrt((length - 1) * np.square(self.deviation) / upper_critical_value), np.sqrt(
            (length - 1) * np.square(self.deviation) / lower_critical_value)

    def checks(self):
        print(np.std(self.data))
        print(statistics.stdev(self.data))
        data = self.data
        mean = np.mean(data)
        std_dev = np.std(data)
        confidence_level = 0.95
        n = len(data)
        df = n - 1
        t_critical = t.ppf((1 + confidence_level) / 2, df)
        margin_of_error = t_critical * (std_dev / np.sqrt(n))
        confidence_interval = (mean - margin_of_error, mean + margin_of_error)
        print("Довірчий інтервал:", confidence_interval)


class DependencyResolver:
    def __init__(self, d_s: DescriptiveStatistics, percents, samples):
        self.d_s = d_s
        self.percents = percents
        self.samples = samples

    def generate_dependencies(self):
        temp = []
        for sample in self.samples:
            for percent in self.percents:
                temp_d_s = DescriptiveStatistics(self.d_s.loc, self.d_s.scale, len(sample), percent)
                temp.append([len(sample), percent, temp_d_s.confidence_interval_expected_value,
                             temp_d_s.confidence_interval_deviation])
        return temp

    def print_dependencies(self):
        data = self.generate_dependencies()
        data.insert(0, ["Довжина", "Відсоток", "Довірчий інтервал на математичне сподівання",
                        "Довірчий інтервал на середньоквадратичне відхилення"])
        table = tabulate(data, headers="firstrow", tablefmt="fancy_grid")
        print("Досліження залежності оцінок від рівня довіри та обсягу вибірки", table, sep="\n")


class StatisticsPrinter:
    def __init__(self, d_s: DescriptiveStatistics):
        self.d_s = d_s

    def print_credentials(self):
        print(f"Вибірка. {self.d_s.data}\n")
        print(f"n = {self.d_s.size}, σ = {self.d_s.scale}. % = {self.d_s.percent}")
        print(f"Середньоквадратичне відхилення  = {self.d_s.deviation}")
        print(f"Середнє значення = {self.d_s.mean}")
        print(f"Довірчий інтервал на математичне сподівання = {self.d_s.confidence_interval_expected_value}")
        print(f"Довірчий інтервал на середньоквадратичне відхилення = {self.d_s.confidence_interval_deviation}\n")


class ResolverUtils:
    @staticmethod
    def generate_samples(initial_data, size):
        sizes = [len(initial_data) // (2 ** i) for i in range(size)]
        return [initial_data[:size] for size in sizes]

    @staticmethod
    def generate_percents(initial_percent, size):
        decreasing_values = [max(initial_percent - (0.1 * i), 0.2) for i in range(size // 2)]
        increasing_values = [min(initial_percent + (0.1 * (i - size // 2)), 0.99) for i in range(size // 2, size)]
        return list(set(decreasing_values + increasing_values))


ds = DescriptiveStatistics(10, 2.2, 145, 0.95)
printer = StatisticsPrinter(ds)
printer.print_credentials()
resolver = DependencyResolver(ds, ResolverUtils.generate_percents(0.95, 10), ResolverUtils.generate_samples(ds.data, 6))
resolver.print_dependencies()
