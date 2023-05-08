import math

import numpy as np
import matplotlib.pyplot as plt


class DescriptiveStatistics:
    def __init__(self, loc, scale, size):
        self.data = np.random.normal(loc, scale, size)

    @property
    def median(self):
        sorted_data = sorted(self.data)
        mid_value = len(sorted_data) // 2
        return (sorted_data[mid_value - 1] + sorted_data[mid_value]) / 2 if len(sorted_data) % 2 == 0 \
            else sorted_data[mid_value]

    @property
    def mode(self):
        counts = {}
        for elem in self.data:
            counts[elem] = counts.get(elem, 0) + 1
        return np.array([elem for elem, amount in counts.items() if amount == max(counts.values())])

    @property
    def mean(self):
        return 1 / len(self.data) * sum(self.data)

    @property
    def dispersion(self):
        mean = self.mean
        squared_s = sum(list(map(lambda elem: (elem - mean) ** 2, self.data))) / (len(self.data) - 1)
        return squared_s, math.sqrt(squared_s)

    def frequency_polygon(self):
        bins = np.linspace(min(self.data), max(self.data), num=10)
        hist, _ = np.histogram(self.data, bins=bins)

        midpoints = (bins[:-1] + bins[1:]) / 2

        plt.plot(midpoints, hist, '-o')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    def frequency_histogram(self):
        plt.hist(self.data, bins=15, alpha=0.6, color='g', edgecolor='black')

        plt.xlabel('Значення')
        plt.ylabel('Частота')
        plt.title('Гістограма частот')

        plt.show()

    def pareto(self):
        hist, bins = np.histogram(self.data, bins=10)

        cumulative_freq = np.cumsum(hist)
        cumulative_perc = 100 * cumulative_freq / np.sum(hist)

        sorted_idx = np.argsort(cumulative_perc)[::-1]
        cumulative_perc = cumulative_perc[sorted_idx]
        bins = bins[sorted_idx]

        fig, ax1 = plt.subplots()
        ax1.bar(bins, cumulative_perc, color='b')
        ax1.set_ylabel('Кумулятивна частота')
        ax1.invert_xaxis()  # Доданий рядок

        ax2 = ax1.twinx()
        ax2.plot(bins, cumulative_freq, '-ro')
        ax2.set_ylabel('Кумулятивний відсоток')

        plt.title('Діаграма Парето')

        plt.show()

    def box_and_whisker_plot(self):
        q1, q2, q3 = np.percentile(self.data, [25, 50, 75])
        iqr = q3 - q1
        min_val, max_val = np.min(self.data), np.max(self.data)
        outliers = np.logical_or(self.data < q1 - 1.5 * iqr, self.data > q3 + 1.5 * iqr)
        fig, ax = plt.subplots()
        ax.boxplot(self.data, vert=False, whis=1.5)

        ax.set_xlabel('Значення')
        ax.set_ylabel('Вибірка')
        ax.set_title('Діаграма розмаху')

        ax.axvline(q1, linestyle='--', color='r', label='Q1')
        ax.axvline(q2, linestyle='--', color='g', label='Медіана')
        ax.axvline(q3, linestyle='--', color='b', label='Q3')
        ax.plot(min_val, 1, marker='o', markersize=5, color='black', label='Min')
        ax.plot(max_val, 1, marker='o', markersize=5, color='black', label='Max')

        ax.legend()

        if np.sum(outliers) > 0:
            ax.text(0.05, 0.95, 'Присутні статистичні викиди', transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(facecolor='red', alpha=0.3), fontsize=10)

        plt.show()

    def circle(self):
        counts, bins = np.histogram(self.data, bins=10)
        percents = counts / len(self.data) * 100

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(percents, labels=bins[:-1], autopct='%1.1f%%')
        ax.set_title('Кругова діаграма')
        plt.show()

    def draw_all_graphics(self):
        self.frequency_polygon()
        self.frequency_histogram()
        self.box_and_whisker_plot()
        self.pareto()
        self.circle()

    def print_all_properties(self):
        print("Sample data", self.data, "\n")
        properties = [name for name in dir(DescriptiveStatistics) if
                      isinstance(getattr(DescriptiveStatistics, name), property)]
        for elem in properties:
            value = getattr(self, elem)
            print(f"{elem} = {value}\n")


stats = DescriptiveStatistics(5, 2, 115)
stats.print_all_properties()
stats.draw_all_graphics()
