import matplotlib.pyplot as plt

class Visualizations:
    """
    A class to provide visualization tools for data analysis, including histograms and boxplots.

    Attributes:
        data (array-like): The data to be visualized.
    """

    def __init__(self, data):
        """
        Initializes the Visualizations class with the given data.

        Args:
            data (array-like): The data to be visualized.
        """
        self.data = data

    def histogram(self, bins=10, title="Histogram", xlabel="Values", ylabel="Frequency"):
        """
        Generates a histogram for the data.

        Args:
            bins (int, optional): Number of bins for the histogram. Default is 10.
            title (str, optional): Title of the histogram. Default is "Histogram".
            xlabel (str, optional): Label for the x-axis. Default is "Values".
            ylabel (str, optional): Label for the y-axis. Default is "Frequency".
        """
        plt.figure(figsize=(8, 6))
        plt.hist(self.data, bins=bins, edgecolor="k", alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    def boxplot(self, title="Boxplot", ylabel="Values"):
        """
        Generates a boxplot for the data.

        Args:
            title (str, optional): Title of the boxplot. Default is "Boxplot".
            ylabel (str, optional): Label for the y-axis. Default is "Values".
        """
        plt.figure(figsize=(8, 6))
        plt.boxplot(self.data, vert=True, patch_artist=True, boxprops=dict(facecolor="skyblue", color="black"))
        plt.title(title)
        plt.ylabel(ylabel)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

# Example usage
# Uncomment the following lines to test the Visualizations class.
# data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
# vis = Visualizations(data)
# vis.histogram(bins=5, title="Sample Histogram", xlabel="Data Points", ylabel="Count")
# vis.boxplot(title="Sample Boxplot", ylabel="Data Points")
