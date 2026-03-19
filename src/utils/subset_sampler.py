import numpy as np
import pandas as pd

class SubsetSampler:
    def __init__(self, dataframe, target_column, samples_per_class=100):
        self.dataframe = dataframe
        self.target_column = target_column
        self.samples_per_class = samples_per_class

    def stratified_sample(self):
        # Get the classes and their counts
        class_counts = self.dataframe[self.target_column].value_counts()
        sampled_data = []

        for class_label, count in class_counts.items():
            # Get the subset of data for this class
            class_data = self.dataframe[self.dataframe[self.target_column] == class_label]
            # Randomly sample from this class
            sampled_class_data = class_data.sample(n=min(self.samples_per_class, len(class_data)), random_state=42)
            sampled_data.append(sampled_class_data)

        # Combine all sampled data into a single DataFrame
        balanced_subset = pd.concat(sampled_data, ignore_index=True)
        return balanced_subset

# Example usage:
# df = pd.read_csv('your_dataset.csv')
# sampler = SubsetSampler(dataframe=df, target_column='sentiment')
# balanced_subset = sampler.stratified_sample()