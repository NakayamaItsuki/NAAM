import re
import pandas as pd
import os
import argparse

script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_means_from_file(file_path):
    datasets = {}
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, 'r') as file:
        dataset_name = None
        for line in file:
            if "Namespace(dataset=" in line:
                dataset_name = re.search(r"dataset='(\w+)'", line).group(1)

            if 'mean' in line and dataset_name:
                mean_values = list(map(float, re.findall(r"[\d\.]+", line)))
                obs_test_acc_mean, unobs_test_acc_mean, test_acc_mean = mean_values[:3]

                datasets.setdefault(dataset_name, {})[file_name] = {
                    'obs_test_acc_mean': obs_test_acc_mean,
                    'unobs_test_acc_mean': unobs_test_acc_mean,
                    'test_acc_mean': test_acc_mean
                }
    return datasets


def combine_results(file_paths):
    combined_data = {}

    for file_path in file_paths:
        file_data = extract_means_from_file(file_path)
        for dataset, method_data in file_data.items():
            combined_data.setdefault(dataset, {}).update(method_data)

    final_dfs = {
        dataset: pd.DataFrame.from_dict(
            methods, orient='index',
            columns=['obs_test_acc_mean', 'unobs_test_acc_mean', 'test_acc_mean']
        ).rename_axis('Method')
        for dataset, methods in combined_data.items()
    }
    return final_dfs


def main(ratio):
    target_dir = os.path.join(script_dir, str(ratio))

    if not os.path.isdir(target_dir):
        print(f"Error: The directory '{target_dir}' does not exist.")
        return

    file_paths = [os.path.join(target_dir, file_path) for file_path in os.listdir(target_dir) if file_path.endswith('.txt')]

    dataset_dfs = combine_results(file_paths)

    for dataset in ["cora", "citeseer", "pubmed", "amazon_computers", "amazon_photo"]:
        if dataset in dataset_dfs:
            print(f"Dataset: {dataset}")
            print(dataset_dfs[dataset].reset_index())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=str, help="Inductive ratio")
    args = parser.parse_args()
    
    main(args.ratio)
