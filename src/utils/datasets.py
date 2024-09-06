import os
import urllib.request

available_datasets = {"emnist-byclass-train", "emnist-byclass-test",
                   "emnist-bymerge-train", "emnist-bymerge-test",
                   "sdss17-train", "sdss17-test"}


def fetch_dataset(name, format="raw"):    
# A function that fetches the datasets from a git repository: "https://github.com/dirakie/DataReductionML".
# example usage: df = pd.read_parquet(**fetch_dataset("raw", "emnist-byclass-train"))
    if name not in available_datasets:
        raise ValueError(f"Desired dataset does not exist. Choose from {available_datasets}")

    root_path = "https://github.com/dirakie/DataReductionML/raw/main/Datasets/"
    dataset_dir = "emnist" if name.startswith("emnist") else "sdss17"
    dwn_path = os.path.join(root_path, f"{format}/{dataset_dir}/{name}.gzip.parquet")
    
    file_path, _ = urllib.request.urlretrieve(dwn_path)
    return {"path": file_path, "engine": "pyarrow"}