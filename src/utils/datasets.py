import os
import urllib.request
from urllib.parse import quote

available_datasets = {"emnist-byclass-train", "emnist-byclass-test",
                   "emnist-bymerge-train", "emnist-bymerge-test",
                   "sdss17-train", "sdss17-test"}


def fetch_dataset(name, format="raw", return_map=False):    
# A function that fetches the datasets from a git repository: "https://github.com/dirakie/DataReductionML".
# example usage: df = pd.read_parquet(**fetch_dataset("emnist-byclass-train", "raw"))
    if name not in available_datasets:
        raise ValueError(f"Desired dataset does not exist. Choose from {available_datasets}.")

    root_path = "https://github.com/dirakie/DataReductionML/raw/main/Datasets/"
    dataset_dir = "emnist" if name.startswith("emnist") else "sdss17"
    dwn_path = os.path.join(root_path, f"{format}/{dataset_dir}/{name}.gzip.parquet")
    
    file_path, _ = urllib.request.urlretrieve(dwn_path)

    if return_map and not name.startswith("sdss"):
        map_path = os.path.join(root_path, name.rpartition("-")[0] +  "-" + "mapping.txt")
        return {"path": file_path, "engine": "pyarrow"}

    return {"path": file_path, "engine": "pyarrow"}

def get_map(name):
    av_maps = ["emnist-byclass", "emnist-bymerge"]
    if name not in av_maps:
        raise ValueError(f"Map of classes only exists for emnist datasets! Choose from {av_maps}.")

    root_path = "https://github.com/dirakie/DataReductionML/raw/main/Datasets/raw"
    dwn_path = f"{root_path}/emnist/{quote(name)}-mapping.txt"
    
    file_path, _ = urllib.request.urlretrieve(dwn_path)
    print(f"File downloaded to: {file_path}")
    return {"filepath_or_buffer": file_path, "header": None, "sep": " "}
