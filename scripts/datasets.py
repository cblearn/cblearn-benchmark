import os
from pathlib import Path
import json

from cblearn import datasets, preprocessing
import pandas as pd
import tqdm
import numpy as np


DATA_HOME = Path(os.environ.get('CBLEARN_DATA', Path(__file__).parent / '../datasets'))

DATASETS = {
   'car', 'food', 'imagenet-v1', 'imagenet-v2', 'nature', 'material', 'musician', 'vogue', 'things'
}


def fetch_dataset(dataset, download_if_missing=False):
    data_home = DATA_HOME / 'download'
    data_home.mkdir(exist_ok=True)
    match dataset:
        case 'car':
            data = datasets.fetch_car_similarity(data_home, download_if_missing)
            triplets = preprocessing.triplets_from_mostcentral(data.triplet, data.response)
        case 'food':
            data = datasets.fetch_food_similarity(data_home, download_if_missing)
            triplets = data.data
        case 'imagenet-v1':
            data = datasets.fetch_imagenet_similarity(data_home, download_if_missing)
            triplets = preprocessing.triplets_from_multiselect(data.data, data.n_select, is_ranked=data.is_ranked)
        case 'imagenet-v2':
            data = datasets.fetch_imagenet_similarity(data_home, download_if_missing, version='0.2')
            triplets = preprocessing.triplets_from_multiselect(data.data, data.n_select, is_ranked=data.is_ranked)

            mask = ((triplets[:, 0] != triplets[:, 1])
                    & (triplets[:, 1] != triplets[:, 2])
                    & (triplets[:, 0] != triplets[:, 2]))
            triplets = triplets[mask]  # remove invalid triplets (n=1)
        case 'nature':
            data = datasets.fetch_nature_scene_similarity(data_home, download_if_missing)
            triplets = data.triplet
        case 'material':
            data = datasets.fetch_material_similarity(data_home, download_if_missing)
            triplets = data.triplet
        case 'musician':
            data = datasets.fetch_musician_similarity(data_home, download_if_missing,
                                                      valid_triplets=True)
            triplets = data.data
        case 'vogue':
            data = datasets.fetch_vogue_cover_similarity(data_home, download_if_missing)
            triplets = data.triplet
        case 'things':
            data = datasets.fetch_things_similarity(data_home, download_if_missing)
            triplets = preprocessing.triplets_from_oddoneout(data.data)
        case other:
            raise ValueError(f'Unknown dataset `{dataset}`')
    return {
        'train_triplets': triplets.astype(int).tolist(),
        'n_objects': int(np.amax(triplets)) + 1
    }



def download_all():
    for dataset in (pbar := tqdm.tqdm(DATASETS)):
        pbar.set_description(f"Fetch {dataset} ...")
        data = fetch_dataset(dataset, download_if_missing=True)
        if len(data['train_triplets']) < 1:
            raise ValueError('Something got wrong.')
        with (DATA_HOME / f'{dataset}.json').open('w') as f:
            json.dump(data, f, indent=4)

def fetch_all(download_if_missing=False):
    return {
        dataset: fetch_dataset(dataset, download_if_missing)
        for dataset in DATASETS
    }


if __name__ == "__main__":
    print(f"Download datasets to {DATA_HOME} ...")
    download_all()