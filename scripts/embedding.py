import os

from pathlib import Path
import argparse
import json
import time
import numpy as np


DATA_HOME = Path(os.environ.get('CBLEARN_DATA', Path(__file__).parent / '../datasets'))


parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, help='The embedding algorithm.')
parser.add_argument('dataset', type=str, help='The dataset file.')
#parser.add_argument('result', type=str, help='The result file.')
args = parser.parse_args()
algo = args.algo
dataset =  args.dataset
dims = 2

dataset_file = (DATA_HOME / dataset).with_suffix('.json')
result_file = Path(__file__).parent / f'../results/python_{algo}_{dataset}_{int(time.time())}.json'

with dataset_file.open('r') as f:
    data = json.load(f)

train_triplets = np.array(data['train_triplets'])

### CBLEARN
library = 'cblearn'
from cblearn import embedding
algo_parts = algo.split('-')
is_kernel = 'K' in algo_parts
if 'GPU' in algo_parts:
    backend = 'torch'
    device = 'cuda'
else:
    backend = 'scipy'
    device = 'cpu'

match algo_parts[0]:
    case 'MLDS':
        estimator = embedding.MLDS(dims)  # should raise error for dims != 1
    case 'CKL':
        estimator = embedding.CKL(dims, backend=backend, device=device, kernel=is_kernel)
    case 'FORTE':
        if backend != 'torch':
            raise ValueError(f"FORTE does not support multiple backends at the moment.")
        if is_kernel:
            raise ValueError(f"FORTE does not support kernel at the moment.")
        estimator = embedding.FORTE(dims)
    case 'GNMDS':
        estimator = embedding.GNMDS(dims, backend=backend, device=device, kernel=is_kernel)
    case 'SOE':
        estimator = embedding.SOE(dims, n_init=1, backend=backend, device=device)
    case 'STE':
        estimator = embedding.STE(dims, backend=backend, device=device)
    case 'tSTE':
        estimator = embedding.TSTE(dims, backend=backend, device=device)
    case _:
        raise ValueError(f"Unexpected algorithm {algo}.")

start_time = time.process_time()
estimator.fit(train_triplets)
end_time = time.process_time()
print(end_time - start_time)
embedding = estimator.embedding_
loss = estimator.stress_
### END CBLEARN

result = {'dataset': dataset,
          'library': library,
          'algorithm': algo,
          'loss': float(loss),
          'cpu_time': float(end_time - start_time),
          'embedding': embedding.tolist()}
print(f"Save results to {result_file} ...")
with result_file.open('w') as f:
    json.dump(result, f, indent=4)
