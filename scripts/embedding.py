import argparse
import pandas
import json
import time


parser = argparse.ArgumentParser()
parser.add_argument('algo', type=str, help='The embedding algorithm.')
parser.add_argument('dataset', type=str, help='The dataset file.')
parser.add_argument('result', type=str, help='The result file.')
args = parser.parse_args()

algo = args.algo
dataset_file = args.dataset
result_file = args.result


with open(dataset_file, 'r') as f:
    dataset = json.load(f)
train_triplets = pandas.read_csv(dataset['train_triplets'])

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
        estimator = embedding.FORTE(dims, backend=backend, device=device, kernel=is_kernel)
    case 'GNMDS':
        estimator = embedding.GNMDS(dims, backend=backend, device=device, kernel=is_kernel)
    case 'SOE':
        estimator = embedding.SOE(dims, n_init=1, backend=backend, device=device)
    case 'STE':
        estimator = embedding.STE(dims, backend=backend, device=device)
    case 'tSTE':
        estimator = embedding.TSTE(dims, backend=backend, device=device)
    else:
        raise ValueError(f"Unexpected algorithm {algo}.")

start_time = time.process_time()
estimator.fit(train_triplets)
end_time = time.process_time()
embedding = estimator.embedding_
loss = estimator.stress_
### END CBLEARN

result = {'dataset': dataset_file,
          'library': library,
          'algorithm': algo,
          'loss': loss,
          'cpu_time': end_time - start_time,
          'embedding': embedding.tolist()}
print(f"Save results to {result_file} ...")
json.dump(result, result_file, indent=2)
