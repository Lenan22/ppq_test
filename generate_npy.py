import numpy as np
import pdb
import os
from tqdm import tqdm

result_path = './data'

if __name__ == '__main__':
    for i in tqdm(range(500)):
        data = np.random.random((1,3,608,608))
        x = np.array(data, dtype=np.float32)
        new_path_name = os.path.join(result_path,str(i)) + '.npy'
        np.save(new_path_name,x)
