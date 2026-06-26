import os
import json
import torch
import random
import numpy as np

from typing import Union
from expigeo import EXPIGEO
from torch_geometric.data import Dataset, Data
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import add_self_loops
from torch.utils.data import random_split, Subset
from utils import FileHandler, GaussianCurvatureEstimator



class IntrABuilder:

  def __init__(self, 
               intra_root_dir: str, 
               loadmap: str):
    
    self.handler = FileHandler()

    if os.path.exists(loadmap):
      self.loadmap = loadmap
    else:
      self.handler.generate_intra_map(intra_root_dir, loadmap)
      self.loadmap = loadmap
      
    self.expigeo = EXPIGEO()
    self.files = self.get_intra_files(balance_ratio=1)
  
  def get_sample(self, idx: int, target_files: str):
    path = self.files[target_files][idx]
    if isinstance(path, tuple) and path[0].find('IntrA') != -1:
      data = self.handler.annotate_mesh_to_array(path[0], path[1])
    else:
      data = np.loadtxt(path)

    return data
  
  def explore_full_dataset(self, save_directory: str, 
                           target_files: str='balanced',
                           size: float = 1.0,
                           expigeo_params: Union[dict, str]=None):

    os.makedirs(save_directory,
                exist_ok=True)

    skipped = []

    possible_keys = ('balanced', 'positives', 'negatives',
                     'rem_negatives', 'all_files')
    
    tkey = target_files if target_files in possible_keys else 'balanced'

    files = self.files[tkey]
    files = files[:int(len(files) * size)]

    for idx, item in enumerate(files):

      filepath = files[idx][0]
      filename = filepath.split(os.sep)[-1].replace('.obj', '')

      if filepath.find('aneurysm') != -1 or filepath.find('annotated') != -1:
        subset = 'aneurysms'
      else:
        subset = 'arteries'

      os.makedirs(os.path.join(save_directory, subset), exist_ok=True)

      raw = self.get_sample(idx, tkey)

      print(f'Processing Sample ({idx+1}/{len(files)}): {filename}, Number of Points: {raw.shape[0]}')

      try:
        explored_pc = self.expigeo.explore(raw, params=expigeo_params)
        save_filepath = os.path.join(
            save_directory,
            subset,
            filepath.split(os.sep)[-1].replace('.obj', '.txt')
        )
        np.savetxt(save_filepath, 
                   explored_pc, 
                   delimiter=',')

        print(f'Saved Sample: {filename}\n')

      except Exception as e:
        skipped.append(f'{filename}')
        print(f'Skipped {filename} for the following exception:')
        print(e)
        continue

    if skipped:
      print('\n\nThe following files have been skipped due to runtime errors:\n')
      for item in skipped:
        print(item)
        
      
  def get_intra_files(self, balance_ratio: float=1):
    
    positives, negatives, all_files = [], [], []

    with open(self.loadmap, 'r') as fp:
      correspondences = json.load(fp)

    for item in correspondences:
      for pe in self.handler.zip_mesh_with_mask(item['aneurysm']['mesh'],
                                item['aneurysm']['mask']):
        positives.append(pe)
        all_files.append(pe)
      for ne in self.handler.zip_mesh_with_mask(item['vessel']['mesh'],
                                item['vessel']['mask']):
        negatives.append(ne)
        all_files.append(ne)

    if balance_ratio > 0:
      num_negative_samples = int(balance_ratio * len(positives))
    else:
      num_negative_samples = len(negatives)

    random.shuffle(negatives)

    sampled_negatives = negatives[:num_negative_samples]
    rem_negatives = negatives[num_negative_samples:]

    random.shuffle(all_files)

    files = {
        'balanced': [*positives, *sampled_negatives],
        'positives': positives,
        'negatives': negatives,
        'rem_negatives': rem_negatives,
        'all_files': all_files,
    }

    return files
  

class IntrA(Dataset):
  def __init__(self, 
               build_dir: str,
               graph_neighbors: int=32):

    self.handler = FileHandler()

    self.correspondences = []

    intra_positives = os.listdir(os.path.join(build_dir, 
                                              'aneurysms'))
    intra_negatives = os.listdir(os.path.join(build_dir,
                                               'arteries'))

    self.files = [*list(map(lambda x: os.path.join(build_dir,
                                                    'aneurysms', x),
                        intra_positives)),
              *list(map(lambda x: os.path.join(build_dir,
                                                'arteries', x),
                        intra_negatives))]
    
    self.k = graph_neighbors

    self.gcurv_estimator = GaussianCurvatureEstimator()

  def get_sample(self, idx: int,
                 drop_normals: bool=True, 
                 mask_only: bool=False,
                 xyz_normals_only: bool=False,
                 xyz_normals_gcurv_only: bool=False) -> np.ndarray:
    
    # xyz_normals_only and xyz_normals_gcurv_only are used for ablation purposes.

    filepath = self.files[idx]

    data = np.loadtxt(filepath, delimiter=',')

    if mask_only:
      data = data[:, -1]
    elif xyz_normals_only:
      data = data[:, :6]
    elif xyz_normals_gcurv_only:
      gcurv = self.gcurv_estimator.estimate_gaussian_curvature(data[:, :3], 
                                                               data[:, 3:6])
      data = np.concatenate([data, gcurv.reshape(-1, 1)], axis=1)

    if drop_normals and not mask_only:
      data = np.delete(data, [3, 4, 5], axis=1)

    return data

  def __getitem__(self, idx: int) -> Data:

    data = self.get_sample(idx, 
                           drop_normals=True, 
                           mask_only=False,
                           xyz_normals_only=False,
                           xyz_normals_gcurv_only=False)

    labels = torch.from_numpy(data[:, -1]).to(torch.long)

    X = torch.from_numpy(
      self.normalize(data[:, :-1])
    ).to(torch.float)

    edge_index = self.build_knn_graph(X[:, :3], 
                                      k=self.k)

    graph = Data(x=X, 
                 pos=X[:, :3],
                 edge_index=edge_index,
                 y=labels)

    return graph
  
  def normalize(self, data: np.ndarray) -> np.ndarray:
    centroid = np.mean(data, axis=0)
    data -= centroid
    furthest_distance = np.max(np.sqrt(np.sum(abs(data)**2,axis=-1)))
    data /= furthest_distance

    return data

  def build_knn_graph(self, points: torch.Tensor, 
                      k: int) -> torch.Tensor:
    points_np = points.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points_np)
    _, indices = nbrs.kneighbors(points_np)

    row, col = [], []
    for i in range(points_np.shape[0]):
        for j in indices[i][1:]:
            row.append(i)
            col.append(j)

    edge_index = torch.tensor([row, col], dtype=torch.long)

    edge_index, _ = add_self_loops(edge_index, num_nodes=points.size(0))

    return edge_index

  def __len__(self) -> int:
    return len(self.files)
  
  
def class_weights(split: Subset) -> torch.Tensor:
  all_labels = torch.cat([torch.tensor(split.dataset.get_sample(index, mask_only=True))
                          for index in range(len(split.indices))], dim=0)
  unique_labels, class_counts = torch.unique(all_labels, return_counts=True)
  total_samples = all_labels.shape[0]
  num_classes = len(unique_labels)
  if num_classes != len(class_counts):
      raise ValueError("The number of unique labels and class counts do not match.")
  class_counts_dict = {label.item(): count.item() for label, count in zip(unique_labels, class_counts)}
  weights = []
  for i in sorted(class_counts_dict.keys()):
      if class_counts_dict[i] == 0:
          weight = 1.0
      else:
          weight = total_samples / (num_classes * class_counts_dict[i])
      weights.append(weight)
  weights_tensor = torch.tensor(weights, dtype=torch.float)
  return weights_tensor


def serialize_data_state(train_split: Subset, 
                         val_split: Subset, 
                         test_split: Subset,
                         save_path: str):
  
  state = {
    'train': [train_split.dataset.files[idx] for idx in train_split.indices],
    'val': [val_split.dataset.files[idx] for idx in val_split.indices],
    'test': [test_split.dataset.files[idx] for idx in test_split.indices]
  }

  with open(save_path, 'w') as fp:
    json.dump(state, fp, indent=2)


def load_data_state(dataset: IntrA, 
                    state_path: str,
                    train_data_ratio: float, 
                    val_data_ratio: float,
                    test_data_ratio: float):
  
  with open(state_path, 'r') as fp:
    state = json.load(fp)

  dataset.files = [*state['train'], *state['val'], *state['test']]

  random.shuffle(dataset.files)

  train_split, val_split, test_split = random_split(dataset, (train_data_ratio,
                                                              val_data_ratio,
                                                              test_data_ratio))
  
  new_train_indices = []
  new_val_indices = []
  new_test_indices = []

  for f1 in state['train']:
    for idy, f2 in enumerate(dataset.files):
      if f1 == f2:
        new_train_indices.append(idy)

  for f1 in state['val']:
    for idy, f2 in enumerate(dataset.files):
      if f1 == f2:
        new_val_indices.append(idy)

  for f1 in state['test']:
    for idy, f2 in enumerate(dataset.files):
      if f1 == f2:
        new_test_indices.append(idy)

  train_split.indices = new_train_indices
  val_split.indices = new_val_indices
  test_split.indices = new_test_indices

  return train_split, val_split, test_split