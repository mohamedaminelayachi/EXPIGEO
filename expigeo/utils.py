import os
import re
import json
import random
import numpy as np


class FileHandler:

    # converts a .obj mesh into a point cloud without normals.
    @staticmethod
    def convert_mesh_to_pc(mesh_path: str) -> np.ndarray:

        assert mesh_path.endswith('.obj'), f"{mesh_path} must be .obj file."
        if not os.path.exists(mesh_path):
            return None
        vertices = []
        with open(mesh_path, 'r') as file:
            for line in file.readlines():
                if line.startswith('v '):
                    parts = line.strip().split()
                    x, y, z = map(float, parts[1:4])
                    if len(parts) >= 7:
                        r, g, b = map(float, parts[4:7])
                    else:
                        r, g, b = 0.0, 0.0, 0.0
                    vertices.append((x, y, z, r, g, b))

        return np.array(vertices)

    # attaches the annotation mask of IntrA from .ad file in the .obj file.
    # the two files must correspond to the same mesh.
    def annotate_mesh_to_array(self, mesh_path: str, annotation_path: str):

        if not os.path.exists(mesh_path):
            return None
        point_cloud = self.convert_mesh_to_pc(mesh_path)
        with open(annotation_path, 'r') as fp:
            mask = fp.readlines()

        mask = [list(map(float, mask[i].split(' '))) for i in range(len(mask))]

        data = []
        for idx, vertex in enumerate(point_cloud):
            label = 0
            normals = [0, 0, 0]
            for idy, mask_coord in enumerate(mask):
                if (vertex[:3] == mask_coord[:3]).all():
                    if int(mask_coord[6]) in [1, 2]:
                        label = 1
                    mask.pop(idy)
                    normals = mask_coord[3:6]
                    break

            data.append([*vertex[:3],  *normals, label])

        return np.array(data)

    def map_similar_files(self, 
                          filename: str, intra_root_dir: str,
                        search_for_complete: bool=True):
        if not os.path.exists(intra_root_dir):
            raise FileExistsError
        temp_filename = filename.replace('_', '-')
        fid = int(re.findall(r'\d+', temp_filename.split('-')[0])[0])

        annotated_files = {
            "mask": [*os.listdir(os.path.join(intra_root_dir, 'annotated', 'ad'))],
            "mesh": [*os.listdir(os.path.join(intra_root_dir, 'annotated', 'obj'))]
        }
        generated_aneurysm_files = {
            "mask": [*os.listdir(os.path.join(intra_root_dir, 'generated', 'aneurysm', 'ad'))],
            "mesh": [*os.listdir(os.path.join(intra_root_dir, 'generated', 'aneurysm', 'obj'))]
        }
        generated_vessel_files = {
            "mask": [*os.listdir(os.path.join(intra_root_dir, 'generated', 'vessel', 'ad'))],
            "mesh": [*os.listdir(os.path.join(intra_root_dir, 'generated', 'vessel', 'obj'))]
        }

        all_correspondences = {
            'aneurysm': {
                'mask': [],
                'mesh': []
            },
            'vessel': {
                'mask': [],
                'mesh': []
            },
            'complete': []
        }

        for anf in [*annotated_files['mask'], *annotated_files['mesh']]:
            temp = anf.replace('_', '-').replace('.', '-').split('-')[0]
            if re.match(r'AN(\d+)', temp):
                index = int(temp.replace('AN', ''))
            if index == fid and anf.endswith('.ad'):
                all_correspondences['aneurysm']['mask'].append(
                    os.path.join(intra_root_dir, 'annotated', 'ad', anf)
                )
            if index == fid and anf.endswith('.obj'):
                all_correspondences['aneurysm']['mesh'].append(
                    os.path.join(intra_root_dir, 'annotated', 'obj', anf)
                )

        for gaf in [*generated_aneurysm_files['mask'], *generated_aneurysm_files['mesh']]:
            temp = gaf.replace('_', '-').replace('.', '-').split('-')[0]
            if re.match(r'ArteryObjAN(\d+)', temp):
                index = int(temp.replace('ArteryObjAN', ''))
            if index == fid and gaf.endswith('.ad'):
                all_correspondences['aneurysm']['mask'].append(
                    os.path.join(intra_root_dir, 'generated', 'aneurysm', 'ad', gaf)
                )
            if index == fid and gaf.endswith('.obj'):
                all_correspondences['aneurysm']['mesh'].append(
                    os.path.join(intra_root_dir, 'generated', 'aneurysm', 'obj', gaf)
                )

        for gvf in [*generated_vessel_files['mask'], *generated_vessel_files['mesh']]:
            temp = gvf.replace('_', '-').replace('.', '-').split('-')[0]
            if re.match(r'ArteryObjAN(\d+)', temp):
                index = int(temp.replace('ArteryObjAN', ''))
            if index == fid and gvf.endswith('.ad'):
                all_correspondences['vessel']['mask'].append(
                    os.path.join(intra_root_dir, 'generated', 'vessel', 'ad', gvf)
                )
            if index == fid and gvf.endswith('.obj'):
                all_correspondences['vessel']['mesh'].append(
                    os.path.join(intra_root_dir, 'generated', 'vessel', 'obj', gvf)
                )

        if search_for_complete:
            for complete in os.listdir(os.path.join(intra_root_dir, 'complete')):
                temp = complete.replace('_', '-').replace('.', '-').split('-')[0]
            if re.match(r'ArteryObjAN(\d+)', temp):
                index = int(temp.replace('ArteryObjAN', ''))
                if index == fid:
                    all_correspondences['complete'].append(
                        os.path.join(intra_root_dir, 'complete', complete)
                    )
        else:
            all_correspondences['complete'].append(
                os.path.join(intra_root_dir, 'complete', filename)
            )

        return all_correspondences
    
    def generate_intra_map(self, intra_root_dir: str, savepath: str):
        if not os.path.exists(intra_root_dir):
            raise FileExistsError

        new_structure = [self.map_similar_files(file, intra_root_dir, False)
                    for file in sorted(os.listdir(os.path.join(intra_root_dir, 'complete')),
                                        key=lambda file: int(file.split('.')[0].replace('ArteryObjAN', '')))]

        with open(savepath, 'w') as fp:
            json.dump(new_structure, fp, indent=4)

        return new_structure
    

    def get_files(self, 
                  explored_dir: str, 
                  balance_ratio: float=1):
        
        positives, negatives, all_files = [], [], []

        for item in self.correspondences:
            for pe in self.zip_mesh_with_mask(item['aneurysm']['mesh'],
                                    item['aneurysm']['mask']):
                filename = pe[0].split('/')[-1].replace('.obj', '.txt')
                if os.path.exists(os.path.join(explored_dir, filename)):
                    positives.append(pe)
                    all_files.append(pe)
            for ne in self.zip_mesh_with_mask(item['vessel']['mesh'],
                                    item['vessel']['mask']):
                filename = ne[0].split('/')[-1].replace('.obj', '.txt')
                if os.path.exists(os.path.join(explored_dir, filename)):
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

    @staticmethod
    def zip_mesh_with_mask(meshes: list, masks: list):
        zipped = []
        for mesh in meshes:
            if mesh.find('annotated') != -1:
                mesh_id = mesh.split(os.sep)[-1].replace('_full', '').replace('.obj', '')
                for mask in masks:
                    if mask.endswith('_norm.ad') and mask.find(mesh_id) != -1:
                        zipped.append((mesh, mask))
            elif mesh.find('generated') != -1:
                mesh_id = mesh.split(os.sep)[-1].replace('.obj', '')
                for mask in masks:
                    if mask.endswith('.ad') and mask.split(os.sep)[-1].replace('.ad', '') == mesh_id:
                        zipped.append((mesh, mask))

        return zipped

    @staticmethod
    def check_map(map_filepath: str):
        if os.path.exists(map_filepath):
            raise map_filepath

        with open(map_filepath, 'r') as fp:
            correspondences = json.load(fp)

        flag = True
        for entry in correspondences:
            if not ({'aneurysm', 'vessel', 'complete'} <= entry.keys()):
                flag = False
                break
            elif not ({'mask', 'mesh'} <= entry['aneurysm'].keys()):
                flag = False
                break
            elif not ({'mask', 'mesh'} <= entry['vessel'].keys()):
                flag = False
                break

        return flag
    


class GaussianCurvatureEstimator:
  
  # This class implements the discrete case (because it's on point clouds) 
  # of Gaussian Curvature, which is used for ablations (see paper).
  # The implementation is outside the scope of the paper. Thus, to understand 
  # it in depth, I (Mohamed Amine) recommend the following resources:
  # 10.1145/3409501.3409505
  # Gaussian and Mean Curvatures, (COMS 4770/5770 Notes), Yan-Bin Jia
  # Gaussian Curvature and The Gauss-Bonnet Theorem (Bachelor’s thesis), O. Jaibi

  def _tangent_basis(self, n):
    n = n / (np.linalg.norm(n) + 1e-12)
    if abs(n[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(helper, n)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)
    return u, v

  def estimate_gaussian_curvature(self, points, normals, k=50, reg=1e-8):
      points = np.asarray(points, dtype=float)
      normals = np.asarray(normals, dtype=float)
      N = points.shape[0]
      if points.shape[1] != 3 or normals.shape[1] != 3:
          raise ValueError("points and normals must be (N,3) arrays")

      try:
          from scipy.spatial import cKDTree as KDTree
          tree = KDTree(points)
          kq = min(k, N)
          dists, idx = tree.query(points, k=kq)
      except Exception:
          if N > 20000:
              raise RuntimeError("Install scipy for kNN on large clouds.")
          D2 = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
          idx = np.argsort(D2, axis=1)[:, :min(k, N)]

      if idx.ndim == 1:
          idx = idx[:, None]

      kq = idx.shape[1]

      k1 = np.zeros(N, dtype=float)
      k2 = np.zeros(N, dtype=float)
      K  = np.zeros(N, dtype=float)

      for i in range(N):
          neigh_idx = idx[i]
          pj = points[neigh_idx]
          nj = normals[neigh_idx]
          pi = points[i]
          ni = normals[i]

          u, v = self._tangent_basis(ni)

          vij = pj - pi
          t1 = vij.dot(u)
          t2 = vij.dot(v)

          dn = nj - ni
          dn_u = dn.dot(u)
          dn_v = dn.dot(v)

          t_norm2 = t1**2 + t2**2
          valid = t_norm2 > 1e-12
          if np.count_nonzero(valid) < 3:
              k1[i] = 0.0
              k2[i] = 0.0
              K[i]  = 0.0
              continue

          t1v = t1[valid]
          t2v = t2[valid]
          dnu = dn_u[valid]
          dnv = dn_v[valid]

          m = t1v.shape[0]
          A = np.zeros((2*m, 3), dtype=float)
          b = np.zeros((2*m,), dtype=float)
          A[:m, 0] = -t1v
          A[:m, 1] = -t2v
          b[:m] = dnu

          A[m:, 1] = -t1v
          A[m:, 2] = -t2v
          b[m:] = dnv

          ATA = A.T @ A
          ATb = A.T @ b
          ATA += reg * np.eye(3)
          try:
              x = np.linalg.solve(ATA, ATb)
          except np.linalg.LinAlgError:
              x, *_ = np.linalg.lstsq(A, b, rcond=None)

          S11, S12, S22 = x
          S = np.array([[S11, S12],
                        [S12, S22]], dtype=float)

          w, _ = np.linalg.eig(S)
          w = np.real(w)
          w = np.sort(w)[::-1]
          k1[i] = w[0]
          k2[i] = w[1]
          K[i]  = k1[i] * k2[i]

      return np.max([k1, k2, K], axis=0)