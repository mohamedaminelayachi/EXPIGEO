import numpy as np
import multiprocessing
import heapq
import yaml

from functools import partial
from typing import Tuple, Union
from scipy.spatial import KDTree, distance
from collections import defaultdict
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import minimum_spanning_tree
from typing import Union
from utils import FileHandler


class EXPIGEO:

    # used to sample the (random) directions of the rays.
    @staticmethod
    def sample_cone_directions(d0: np.ndarray, 
                               theta_max: float,
                                 K: int) -> np.ndarray:
        w = d0 / np.linalg.norm(d0)

        if np.abs(w[2]) > 0.9:
            a = np.array([0.0, 1.0, 0.0])
        else:
            a = np.array([0.0, 0.0, 1.0])

        u = np.cross(a, w)
        u = u / np.linalg.norm(u)
        v = np.cross(w, u)

        u_j = np.random.rand(K)
        phi_j = np.random.rand(K) * 2 * np.pi

        cos_theta_max = np.cos(theta_max)
        cos_theta_j = 1.0 - u_j * (1.0 - cos_theta_max)
        theta_j = np.arccos(cos_theta_j)
        sin_theta_j = np.sin(theta_j)

        d_x = sin_theta_j * np.cos(phi_j)
        d_y = sin_theta_j * np.sin(phi_j)
        d_z = cos_theta_j

        directions = (d_x[:, np.newaxis] * u + d_y[:, np.newaxis] * v + d_z[:, np.newaxis] * w)

        norm_directions = directions / np.linalg.norm(directions, axis=1,
                                                    keepdims=True)

        return norm_directions
    
    # used to sample the cones (including their rays) for all points. 
    def sample_initial_cones(self, directions: np.ndarray, theta_max_rad: float,
                         K: int) -> np.ndarray:
        N = directions.shape[0]
        all_directions = np.zeros((N, K, 3))
        for i in range(N):
            all_directions[i] = self.sample_cone_directions(directions[i], theta_max_rad, K)
        return all_directions
    
    # used to find points that (approximately) intersects with the source point given a direction.
    @staticmethod
    def get_ray_candidate_indices(tree: KDTree, q: np.ndarray, d: np.ndarray,
                              max_range: float, step_size: float) -> np.ndarray:
        n_steps = int(max_range / step_size) + 1
        t_steps = np.linspace(0, max_range, n_steps)
        ray_samples = q + t_steps[:, np.newaxis] * d
        search_radius = step_size * 2.0
        nearby_indices_list = tree.query_ball_point(ray_samples, r=search_radius)
        return np.unique(np.concatenate(nearby_indices_list).astype(int))
    

    # used to cast a single ray from the source point given a direction.
    def cast_single_ray(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        tree: KDTree,
        all_points: np.ndarray,
        all_normals: np.ndarray,
        delta: float,
        max_range: float
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.isnan(origin[0]) or np.isnan(direction[0]):
            return (np.array([np.nan]*3), np.array([np.nan]*3))

        q = origin
        d_j = direction
        ray_step_size = delta
        t_min_epsilon = 1e-6
        denom_epsilon = 1e-6

        candidate_indices = self.get_ray_candidate_indices(
            tree, q, d_j, max_range, ray_step_size
        )

        if candidate_indices.size == 0:
            return (np.array([np.nan]*3), np.array([np.nan]*3))

        candidate_points = all_points[candidate_indices]
        candidate_normals = all_normals[candidate_indices]

        p_minus_q = candidate_points - q
        dot_p_n = np.sum(p_minus_q * candidate_normals, axis=1)
        dot_d_n = np.dot(candidate_normals, d_j)

        valid_denom_mask = np.abs(dot_d_n) > denom_epsilon
        t_i = np.full(candidate_indices.size, np.inf)
        t_i[valid_denom_mask] = dot_p_n[valid_denom_mask] / dot_d_n[valid_denom_mask]

        valid_t_mask = t_i > t_min_epsilon
        x_i = q + t_i[:, np.newaxis] * d_j
        dist_to_p_i = np.linalg.norm(x_i - candidate_points, axis=1)
        valid_prox_mask = dist_to_p_i < delta

        final_valid_mask = valid_denom_mask & valid_t_mask & valid_prox_mask
        valid_ts = t_i[final_valid_mask]

        if valid_ts.size > 0:
            best_t_idx_in_valid = np.argmin(valid_ts)
            t_star = valid_ts[best_t_idx_in_valid]
            original_indices = candidate_indices[final_valid_mask]
            winning_p_index = original_indices[best_t_idx_in_valid]

            hit_point = q + t_star * d_j
            hit_normal = all_normals[winning_p_index]
            return (hit_point, hit_normal)

        # no hit found
        return (np.array([np.nan]*3), np.array([np.nan]*3))
    
    # used to compute the average spacing between points.
    @staticmethod
    def get_avg_spacing(points: np.ndarray, 
                        k: int = 10) -> float:
        if points.shape[0] <= k:
            k = 1
        tree = KDTree(points[:, :3])
        distances, _ = tree.query(points[:, :3], k=k+1)
        neighbor_distances = distances[:, 1:]
        avg_spacing = np.mean(neighbor_distances)
        return avg_spacing
    
    # main ray probing function
    def probe(self,
            point_cloud: np.ndarray, 
            K: int, theta_max_deg: float,
            gamma: float=0.33) -> list:

        N = point_cloud.shape[0]
        points, normals = point_cloud[:, :3], point_cloud[:, 3:6]
        delta = gamma * self.get_avg_spacing(points)

        tree = KDTree(points)
        min_coords, max_coords = points.min(axis=0), points.max(axis=0)
        max_range = np.linalg.norm(max_coords - min_coords)

        q_points = points
        q_normals = normals
        theta_max_rad = np.radians(theta_max_deg)

        origins = np.stack([q_points] * K, axis=1)
        directions = self.sample_initial_cones(-q_normals, theta_max_rad, K)

        origins_flat = origins.reshape(-1, 3)
        directions_flat = directions.reshape(-1, 3)

        job_args = list(zip(origins_flat, directions_flat))

        partial_worker = partial(
            self.cast_single_ray,
            tree=tree,
            all_points=points,
            all_normals=normals,
            delta=delta,
            max_range=max_range
        )

        with multiprocessing.Pool() as pool:
            results_flat = list(pool.starmap(partial_worker, job_args))

        hits_flat = np.array([res[0] for res in results_flat])

        hits = hits_flat.reshape(N, K, 3)

        path_history = [origins, hits]

        return path_history
    
    # used to compute the centerline, radius field with gradient magnitude.
    def compute_centerline_and_radius(
        self,
        probed: list,
        dist_variance_percentile: float = 90.0,
        k_neighbors=10,
        std_multiplier=0.03,
        min_hits: int = 5,
        rgm_neighbors: int=50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        origins, hits = probed[0], probed[1]

        N = origins.shape[0]

        pre_filter_candidates = []
        pre_filter_dist_variances = []
        pre_filter_radii = []

        for q_idx in range(N):
            q_point = origins[q_idx, 0, :]
            valid_hits = hits[q_idx, ~np.isnan(hits[q_idx]).any(axis=1)]
            if valid_hits.shape[0] > min_hits:
                # eq (1) from the paper.
                x_median_i = np.median(valid_hits, axis=0)
                c_i = (q_point + x_median_i) / 2.0
                hit_distances = np.linalg.norm(valid_hits - q_point, axis=1)
                dist_variance = np.std(hit_distances)
                mean_radius = np.median(hit_distances) / 2.0

                pre_filter_candidates.append(c_i)
                pre_filter_dist_variances.append(dist_variance)
                pre_filter_radii.append(mean_radius)

        if not pre_filter_candidates:
            return np.array([]), np.array([])

        candidate_points_arr = np.array(pre_filter_candidates)
        dist_variances_arr = np.array(pre_filter_dist_variances)

        radii_arr = np.array(pre_filter_radii)
        variance_threshold = np.percentile(dist_variances_arr,
                                        dist_variance_percentile)

        keep_mask = dist_variances_arr < variance_threshold

        candidate_points = candidate_points_arr[keep_mask]
        radii = radii_arr[keep_mask]

        tree = KDTree(candidate_points)
        distances, _ = tree.query(candidate_points, k=k_neighbors + 1)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        global_mean_dist = np.mean(avg_distances)
        global_std_dist = np.std(avg_distances)
        distance_threshold = global_mean_dist + std_multiplier * global_std_dist

        keep_mask = avg_distances < distance_threshold

        centerline = candidate_points[keep_mask]
        radius_field = radii[keep_mask]
        radius_gradient_mag = self.compute_radius_gradient_magnitude(centerline,
                                                                     radius_field,
                                                                     k_neighbors=rgm_neighbors)


        return centerline, radius_field, radius_gradient_mag
    
    # used to merge close-by vertices.
    @staticmethod
    def merge_close_vertices(vertices, edges, merge_factor: float=1.0):
        edge_lengths = np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]],
            axis=1
        )

        base_scale = np.median(edge_lengths)

        tol = merge_factor * base_scale

        if len(vertices) == 0:
            return vertices, edges

        vertices = np.array(vertices)
        edges = np.array(edges)

        M = vertices.shape[0]

        parent = np.arange(M)

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_j] = root_i

        tree = KDTree(vertices)
        pairs = tree.query_pairs(tol)

        for i, j in pairs:
            union(i, j)

        groups = defaultdict(list)
        for i in range(M):
            root = find(i)
            groups[root].append(i)

        new_vertices_list = []
        old_to_new_map = np.zeros(M, dtype=int)

        for root_idx, component_indices in groups.items():
            cluster_points = vertices[component_indices]
            mean_position = np.mean(cluster_points, axis=0)
            new_id = len(new_vertices_list)
            new_vertices_list.append(mean_position)
            old_to_new_map[component_indices] = new_id

        new_vertices = np.array(new_vertices_list)

        if len(edges) > 0:
            new_edges_mapped = old_to_new_map[edges]

            valid_mask = new_edges_mapped[:, 0] != new_edges_mapped[:, 1]
            new_edges_cleaned = new_edges_mapped[valid_mask]

            new_edges_cleaned = np.sort(new_edges_cleaned, axis=1)

            new_edges = np.unique(new_edges_cleaned, axis=0)
        else:
            new_edges = np.empty((0, 2), dtype=int)

        return new_vertices, new_edges
    
    # used to compute the gradient magnitude of radii
    @staticmethod
    def compute_radius_gradient_magnitude(
        centerline: np.ndarray,
        radii: np.ndarray,
        k_neighbors: int = 50
    ) -> np.ndarray:
        if centerline.shape[0] == 0:
            return np.array([])

        if centerline.shape[0] <= k_neighbors:
            k_neighbors = centerline.shape[0] - 1
            if k_neighbors <= 0:
                return np.zeros(centerline.shape[0])

        tree = KDTree(centerline)
        distances, indices = tree.query(centerline, k=k_neighbors + 1)

        M = centerline.shape[0]
        gradient_magnitudes = np.zeros(M)

        for i in range(M):
            p_i = centerline[i]
            r_i = radii[i]

            neighbor_indices = indices[i, 1:]
            neighbor_points = centerline[neighbor_indices]
            neighbor_radii = radii[neighbor_indices]
            neighbor_distances = distances[i, 1:]
            sigma = np.mean(neighbor_distances)

            if sigma == 0:
                sigma = 1e-6

            weights = np.exp(-0.5 * (neighbor_distances ** 2) / (sigma ** 2))
            W_sqrt = np.sqrt(weights)

            A = neighbor_points - p_i
            b = neighbor_radii - r_i

            A_weighted = A * W_sqrt[:, np.newaxis]
            b_weighted = b * W_sqrt

            try:
                g, _, _, _ = np.linalg.lstsq(A_weighted, b_weighted, rcond=None)
                gradient_magnitudes[i] = np.linalg.norm(g)
            except np.linalg.LinAlgError:
                gradient_magnitudes[i] = 0.0

        return gradient_magnitudes

    # used to extract the skeleton from the centerline.
    def extract_skeleton(
        self,
        centerline: np.ndarray,
        voxel_size_multiplier: float = 6.0,
        prune_multiplier: float = 0.6,
        density_percentile: float = 10.0,
        factor: float=0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        if centerline.shape[0] < 20:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 2), dtype=int)
            )

        pruning_threshold = 0.0

        num_samples = min(2000, centerline.shape[0])
        sample_indices = np.random.choice(centerline.shape[0], num_samples, replace=False)
        sample_points = centerline[sample_indices]
        tree = KDTree(sample_points)
        k = min(10, num_samples - 1)

        if k <= 0:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 2), dtype=int)
            )

        distances, _ = tree.query(sample_points, k=k+1)
        median_k_distance = np.median(distances[:, k])

        voxel_size = median_k_distance * voxel_size_multiplier
        if voxel_size == 0:
            diag = np.linalg.norm(centerline.max(axis=0) - centerline.min(axis=0))
            if diag == 0 and centerline.shape[0] > 0:
                return (
                    np.array([centerline[0]]),
                    np.empty((0, 2), dtype=int)
                )
            voxel_size = diag * 0.05 if diag > 0 else 1.0

        pruning_threshold = voxel_size * prune_multiplier

        min_bound = centerline.min(axis=0)
        voxel_indices = np.floor((centerline - min_bound) / voxel_size).astype(int)

        voxel_groups = {}
        for i in range(centerline.shape[0]):
            voxel_key = tuple(voxel_indices[i])
            if voxel_key not in voxel_groups:
                voxel_groups[voxel_key] = []
            voxel_groups[voxel_key].append(centerline[i])

        nodes_list = []
        densities_list = []
        for points in voxel_groups.values():
            if len(points) > 0:
                nodes_list.append(np.median(points, axis=0))
                densities_list.append(len(points))

        nodes = np.array(nodes_list)
        node_densities = np.array(densities_list)

        if nodes.shape[0] < 2:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 2), dtype=int)
            )

        dist_matrix = distance.cdist(nodes, nodes)
        mst_sparse = minimum_spanning_tree(dist_matrix)
        adj_matrix = mst_sparse + mst_sparse.T

        density_threshold = np.percentile(node_densities, density_percentile)
        adj_matrix_prune = adj_matrix.tolil()

        pruning_iterations = 0

        while True:
            pruning_iterations += 1
            adj_csr = adj_matrix_prune.tocsr()
            degrees = np.diff(adj_csr.indptr)
            leaves = np.where(degrees == 1)[0]

            if leaves.size == 0:
                break

            edges_to_remove = []
            for leaf in leaves:
                neighbor = adj_csr.indices[adj_csr.indptr[leaf]]
                weight = adj_matrix_prune[leaf, neighbor]
                leaf_density = node_densities[leaf]

                if weight < pruning_threshold and leaf_density < density_threshold:
                    edges_to_remove.append((leaf, neighbor))

            if not edges_to_remove:
                break

            for u, v in edges_to_remove:
                if adj_matrix_prune[u, v] > 0:
                    adj_matrix_prune[u, v] = 0.0
                    adj_matrix_prune[v, u] = 0.0

            if pruning_iterations > 100:
                break

        adj_matrix_final = adj_matrix_prune.tocsr()
        adj_matrix_final.eliminate_zeros()

        n_comp, labels = connected_components(
            adj_matrix_final, directed=False, connection='weak'
        )

        if n_comp == 0 or adj_matrix_final.nnz == 0:
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 2), dtype=int)
            )

        component_sizes = np.bincount(labels)
        small_components = np.where(component_sizes < 3)[0]
        keep_mask = ~np.isin(labels, small_components)

        if not np.any(keep_mask):
            return (
                np.empty((0, 3), dtype=float),
                np.empty((0, 2), dtype=int)
            )

        keep_indices = np.where(keep_mask)[0]

        vertices = nodes[keep_indices]

        final_adj_csr = adj_matrix_final[keep_indices, :][:, keep_indices]

        coo_graph_upper = final_adj_csr.tocoo()
        upper_triangle_mask = coo_graph_upper.row < coo_graph_upper.col
        edges = np.stack([
            coo_graph_upper.row[upper_triangle_mask],
            coo_graph_upper.col[upper_triangle_mask]
        ], axis=1).astype(int)

        merged_vertices, merged_edges = self.merge_close_vertices(vertices, edges, factor)

        return merged_vertices, merged_edges
    

    def compute_geometric_descriptors(self, vertices, edges, point_cloud,
                           centerline, radius_field, radius_grad_mag,
                           aperture_deg=15.0, decay_sigma=0.5, smoothing_radius=1.5) -> dict:
        
        M = len(vertices)
        N = len(point_cloud)

        skeleton_tree = KDTree(vertices)
        surface_tree = KDTree(point_cloud)
        centerline_tree = KDTree(centerline)

        # voronoi partitioning
        _, nearest_skeleton_indices = skeleton_tree.query(point_cloud, k=1)

        adj = {i: [] for i in range(M)}
        for u, v in edges:
            dist = np.linalg.norm(vertices[u] - vertices[v])
            adj[u].append((v, dist))
            adj[v].append((u, dist))

        dists_to_leaf = np.full(M, np.inf)
        pq = []

        leaf_nodes = []
        for i in range(M):
            degree = len(adj[i])
            if degree == 1:
                dists_to_leaf[i] = 0.0
                leaf_nodes.append(i)
                heapq.heappush(pq, (0.0, i))

        while pq:
            d, curr = heapq.heappop(pq)
            if d > dists_to_leaf[curr]: continue

            for neighbor, weight in adj[curr]:
                new_dist = d + weight
                if new_dist < dists_to_leaf[neighbor]:
                    dists_to_leaf[neighbor] = new_dist
                    heapq.heappush(pq, (new_dist, neighbor))

        # Eq (4) from the paper
        vertex_term_probs = np.exp(-(dists_to_leaf**2) / (2 * decay_sigma**2))

        vertex_blockage = np.zeros(M, dtype=int)
        vertex_sphericity = np.zeros(M, dtype=float)
        vertex_radii_mean = np.zeros(M, dtype=float)
        vertex_grad_mean = np.zeros(M, dtype=float)

        vertex_flow_vectors = np.zeros((M, 3))

        cos_threshold = np.cos(np.radians(aperture_deg))
        d_min_surface, _ = surface_tree.query(vertices, k=1)
        inner_cluster_idxs = centerline_tree.query_ball_point(vertices, smoothing_radius)

        for i in range(M):
            current_v = vertices[i]
            neighbors = adj[i]

            # flow direction
            direction = np.zeros(3)
            if not neighbors:
                direction = np.array([0,0,1])
            elif len(neighbors) == 1:
                # if leaf then the direction is outward.
                direction = current_v - vertices[neighbors[0][0]]
            else:
                # otherwise, the direction is downstream (towards the leaf)
                best = min(neighbors, key=lambda x: dists_to_leaf[x[0]])
                direction = vertices[best[0]] - current_v

            norm_dir = np.linalg.norm(direction).astype(np.float32)
            direction = direction.astype(np.float32)
            
            if norm_dir > 1e-9:
                direction /= norm_dir
            vertex_flow_vectors[i] = direction
            
            # take the mean radius (and gradient magnitude) w.r.t the centerline.
            c_neighbors = inner_cluster_idxs[i]
            if c_neighbors:
                vertex_radii_mean[i] = np.mean(radius_field[c_neighbors])
                vertex_grad_mean[i] = np.mean(radius_grad_mag[c_neighbors])
            else:
                _, idx = centerline_tree.query(current_v, k=1)
                vertex_radii_mean[i] = radius_field[idx]
                vertex_grad_mean[i] = radius_grad_mag[idx]

            search_radius = max(d_min_surface[i] * smoothing_radius, 1e-3)
            analysis_idxs = surface_tree.query_ball_point(current_v, search_radius)

            if not analysis_idxs: 
                _, analysis_idxs = surface_tree.query(current_v, k=5)

            analysis_pts = point_cloud[analysis_idxs]

            vertex_sphericity[i] = self.compute_geometric_sphericity(analysis_pts, current_v)

            # Blockage
            if len(neighbors) > 1:
                vertex_blockage[i] = 0
            else:
                vecs = analysis_pts - current_v
                # flow direction
                v_norms = np.linalg.norm(vecs, axis=1)
                valid = v_norms > 1e-6

                if np.any(valid):
                    unit_vecs = vecs[valid] / v_norms[valid][:, None]
                    dots = np.dot(unit_vecs, direction)
                    # condition from the paper.
                    if np.max(dots) >= cos_threshold:
                        vertex_blockage[i] = 1

        # broadcasting from features per vertex to features per point.
        
        parent_coords = vertices[nearest_skeleton_indices]

        rel_vecs = point_cloud - parent_coords
        parent_flows = vertex_flow_vectors[nearest_skeleton_indices]
        proj_flow = np.sum(rel_vecs * parent_flows, axis=1)

        return {
            'term_likelihood': vertex_term_probs[nearest_skeleton_indices],
            'blockage': vertex_blockage[nearest_skeleton_indices],
            'geo_sphericity': vertex_sphericity[nearest_skeleton_indices],
            'radius_field': vertex_radii_mean[nearest_skeleton_indices],
            'radii_grad_mag': vertex_grad_mean[nearest_skeleton_indices],
            'proj_flow': proj_flow
        }
    
    # used to calculate the geometric sphericity given a vertex from
    # the skeleton and its neighbors from the surface point cloud.
    @staticmethod
    def compute_geometric_sphericity(point_cluster,
                                      skeleton_vertex) -> float:
        if len(point_cluster) < 6: return 0.0

        centered = point_cluster - skeleton_vertex

        dists_sph = np.linalg.norm(centered, axis=1)
        err_sph = np.std(dists_sph)

        try:
            _, _, vt = np.linalg.svd(centered)
        except np.linalg.LinAlgError:
            return 0.5

        candidate_axes = [vt[0], vt[2]]
        best_err_cyl = np.inf

        for axis in candidate_axes:
            cross_prods = np.cross(centered, axis)
            dists_from_axis = np.linalg.norm(cross_prods, axis=1)
            curr_err = np.std(dists_from_axis)
            if curr_err < best_err_cyl:
                best_err_cyl = curr_err
        
        # Eq (5) from the paper.
        geo_sphericity = float(best_err_cyl / (best_err_cyl + err_sph + 1e-9))

        return geo_sphericity
    

    def explore(self,
                point_cloud: Union[np.ndarray, str], 
                params: Union[dict, str]=None):

        if isinstance(point_cloud, np.ndarray):
            assert point_cloud.shape[1] == 7, f"Point cloud must have coordinates, normals, and segmentation mask."
        elif isinstance(point_cloud, str):
            assert point_cloud.endswith('.txt'), "Point cloud be a path to numpy text file with delimiter=','."
        
        mask = point_cloud[:, 6]
        point_cloud = point_cloud[:, :6]

        default_params = {
            'K': 8,
            'theta': 10.0,
            'gamma': 0.33,
            'dist_var_perc': 90,
            'min_valid_hits': 3,
            'k_neighbors': 10,
            'std_multiplier': 0.03,
            'local_rgm_points': 50,
            'skeleton_voxel_size': 10,
            'skeleton_pruning_multiplier': 0.4,
            'skeleton_dense_perc': 90,
            'skeleton_merge_factor': 0.7,
            'blockage_cone_angle': 15,
            'term_decay_sigma': 2,
            'smoothing_radius': 1.5
        }

        if params:
            if isinstance(params, str) and params.endswith('.yaml'):
                with open(params, "r") as file:
                    params = yaml.safe_load(file)

            params = default_params | params
        else:
            params = default_params

        with open("expigeo_params.yaml", "w") as file:
            yaml.dump(params, file)

        probes = self.probe(point_cloud, 
                            K=params['K'],
                            theta_max_deg=params['theta'],
                            gamma=params['gamma'])

        centerline, radii, rgm = self.compute_centerline_and_radius(
            probes, dist_variance_percentile=params['dist_var_perc'],
            min_hits=params['min_valid_hits'],
            k_neighbors=params['k_neighbors'],
            std_multiplier=params['std_multiplier'],
            rgm_neighbors=params['local_rgm_points']
        )

        vertices, edges = self.extract_skeleton(
            centerline,
            voxel_size_multiplier=params['skeleton_voxel_size'],
            prune_multiplier=params['skeleton_pruning_multiplier'],
            density_percentile=params['skeleton_dense_perc'],
            factor=params['skeleton_merge_factor']
        )

        geometric_descriptors = self.compute_geometric_descriptors(vertices, 
                                                                    edges, 
                                                                    point_cloud[:, :3], 
                                                                    centerline, 
                                                                    radii, rgm,
                                                                    params['blockage_cone_angle'],
                                                                    params['term_decay_sigma'],
                                                                    params['smoothing_radius']
                                                                    )
        
        features = np.column_stack([attr for attr in geometric_descriptors.values()])

        explored_pc = np.concatenate([point_cloud,
                                      features, mask.reshape(-1, 1)], axis=1,
                                      dtype=np.float32)

        return explored_pc

if __name__ == '__main__':

    # this is example on how to use EXPIGEO as a standlone component.
    expigeo = EXPIGEO()
    fh = FileHandler()

    # select a mesh (.obj) from IntrA along with its annotation file (.ad)
    point_cloud = fh.annotate_mesh_to_array(mesh_path='AN1_full.obj', 
                                            annotation_path='AN1-_norm.ad')
    
    # the following function builds the feature matrix that is used by
    # ExpigeoGNN to segment aneurysms. It contains all the features:
    # coordinates and geometric descriptors.
    explored_pc = expigeo.explore(point_cloud, params='expigeo_params.yaml')

    print(explored_pc)