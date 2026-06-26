import yaml
import torch
import numpy as np

from tqdm import tqdm
from gnn import ExpigeoGNN
from scipy.spatial import KDTree
from dataclasses import dataclass
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader


@dataclass
class TestConfig:
    model: ExpigeoGNN
    model_state: str
    model_config: str
    build_dir: str
    graph_neighbors: int
    test_loader: DataLoader
    train_data_ratio: float
    val_data_ratio: float
    test_data_ratio: float
    batch_size: int
    metric: JaccardIndex
    device: str
    data_state: str
    save_results_in: str

    def serialize_config_as_yaml(self, save_path: str):

        config = {
            'model': None,
            'model_state': self.model_state,
            'model_config': self.model_config,
            'build_dir': self.build_dir,
            'graph_neighbors': self.graph_neighbors,
            'test_loader': None,
            'train_data_ratio': self.train_data_ratio,
            'val_data_ratio': self.val_data_ratio,
            'test_data_ratio': self.test_data_ratio,
            'batch_size': self.batch_size,
            'metric': None,
            'device': self.device,
            'data_state': None,
            'save_results_in': self.save_results_in
        }

        with open(save_path, "w") as file:
            yaml.dump(config, file)


def compute_iou_verbose(pred: np.ndarray,
                   gt:   np.ndarray,
                   tol:  float = 1e-5) -> float:

    P_A = pred[pred[:, 3] == 1][:, :3]
    G_A = gt[gt[:, 3] == 1][:, :3]

    P_V = pred[pred[:, 3] == 0][:, :3]
    G_V = gt[gt[:, 3] == 0][:, :3]

    tree_A = KDTree(G_A)
    tree_V = KDTree(G_V)

    idxs_A = tree_A.query_ball_point(P_A, tol)
    idxs_V = tree_V.query_ball_point(P_V, tol)

    matched_gt_A = {j for neighbors in idxs_A for j in neighbors}
    matched_gt_V = {j for neighbors in idxs_V for j in neighbors}

    intersection_A = len(matched_gt_A)
    union_A = len(P_A) + len(G_A) - intersection_A
    iou_A = intersection_A / union_A if union_A != 0 else 0
    dice_A = 2 * iou_A / (1 + iou_A)

    intersection_V = len(matched_gt_V)
    union_V = len(P_V) + len(G_V) - intersection_V
    iou_V = intersection_V / union_V if union_V != 0 else 0
    dice_V = 2 * iou_V / (1 + iou_V)

    stats = {
        'total_vertices': len(pred),
        'vessel_vertices': len(G_V),
        'aneurysm_vertices': len(G_A),
        'predicted_vessel_vertices': len(P_V),
        'matched_vessel_vertices': intersection_V,
        'predicted_aneurysm_vertices': len(P_A),
        'matched_aneurysm_vertices': intersection_A,
        'no_match_vessel_vertices': len(P_V) - intersection_V,
        'no_match_aneurysm_vertices': len(P_A) - intersection_A,
        'IoU_A': iou_A,
        'Dice_A': dice_A,
        'IoU_V': iou_V,
        'Dice_V': dice_V
    }

    return stats


def calculate_stats(results,
                    iou_detect_threshold: float=0.1):
    aneurysm_seg_iou, vessel_seg_iou = 0, 0
    aneurysm_seg_dice, vessel_seg_dice = 0, 0

    aneurysms, vessels = 0, 0

    TP_aneurysm, FP_aneurysm = 0, 0
    FN_aneurysm, TN_aneurysm = 0, 0

    TP_vessel, FP_vessel = 0, 0
    FN_vessel, TN_vessel = 0, 0

    for item in results['stats']:
        if item['aneurysm_vertices'] != 0 and item['IoU_A'] > iou_detect_threshold:
            aneurysms += 1
            aneurysm_seg_iou += item['IoU_A']
            aneurysm_seg_dice += item['Dice_A']

            TP_aneurysm += item['matched_aneurysm_vertices']

            FP_aneurysm += abs(item['predicted_aneurysm_vertices'] - item['matched_aneurysm_vertices'])

            FN_aneurysm += abs(item['aneurysm_vertices'] - item['matched_aneurysm_vertices'])

            TN_aneurysm += abs(item['total_vertices'] - TP_aneurysm - FP_aneurysm - FN_aneurysm)

        if item['vessel_vertices'] != 0 and item['IoU_V'] > iou_detect_threshold:
            vessels += 1
            vessel_seg_iou += item['IoU_V']
            vessel_seg_dice += item['Dice_V']

            TP_vessel += item['matched_vessel_vertices']

            FP_vessel += abs(item['predicted_vessel_vertices'] - item['matched_vessel_vertices'])

            FN_vessel += abs(item['vessel_vertices'] - item['matched_vessel_vertices'])

            TN_vessel += abs(item['total_vertices'] - TP_vessel - FP_vessel - FN_vessel)


    A_accuracy = (TP_aneurysm + TN_aneurysm) / (TP_aneurysm + TN_aneurysm + FP_aneurysm + FN_aneurysm)
    A_sensitivity = TP_aneurysm / (TP_aneurysm + FN_aneurysm)
    A_specificity = TN_aneurysm / (TN_aneurysm + FP_aneurysm)
    A_precision = TP_aneurysm / (TP_aneurysm + FP_aneurysm)
    A_F1_score = 2 * (A_precision * A_sensitivity) / (A_precision + A_sensitivity)

    print(f'Aneurysm IoU: {(aneurysm_seg_iou / aneurysms) * 100 :.3f}%')
    print(f'Aneurysm Dice: {(aneurysm_seg_dice / aneurysms) * 100 :.3f}%')
    print(f'Aneurysm Accuracy: {(A_accuracy) * 100 :.3f}%')
    print(f'Aneurysm Sensitivity (Recall): {(A_sensitivity) * 100 :.3f}%')
    print(f'Aneurysm Specificity: {(A_specificity) * 100 :.3f}%')
    print(f'Aneurysm Precision: {(A_precision) * 100 :.3f}%')
    print(f'Aneurysm F1_Score: {(A_F1_score) * 100 :.3f}%')

    V_accuracy = (TP_vessel + TN_vessel) / (TP_vessel + TN_vessel + FP_vessel + FN_vessel)
    V_sensitivity = TP_vessel / (TP_vessel + FN_vessel)
    V_specificity = TN_vessel / (TN_vessel + FP_vessel)
    V_precision = TP_vessel / (TP_vessel + FP_vessel)
    V_F1_score = 2 * (V_precision * V_sensitivity) / (V_precision + V_sensitivity)

    print(f'Vessel IoU: {(vessel_seg_iou / vessels) * 100 :.3f}%')
    print(f'Vessel Dice: {(vessel_seg_dice / vessels) * 100 :.3f}%')
    print(f'Vessel Accuracy: {(V_accuracy) * 100 :.3f}%')
    print(f'Vessel Sensitivity (Recall): {(V_sensitivity) * 100 :.3f}%')
    print(f'Vessel Specificity: {(V_specificity) * 100 :.3f}%')
    print(f'Vessel Precision: {(V_precision) * 100 :.3f}%')
    print(f'Vessel F1_Score: {(V_F1_score) * 100 :.3f}%')


    print(f'Mean IoU: {0.5 * ((vessel_seg_iou / vessels) + (aneurysm_seg_iou / aneurysms)) * 100 :.3f}%')
    print(f'Mean Dice: {0.5 * ((vessel_seg_dice / vessels) + (aneurysm_seg_dice / aneurysms)) * 100 :.3f}%')
    print(f'Mean Accuracy: {0.5 * (V_accuracy + A_accuracy) * 100 :.3f}%')
    print(f'Mean Sensitivity: {0.5 * (V_sensitivity + A_sensitivity) * 100 :.3f}%')
    print(f'Mean Specificity: {0.5 * (V_specificity + A_specificity) * 100 :.3f}%')
    print(f'Mean Precision: {0.5 * (V_precision + A_precision) * 100 :.3f}%')
    print(f'Mean F1_Score: {0.5 * (V_F1_score + A_F1_score) * 100 :.3f}%')

def test(test_config: TestConfig):
    test_config.model.eval()
    all_preds = []
    all_labels = []

    ground_truth = []
    predictions = []

    global_stats = {'stats': [], 'iou_aneurysm': 0, 'iou_vessel': 0,
                    'dice_aneurysm': 0, 'dice_vessel': 0}
    num_aneurysms = 0

    with torch.inference_mode():
        for idx, data in enumerate(tqdm(test_config.test_loader, desc="Testing", unit='Batch')):
            if data.edge_index.size(0) != 0:
                graph = data
                graph = data.to(test_config.device)
                out = test_config.model(graph)
                pred = out.argmax(dim=1)

                gt_save = torch.concatenate([graph.x[:, :3].detach().cpu(),
                                            graph.y.detach().cpu().unsqueeze(dim=1)], dim=1)

                output_save = torch.concatenate([graph.x[:, :3].detach().cpu(),
                                                pred.detach().cpu().unsqueeze(dim=1)], dim=1)

                local_stats = compute_iou_verbose(output_save.numpy(), gt_save.numpy())
                
                local_stats['filename'] = test_config.test_loader.dataset.dataset.files[idx]

                global_stats['stats'].append(local_stats)

                global_stats['iou_aneurysm'] += local_stats['IoU_A']
                global_stats['dice_aneurysm'] += local_stats['Dice_A']

            if local_stats['filename'].find('aneurysms') != -1:
                num_aneurysms += 1

                global_stats['iou_vessel'] += local_stats['IoU_V']
                global_stats['dice_vessel'] += local_stats['Dice_V']

                predictions.append(output_save.numpy())
                ground_truth.append(gt_save.numpy())

                all_preds.append(pred.cpu())
                all_labels.append(graph.y.cpu())

    global_stats['iou_aneurysm'] = global_stats['iou_aneurysm'] / num_aneurysms
    global_stats['iou_vessel'] =  global_stats['iou_vessel'] / len(test_config.test_loader)
    global_stats['dice_aneurysm'] = global_stats['dice_aneurysm'] / num_aneurysms
    global_stats['dice_vessel'] =  global_stats['dice_vessel'] / len(test_config.test_loader)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print('\nResults\n')

    calculate_stats(global_stats)

    return global_stats