import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from scipy.stats import entropy
from torch_geometric.data import Data
from torch_geometric.explain.config import ModelConfig
from torch_geometric.explain import GNNExplainer, Explainer


class ExplainerWrapper(nn.Module):
    """
    This model's sole goal is to wrap a trained ExpigeoGNN model for explainability.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        data = Data(x=x, edge_index=edge_index)
        return self.model(data)


def compute_receptive_field_entropy(model, loader, device: str,
                                    entropy_threshold: float,
                                    max_samples: int=10, 
                                    num_epochs: int=10):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=num_epochs), # add more epochs for more accurate results. 
        explanation_type='model',
        node_mask_type='object',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    model.eval()
    samples_processed = 0
    high_entropy_count = 0
    results = []

    for batch_idx, data in enumerate(loader):
        data = data.to(device)

        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)

        # we identify prediction and ground-truth mismatch
        errors = (preds != data.y)
        error_indices = errors.nonzero(as_tuple=True)[0]

        if len(error_indices) == 0:
            continue

        # we select random subset of errors to analyze for computation efficiency.
        if len(error_indices) > max_samples:
            perm = torch.randperm(len(error_indices))
            selected_indices = error_indices[perm[:max_samples]]
        else:
            selected_indices = error_indices

        for i, target_idx in enumerate(selected_indices):
            target_idx_int = target_idx.item()
            true_label = data.y[target_idx_int].item()
            pred_label = preds[target_idx_int].item()

            explanation = explainer(
                x=data.x,
                edge_index=data.edge_index,
                index=target_idx_int
            )

            mask = explanation.node_mask.detach().cpu()
            if mask.ndim > 1: mask = mask.mean(dim=1)

            imp_nodes = (mask > 0.5)
            imp_nodes[target_idx_int] = True

            subgraph_labels = data.y[imp_nodes].cpu().numpy()

            # Eq. (6) from the paper.
            values, counts = np.unique(subgraph_labels, return_counts=True)
            probs = counts / len(subgraph_labels)
            ent_score = entropy(probs, base=2)

            is_high_entropy = ent_score > entropy_threshold
            if is_high_entropy:
                high_entropy_count += 1

            results.append({
                'node_idx': target_idx_int,
                'true': true_label,
                'pred': pred_label,
                'entropy': ent_score,
                'is_high': is_high_entropy
            })

            g_label = 'Aneurysm' if true_label == 1 else 'Vessel'
            p_label = 'Aneurysm' if pred_label == 1 else 'Vessel'

            print(f"\tNode {target_idx_int} (Ground-Truth: {g_label} -> Pred: {p_label}) "
                  f"| Entropy: {ent_score:.4f} | {'High' if is_high_entropy else 'Low'}")

            samples_processed += 1
            if samples_processed >= max_samples:
                break

        if samples_processed >= max_samples:
            break

    if samples_processed == 0:
        print("No prediction mismatch found.")
        return

    correlation_pct = (high_entropy_count / samples_processed) * 100

    print(f"\nTotal Misclassifications Analyzed: {samples_processed}")
    print(f"High Entropy Explanations (> {entropy_threshold}): {high_entropy_count}")
    print(f"{correlation_pct:.1f}% of errors occurred in mixed/confusing neighborhoods.")


def compute_feature_importance(model, loader, 
                               device: str, 
                               num_epochs: int=10):
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=num_epochs), # add more epochs for more accurate results. 
        explanation_type='model',
        node_mask_type='attributes',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )

    model.eval()

    imp_sum_vessel = None
    imp_sum_aneurysm = None

    count_vessel = 0
    count_aneurysm = 0

    for batch_idx, data in enumerate(tqdm(loader, desc="Computing Feature Importance")):
        data = data.to(device)
        num_features = data.x.shape[1]

        if imp_sum_vessel is None:
            imp_sum_vessel = torch.zeros(num_features, device=device)
            imp_sum_aneurysm = torch.zeros(num_features, device=device)

        with torch.no_grad():
            out = model(data.x, data.edge_index)
            preds = out.argmax(dim=1)

        # we select correctly classified nodes for valid feature analysis
        vessel_indices = ((preds == 0) & (data.y == 0)).nonzero(as_tuple=True)[0]
        aneurysm_indices = ((preds == 1) & (data.y == 1)).nonzero(as_tuple=True)[0]

        targets_to_explain = []

        if len(vessel_indices) > 0:
            idx = vessel_indices[torch.randint(len(vessel_indices), (1,)).item()]
            targets_to_explain.append((idx, 0)) # (Index, Class)

        if len(aneurysm_indices) > 0:
            idx = aneurysm_indices[torch.randint(len(aneurysm_indices), (1,)).item()]
            targets_to_explain.append((idx, 1))

        for target_idx, class_label in targets_to_explain:
            with torch.enable_grad():
                explanation = explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    index=target_idx.item()
                )

            feature_imp = explanation.node_mask.mean(dim=0)

            # feature importance for each class. Check the paper.
            if class_label == 0:
                imp_sum_vessel += feature_imp
                count_vessel += 1
            else:
                imp_sum_aneurysm += feature_imp
                count_aneurysm += 1

    if count_vessel > 0:
        avg_vessel = (imp_sum_vessel / count_vessel).cpu().numpy()
        avg_vessel = avg_vessel / avg_vessel.max()
    else:
        avg_vessel = np.zeros(num_features)

    if count_aneurysm > 0:
        avg_aneurysm = (imp_sum_aneurysm / count_aneurysm).cpu().numpy()
        avg_aneurysm = avg_aneurysm / avg_aneurysm.max()
    else:
        avg_aneurysm = np.zeros(num_features)


    return avg_aneurysm.tolist(), avg_vessel.tolist()