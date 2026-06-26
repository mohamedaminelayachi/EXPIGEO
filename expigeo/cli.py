import yaml
import json
import torch
import argparse
import torch.nn as nn

from typing import Union
from torch.optim import AdamW
from test import TestConfig, test
from train import TrainConfig, train
from torchmetrics import JaccardIndex
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from data import (IntrABuilder, IntrA, 
                  serialize_data_state, 
                  load_data_state,
                  class_weights)
from explain import (ExplainerWrapper, 
                     compute_receptive_field_entropy,
                     compute_feature_importance)
from gnn import ExpigeoGNN, LovaszSoftmaxLoss, ModelConfig


def build(intra_root_dir: str, 
          loadmap_path: str='intra_loadmap.json',
          save_dir: str='IntrAExpigeo',
          target_files: str = 'balanced',
          size: float = 1,
          expigeo_params: Union[dict, str] = None
          ):
    builder = IntrABuilder(
        intra_root_dir=intra_root_dir,
        loadmap=loadmap_path)

    builder.explore_full_dataset(save_directory=save_dir,
                           target_files=target_files,
                           size=size,
                           expigeo_params=expigeo_params)
    
def train_process(train_config_path: str,
                  data_state_path: str='state.json'):
    
    with open(train_config_path, "r") as fp:
        train_config = TrainConfig(**yaml.safe_load(fp))

    intra = IntrA(build_dir=train_config.build_dir,
                  graph_neighbors=train_config.graph_neighbors)

    if train_config.data_state:
        train_split, val_split, test_split = load_data_state(intra,
                                                             train_config.data_state,
                                                             train_config.train_data_ratio,
                                                             train_config.val_data_ratio,
                                                             train_config.test_data_ratio)
    else:
        train_split, val_split, test_split = random_split(
            intra, (train_config.train_data_ratio,
                    train_config.val_data_ratio,
                    train_config.test_data_ratio))

        serialize_data_state(train_split=train_split,
                            val_split=val_split,
                            test_split=test_split,
                            save_path=data_state_path)
    
    train_loader = DataLoader(train_split, 
                              batch_size=train_config.batch_size,
                              shuffle=True)
    
    val_loader = DataLoader(val_split,
                            batch_size=train_config.batch_size,
                            shuffle=True)

    if train_config.model_config:
        with open(train_config.model_config, "r") as file:
            model_config = ModelConfig(**yaml.safe_load(file))
        model = ExpigeoGNN(model_config=model_config)
    else:
        model = ExpigeoGNN()

    if train_config.model_state:
        model.load_state_dict(torch.load(
            train_config.model_state,
            map_location=train_config.device
        ))

    optimizer = AdamW(model.parameters(),
                      lr=train_config.lr,
                      weight_decay=train_config.weight_decay)
    
    weight_ce = class_weights(split=train_split) 
    criterion_1 = nn.CrossEntropyLoss(weight=weight_ce)
    criterion_2 = LovaszSoftmaxLoss(class_to_optimize=1)

    metric = JaccardIndex(task='multiclass', num_classes=2)

    train_config.model = model
    train_config.train_loader = train_loader
    train_config.val_loader = val_loader
    train_config.optimizer = optimizer
    train_config.criterion_1 = criterion_1
    train_config.criterion_2 = criterion_2
    train_config.metric = metric

    train_loss, val_loss, train_iou, val_iou = train(train_config)

    if train_config.train_logs_save_path:
        with open(train_config.train_logs_save_path, 'w') as fp:
            logs = {
                'training_loss': train_loss,
                'validation_loss': val_loss,
                'training_iou': train_iou,
                'validation_iou': val_iou
            }
            json.dump(logs, fp, indent=2)


def test_process(test_config: str):

    with open(test_config, 'r') as fp:
        test_config: TestConfig = TestConfig(**yaml.safe_load(fp))

    intra = IntrA(build_dir=test_config.build_dir,
                  graph_neighbors=test_config.graph_neighbors)

    _, _, test_split = load_data_state(intra,
                                        test_config.data_state,
                                        test_config.train_data_ratio,
                                        test_config.val_data_ratio,
                                        test_config.test_data_ratio)

    test_loader = DataLoader(test_split,
                            batch_size=test_config.batch_size,
                            shuffle=False)
    
    if test_config.model_config:
        with open(test_config.model_config, "r") as file:
            model_config = ModelConfig(**yaml.safe_load(file))
        model = ExpigeoGNN(model_config=model_config)
    else:
        model = ExpigeoGNN()

    model.load_state_dict(torch.load(
        test_config.model_state,
        map_location=test_config.device
    ))

    metric = JaccardIndex(task='multiclass', num_classes=2)

    test_config.model = model
    test_config.test_loader = test_loader
    test_config.metric = metric

    results = test(test_config=test_config)

    if test_config.save_results_in:
        with open(test_config.save_results_in, 'w') as fp:
            json.dump(results, fp, indent=2)

def explain_process(test_config: str,
                    epochs: int=10,
                    entropy_threshold: float=0.6,
                    max_samples: int=10,
                    fimp_save_path: str='feature_importance.json'):

    with open(test_config, 'r') as fp:
        test_config: TestConfig = TestConfig(**yaml.safe_load(fp))

    intra = IntrA(build_dir=test_config.build_dir,
                  graph_neighbors=test_config.graph_neighbors)

    _, _, test_split = load_data_state(intra,
                                        test_config.data_state,
                                        test_config.train_data_ratio,
                                        test_config.val_data_ratio,
                                        test_config.test_data_ratio)

    test_loader = DataLoader(test_split,
                            batch_size=test_config.batch_size,
                            shuffle=False)
    
    if test_config.model_config:
        with open(test_config.model_config, "r") as file:
            model_config = ModelConfig(**yaml.safe_load(file))
        model = ExpigeoGNN(model_config=model_config)
    else:
        model = ExpigeoGNN()

    model.load_state_dict(torch.load(
        test_config.model_state,
        map_location=test_config.device
    ))

    exp_model = ExplainerWrapper(model)

    print("\nReceptive Field Entropy\n")

    hrf = compute_receptive_field_entropy(model=exp_model,
                                              loader=test_loader,
                                              device=test_config.device,
                                              entropy_threshold=entropy_threshold,
                                              max_samples=max_samples,
                                              num_epochs=epochs
                                              )

    print("\nFeature Importance\n")

    f_imp = compute_feature_importance(model=exp_model,
                                       loader=test_loader,
                                       device=test_config.device,
                                       num_epochs=epochs)
    
    aneurysm_imp = {
        'x': f_imp[0][0], 'y': f_imp[0][1], 'z': f_imp[0][2],
        'P_term': f_imp[0][3], 'beta': f_imp[0][4], 'phi': f_imp[0][5],
        'r': f_imp[0][6], 'rgm': f_imp[0][7], 'proj_flow': f_imp[0][8],
    }

    vessel_imp = {
        'x': f_imp[1][0], 'y': f_imp[1][1], 'z': f_imp[1][2],
        'P_term': f_imp[1][3], 'beta': f_imp[1][4], 'phi': f_imp[1][5],
        'r': f_imp[1][6], 'rgm': f_imp[1][7], 'proj_flow': f_imp[1][8],
    }

    print(f'\nAneurysm Feature Importance: {aneurysm_imp}')
    print(f'\nVessel Feature Importance: {vessel_imp}')

    feature_importance = {
        'aneurysm': aneurysm_imp,
        'vessel': vessel_imp
    }
    with open(fimp_save_path, 'w') as fp:
        json.dump(feature_importance, fp, indent=2)

class CLI:

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers(dest='command', required=True)
        self.build_parser()

    def build_parser(self):

        # build
        self.parser_build = self.subparsers.add_parser("build", help="Given the IntrA root directory," \
        " it builds the EXPIGEO-ready dataset.")
        self.parser_build.add_argument('-rd', '--rootdir', type=str, help='The root directory of IntrA.',
                                  required=True)
        self.parser_build.add_argument('-lp', '--loadmap', type=str, help="The save path of IntrA's" \
        " loadmap.", default='intra_loadmap.json')
        self.parser_build.add_argument('-s', '--save_dir', required=True,
                                   type=str, help="The directory in which the dataset is built.")
        self.parser_build.add_argument('-t', '--target_files', type=str, 
                                  help="Target files from the dataset",
                                  choices=['balanced', 'positives', 'negatives', 'rem_negatives',
                                           'all_files'], default='balanced')
        self.parser_build.add_argument('-ds', '--dataset_size', default=1.0,
                                   type=float, help="Portion of the dataset to use.")
        self.parser_build.add_argument('-exp', '--expigeo_params', default='expigeo_params.yaml',
                                   type=str, help="EXPIGEO Parameters")

        # train
        self.parser_train = self.subparsers.add_parser("train", help='Given the dataset built,' \
        ' it trains an ExpigeoGNN model.')
        self.parser_train.add_argument('-cfg', '--train_config', type=str, required=True,
                                       help='Training configuration Check the YAML file.')
        self.parser_train.add_argument('-sds', '--save_data_state', type=str,
                                       help="JSON file to save the data state." \
        " This is a must to avoid data leaks during testing.", default='data_state.json')
        
        # test
        self.parser_test = self.subparsers.add_parser("test", 
        help='Given a trained ExpigeoGNN model, it tests it on a held-out test split.')
        self.parser_test.add_argument('-cfg', '--test_config', type=str, required=True,
                                       help='Test configuration. Check the YAML file.')

        # explain
        self.parser_explain = self.subparsers.add_parser("explain", 
        help="Use GNNExplainer to explain a trained model's predictions.")
        self.parser_explain.add_argument("-cfg", "--test_config", type=str, required=True,
                                         help='Test configuration. Check the YAML file.')
        self.parser_explain.add_argument("-ep", "--explain_epochs", type=int, default=10,
                                         help='Number of epochs used in GNNExplainer.')
        self.parser_explain.add_argument("-enth", "--entropy_threshold", type=float,
                                         help='Entropy threshold.', default=0.6)
        self.parser_explain.add_argument("-mxs", "--max_samples", type=int,
                                         help='Maximum number of nodes to consider per graph.',
                                         default=10)
        self.parser_explain.add_argument("-fimp", "--feature_importance_file", type=str,
                                         help='JSON file to save feature importance results.',
                                         default='feature_importance.json')

    def run(self):
        args = self.parser.parse_args()
        
        if args.command == 'build':
            build(intra_root_dir=args.rootdir,
                  loadmap_path=args.loadmap,
                  save_dir=args.save_dir,
                  target_files=args.target_files,
                  size=args.dataset_size,
                  expigeo_params=args.expigeo_params)
            
        elif args.command == 'train':
            train_process(train_config_path=args.train_config,
                          data_state_path=args.save_data_state)
                
        elif args.command == 'test':
            test_process(args.test_config)

        elif args.command == 'explain':
            explain_process(test_config=args.test_config,
                            epochs=args.explain_epochs,
                            entropy_threshold=args.entropy_threshold,
                            max_samples=args.max_samples,
                            fimp_save_path=args.feature_importance_file)