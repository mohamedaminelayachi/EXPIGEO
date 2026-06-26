import os
import yaml
import torch
import torch.nn as nn

from tqdm import tqdm
from gnn import ExpigeoGNN
from torch.optim import AdamW
from dataclasses import dataclass
from torchmetrics import JaccardIndex
from torch.utils.data import DataLoader

@dataclass
class TrainConfig:
  
  model: ExpigeoGNN
  model_config: str
  model_state: str
  build_dir: str
  graph_neighbors: int
  train_loader: DataLoader
  val_loader: DataLoader
  train_data_ratio: float
  val_data_ratio: float
  test_data_ratio: float
  optimizer: AdamW
  criterion_1: nn.Module
  criterion_2: nn.Module
  batch_size: int
  lr: float
  weight_decay: float
  alpha: float
  metric: JaccardIndex
  num_epochs: int
  device: str
  checkpoint_each: int # checkpoint at each given value.
  checkpoint_dir: str
  checkpoint_stop: int # continue from the epoch you stopped training at.
  data_state: str # use old data state.
  train_logs_save_path: str

  def serialize_config_as_yaml(self, save_path: str):

    config = {
      'model': None,
      'model_config': self.model_config,
      'model_state': self.model_state,
      'build_dir': self.build_dir,
      'graph_neighbors': self.graph_neighbors,
      'train_loader': None,
      'val_loader': None,
      'train_data_ratio': self.train_data_ratio,
      'val_data_ratio': self.val_data_ratio,
      'test_data_ratio': self.test_data_ratio,
      'optimizer': None,
      'criterion_1': None,
      'criterion_2': None,
      'batch_size': self.batch_size,
      'lr': self.lr,
      'weight_decay': self.weight_decay,
      'alpha': self.alpha,
      'metric': None,
      'num_epochs': self.num_epochs,
      'device': self.device,
      'checkpoint_each': self.checkpoint_each,
      'checkpoint_dir': self.checkpoint_dir,
      'checkpoint_stop': self.checkpoint_stop,
      'data_state': None,
      'train_logs_save_path': self.train_logs_save_path
    }
    with open(save_path, "w") as file:
      yaml.dump(config, file)
     

def train(train_config: TrainConfig):


  if not os.path.exists(train_config.checkpoint_dir):
    os.mkdir(train_config.checkpoint_dir)

  metric = train_config.metric

  training_losses = []
  validation_losses = []
  training_iou_log = []
  validation_iou_log = []

  for epoch in range(train_config.num_epochs):
      train_config.model.train()

      total_training_loss = 0
      total_validation_loss = 0
      total_training_iou = 0
      total_validation_iou = 0

      current_epoch = epoch+1+train_config.checkpoint_stop
      all_epochs = train_config.num_epochs+train_config.checkpoint_stop
      desc = f"Epoch ({current_epoch}/{all_epochs}) - Training"

      for data in tqdm(train_config.train_loader, 
                       desc=desc, unit="Batch"):
        graph = data.to(train_config.device)

        train_config.optimizer.zero_grad()

        pred = train_config.model(graph)
        total_training_iou += metric(pred.detach().cpu(), graph.y.detach().cpu())

        wce = train_config.criterion_1(pred, graph.y)
        lovasz = train_config.criterion_2(pred, graph.y)

        joint_train_loss = wce + train_config.alpha * lovasz
        joint_train_loss.backward()

        train_config.optimizer.step()

        total_training_loss += joint_train_loss.item()

      training_losses.append(total_training_loss / len(train_config.train_loader))
      train_config.model.eval()

      with torch.inference_mode():
        
        desc = f"Epoch ({current_epoch}/{all_epochs}) - Validation"

        for val_data in tqdm(train_config.val_loader, 
                             desc=desc,
                               unit="Batch"):
            graph = val_data.to(train_config.device)
            pred = train_config.model(graph)
            
            wce = train_config.criterion_1(pred, graph.y)
            lovasz = train_config.criterion_2(pred, graph.y)
            
            join_val_loss = wce + train_config.alpha * lovasz
            
            total_validation_iou += metric(pred.detach().cpu(), graph.y.detach().cpu())
            total_validation_loss += join_val_loss.item()

      print(f"Epoch ({current_epoch}/{all_epochs}) -"
            f" Training Loss: {total_training_loss / len(train_config.train_loader):.4f},"
            f" Validation Loss: {total_validation_loss / len(train_config.val_loader):.4f}"
            f", Training mIoU: {100*(total_training_iou / len(train_config.train_loader)):.4f}%"
            f", Validation mIoU: {100*(total_validation_iou / len(train_config.val_loader)):.4f}%\n")

      training_iou_log.append(total_training_iou.item() / len(train_config.train_loader))
      validation_iou_log.append(total_validation_iou.item() / len(train_config.val_loader))

      validation_losses.append(total_validation_loss / len(train_config.val_loader))

      if current_epoch % train_config.checkpoint_each == 0 or current_epoch == all_epochs:
        torch.save(train_config.model.state_dict(),
                   os.path.join(train_config.checkpoint_dir,
                                f'model_{current_epoch}_{all_epochs}.pth'))
        print(f"Saved model_{current_epoch}_{all_epochs}.pth")

  return training_losses, validation_losses, training_iou_log, validation_iou_log
