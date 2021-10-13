import numpy as np
import torch

import matplotlib.pyplot as plt

def visualize_loss_curves(train_loss, val_loss, name):
  train_loss_arr = torch.cat([t.reshape(-1,1) for t in train_loss], axis = 1).numpy()[0]
  val_loss_arr = torch.cat([v.reshape(-1,1) for v in val_loss], axis = 1).numpy()[0]

  indices = np.array(list(range(train_loss_arr.shape[0])))

  fig, ax = plt.subplots(1, 1, figsize=( 10, 8))

  ax.set_title(name)
  # plot lines
  ax.plot(indices, train_loss_arr, label = "Training", c = 'blue')
  ax.plot(indices, val_loss_arr, label = "Validation", c = 'orange')

  ax.legend()
  # ax.set_ylim(0,1)
  plt.show()


def visualize_all_curves(train_loss, val_loss, train_loss2, val_loss2, name):
  train_loss_arr = torch.cat([t.reshape(-1, 1) for t in train_loss], axis=1).numpy()[0]
  val_loss_arr = torch.cat([v.reshape(-1, 1) for v in val_loss], axis=1).numpy()[0]
  train_loss2_arr = torch.cat([t.reshape(-1, 1) for t in train_loss2], axis=1).numpy()[0]
  val_loss2_arr = torch.cat([v.reshape(-1, 1) for v in val_loss2], axis=1).numpy()[0]

  indices = np.array(list(range(train_loss_arr.shape[0])))

  fig, ax = plt.subplots(1, 1, figsize=(10, 8))

  ax.set_title(name)
  # plot lines
  ax.plot(indices, train_loss_arr, label="Training a", c='blue')
  ax.plot(indices, val_loss_arr, label="Validation a", c='orange')
  ax.plot(indices, train_loss2_arr, label="Training c", c='cyan')
  ax.plot(indices, val_loss2_arr, label="Validation c", c='yellow')

  ax.legend()
  # ax.set_ylim(0,1)
  plt.show()