import numpy as np 
import scipy.signal
import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy
import time
from torch.distributions import Categorical
from core import *
from memory import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #A UTILISER PLUS TARD

class Actor(nn.Module):
  def __init__(self,inSize,outSize,low,high,hidden_sizes,activation):
    super().__init__()
    self.net = NN(inSize,outSize,hidden_sizes,activation=activation,finalActivation=nn.Tanh())
    self.low = low
    self.high = high
  
  def forward(self, obs):
    return torch.clamp(self.net(obs),min=self.low,max=self.high)
   
class Critic(nn.Module):
  def __init__(self,inSize,outSize,hidden_sizes,activation):
    super().__init__()
    self.net = NN(inSize,outSize,hidden_sizes,activation=activation)
  
  def forward(self,obs,actions):
    Q = self.net(torch.cat([obs,actions],dim=-1))
    return torch.squeeze(Q, -1)

class ActorCritic(nn.Module):
  def __init__(self, obs_size, n_actions,low,high, hidden_sizes = [128,128], activation=nn.ReLU()):
    super().__init__()
    self.n_actions = n_actions
    self.obs_size = obs_size

    self.actor = Actor(obs_size,n_actions,low,high,hidden_sizes,activation).to(device)
    self.critic = Critic(obs_size+n_actions,1,hidden_sizes,activation).to(device)
    
  @torch.no_grad()
  def act(self, obs):
    return self.actor(obs)