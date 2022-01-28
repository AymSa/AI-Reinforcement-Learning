import numpy as np 
import scipy.signal
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import copy
import time
from core import *
from memory import *
from utils import *
from torch.distributions.normal import Normal
import itertools

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #A UTILISER PLUS TARD

class Actor(nn.Module):
  def __init__(self,inSize,outSize,low,high,hidden_sizes,activation):
    super().__init__()
    
    self.fc1 = NN(inSize,128,hidden_sizes,activation=activation)
    self.fc_mu = nn.Linear(128, outSize)
    self.fc_std  = nn.Linear(128, outSize)

    self.low = low
    self.high = high
  
  def forward(self, obs):
    
    out = F.relu(self.fc1(obs))
    mu = self.fc_mu(out)
    std = F.softplus(self.fc_std(out))
    dist = Normal(mu, std)
    action = dist.rsample()
    log_prob = dist.log_prob(action)

    real_action = torch.tanh(action)
    real_log_prob = log_prob - torch.log(1-torch.tanh(action).pow(2) + 1e-7) #Voir les autres implementations de SAC

    return torch.clamp(real_action,min=self.low,max=self.high),real_log_prob 
   
class Critic(nn.Module):
  def __init__(self,inSize,outSize,hidden_sizes,activation):
    super().__init__()
    self.net = NN(inSize,outSize,hidden_sizes,activation=activation)
  
  def forward(self,obs,actions):
    Q = self.net(torch.cat([obs,actions],dim=-1))
    return torch.squeeze(Q, -1)

class ActorCritic(nn.Module):
  def __init__(self, obs_size, n_actions,low,high, hidden_sizes = [256,256], activation=nn.ReLU()):
    super().__init__()
    self.n_actions = n_actions
    self.obs_size = obs_size

    self.actor = Actor(obs_size,n_actions,low,high,hidden_sizes,activation).to(device)
    self.critic1 = Critic(obs_size+n_actions,1,hidden_sizes,activation).to(device)
    self.critic2 = Critic(obs_size+n_actions,1,hidden_sizes,activation).to(device)
    
  @torch.no_grad()
  def act(self, obs):
    action, _ = self.actor(obs)
    return action
