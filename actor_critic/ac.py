import numpy as np 
import scipy.signal
import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import copy
import time
from core import *
from memory import *
from utils import *
from torch.distributions import Categorical

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #A UTILISER PLUS TARD

class Actor(nn.Module):
  def __init__(self,inSize,outSize,hidden_sizes,activation):
    super().__init__()
    self.net = NN(inSize,outSize,hidden_sizes,activation)
  
  def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

  def _distribution(self, obs):
        logits = self.net(obs)
        return Categorical(logits=logits)

  def _log_prob_from_distribution(self, pi, act):
      return pi.log_prob(act)

class Critic(nn.Module):
  def __init__(self,inSize,outSize,hidden_sizes,activation):
    super().__init__()
    self.net = NN(inSize,outSize,hidden_sizes,activation)
  
  def forward(self,obs):
    return torch.squeeze(self.net(obs), -1)

class ActorCritic(nn.Module):
  def __init__(self, obs_size, n_actions, hidden_sizes = [64,64], activation=nn.Tanh()):
    super().__init__()
    self.n_actions = n_actions
    self.obs_size = obs_size

    self.actor = Actor(obs_size,n_actions,hidden_sizes,activation).to(device)
    self.critic = Critic(obs_size,1,hidden_sizes,activation).to(device)
    

  def step(self, obs):
      with torch.no_grad():
          pi = self.actor._distribution(obs)
          a = pi.sample()
          logp_a = self.actor._log_prob_from_distribution(pi, a)
          v = self.critic(obs)
      return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

  def act(self, obs):
      return self.step(obs)[0]
