import numpy as np
import scipy.signal
import torch
class SumTree:
    def __init__(self, mem_size):
        self.tree = np.zeros(2 * mem_size - 1)
        self.data = np.zeros(mem_size, dtype=object)
        self.size = mem_size
        self.ptr = 0
        self.nentities=0


    def update(self, idx, p):
        tree_idx = idx + self.size - 1
        diff = p - self.tree[tree_idx]
        self.tree[tree_idx] += diff
        while tree_idx:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff

    def store(self, p, data):
        self.data[self.ptr] = data
        self.update(self.ptr, p)
        idx=self.ptr
        self.ptr += 1
        if self.ptr == self.size:
            self.ptr = 0
        self.nentities+=1
        if self.nentities > self.size:
            self.nentities = self.size
        return idx

    def getNextIdx(self):
        return self.ptr

    def sample(self, value):
        ptr = 0
        while ptr < self.size - 1:
            left = 2 * ptr + 1
            if value < self.tree[left]:
                ptr = left
            else:
                value -= self.tree[left]
                ptr = left + 1

        return ptr - (self.size - 1), self.tree[ptr], self.data[ptr - (self.size - 1)]

    @property
    def total_p(self):
        return self.tree[0]

    @property
    def max_p(self):
        return np.max(self.tree[-self.size:])

    @property
    def min_p(self):
        return np.min(self.tree[-self.size:])


class Memory2:

    def __init__(self, mem_size, prior=True,p_upper=1.,epsilon=.01,alpha=1,beta=1):
        self.p_upper=p_upper
        self.epsilon=epsilon
        self.alpha=alpha
        self.beta=beta
        self.prior = prior
        self.nentities=0
        #self.dict={}
        #self.data_len = 2 * feature_size + 2
        self.mem_size = mem_size
        if prior:
            self.tree = SumTree(mem_size)
        else:

            self.mem = np.zeros(mem_size, dtype=object)
            self.mem_ptr = 0

    #def getID(self,transition):
    #    ind=-1
    #    if transition in dict:
    #        ind = dict[transition]
    #    return ind

    def store(self, transition):
        if self.prior:
            p = self.tree.max_p
            if not p:
                p = self.p_upper
            idx=self.tree.store(p, transition)
            self.nentities += 1
            if self.nentities > self.mem_size:
                self.nentities = self.mem_size
        else:
            self.mem[self.mem_ptr] = transition
            idx=self.mem_ptr
            self.mem_ptr += 1

            if self.mem_ptr == self.mem_size:
                self.mem_ptr = 0
            self.nentities += 1
            if self.nentities > self.mem_size:
                self.nentities = self.mem_size
        return idx

    def sample(self, n):
        if self.prior:
            min_p = self.tree.min_p
            if min_p==0:
                min_p=self.epsilon**self.alpha
            seg = self.tree.total_p / n
            batch = np.zeros(n, dtype=object)
            w = np.zeros((n, 1), np.float32)
            idx = np.zeros(n, np.int32)
            a = 0
            for i in range(n):
                b = a + seg
                v = np.random.uniform(a, b)
                idx[i], p, batch[i] = self.tree.sample(v)

                w[i] = (p / min_p) ** (-self.beta)
                a += seg
            return idx, w, batch
        else:
            mask = np.random.choice(range(self.nentities), n)
            return mask, 0,  self.mem[mask]

    def update(self, idx, tderr):
        if self.prior:
            tderr += self.epsilon
            tderr = np.minimum(tderr, self.p_upper)
            #print(idx,tderr)
            for i in range(len(idx)):
                self.tree.update(idx[i], tderr[i] ** self.alpha)

    def getNextIdx(self):
        if self.prior:
            ptr=self.tree.getNextIdx()
        else:
            ptr=self.mem_ptr
        return ptr

    def getData(self,idx):
        if idx >=self.nentities:
            return None
        if self.prior:
            data=self.tree.data[idx]
        else:
            data=self.mem[idx]
        return data
    
    
def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Memory:
   

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, ), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
      
        
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs.cpu()
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean,adv_std = self.adv_buf.mean(),self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}
