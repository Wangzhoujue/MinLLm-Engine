
import torch
from dataclasses import dataclass

@dataclass
class Sampler:

    def sample(self,logits,strategy="greedy",temperature=1.0, top_k=None, top_p=None):
        if strategy not in ["greedy","random","top_k","top_p"]:
            raise ValueError("该采样策略不存在。")
        if temperature<=0:raise ValueError("temperature必须大于0。")
        if strategy=="greedy":
            return self.greedy_sample(logits)
        elif strategy=="random":
            return self.random_sample(logits,temperature)
        elif strategy=="top_k":
            if top_k is not None and (type(top_k) != int or top_k <=0) :raise ValueError("top_k必须为正整数。")
            return self.top_k_sample(logits,temperature,top_k)
        elif strategy=="top_p":
            if top_p is not None and (top_p<=0 or top_p>1):raise ValueError("top_p必须满足 0 < top_p <= 1)。")
            return self.top_p_sample(logits,temperature,top_p)
                
        
        
    
    def greedy_sample(self,logits):
        return torch.argmax(logits,dim=-1)

    def random_sample(self, logits, temperature=1):
        logits=logits/temperature
        probs=torch.softmax(logits,dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def top_k_sample(self,logits,temperature,top_k):
        values,indices=torch.topk(logits,top_k,dim=-1)
        new_logits=torch.full_like(logits,float("-inf"))
        new_logits.scatter_(-1, indices, values)
        return self.random_sample(new_logits,temperature)

    def top_p_sample(self,logits,temperature,top_p):
        logits=logits/temperature
        probs=torch.softmax(logits,dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative_probs > top_p
        remove_mask[:, 0] = False
        mask=torch.zeros_like(remove_mask)
        mask.scatter_(-1,sorted_indices,remove_mask)
        new_logits=logits.masked_fill(mask, float("-inf"))
        return self.random_sample(new_logits)

        
