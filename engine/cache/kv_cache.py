
import torch
from dataclasses import dataclass, field


@dataclass
class KVCache:
    num_blocks:int
    block_size:int
    num_layers:int
    num_kv_heads:int
    head_dim:int
    dtype:torch.dtype
    device:str
    key_cache:torch.Tensor=field(init=False)
    value_cache:torch.Tensor=field(init=False)
    cache_valid:torch.Tensor=field(init=False)

    def __post_init__(self):
        for num in (self.num_layers,self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim):
            if not self.is_legal_parameters(num): raise ValueError(f"数值{num}应为正整数，不合法")
             
        self.key_cache=torch.empty((self.num_layers,self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim),
                                    dtype=self.dtype,
                                    device=self.device
                                    )
        self.value_cache=torch.empty((self.num_layers,self.num_blocks,self.block_size,self.num_kv_heads,self.head_dim),
                                    dtype=self.dtype,
                                    device=self.device
                                    )
        self.cache_valid = torch.zeros((self.num_layers, self.num_blocks, self.block_size),
                                    dtype=torch.bool,
                                    device=self.device
                                    )

    def is_legal_parameters(self,num,min=None,max=None):
            if min is not None and max is not None:
                return  type(num)is int and num>=0 and num>=min and num<=max
            if min is not None:
                return type(num)is int and num>=0 and num>=min
            if max is not None:
                return type(num)is int and num>=0 and num<=max
            return type(num)is int and num>=0 

    def write_kvcache(self,layer,block_id,offset,key,value):
        if not self.is_legal_parameters(layer,max=self.num_layers-1) or not self.is_legal_parameters(block_id,max=self.num_blocks-1) or not self.is_legal_parameters(offset,max=self.block_size-1):
            raise ValueError("数据不合法！")
        if key.shape != (self.num_kv_heads, self.head_dim) or value.shape != (self.num_kv_heads, self.head_dim):
            raise ValueError("数据不合法！")
        self.cache_valid[layer,block_id,offset]=False
        self.key_cache[layer,block_id,offset]=key
        self.value_cache[layer,block_id,offset]=value
        self.cache_valid[layer,block_id,offset]=True

    def read_kvcache(self, layer, block_id, offset):
        if not self.is_legal_parameters(layer,max=self.num_layers-1) or not self.is_legal_parameters(block_id,max=self.num_blocks-1) or not self.is_legal_parameters(offset,max=self.block_size-1):
                    raise ValueError("数据不合法！")
        if self.cache_valid[layer,block_id,offset].item() is not True:
             raise ValueError("数据未曾被写入。")
        key=self.key_cache[layer,block_id,offset]
        value=self.value_cache[layer,block_id,offset]
        return(key,value)

    def reset_block(self, block_id):
        if not self.is_legal_parameters(block_id,max=self.num_blocks-1):
             raise ValueError(f"块号{block_id}不合法。")
        self.cache_valid[:,block_id,:]=False

        