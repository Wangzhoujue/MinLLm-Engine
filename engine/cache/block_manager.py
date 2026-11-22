from dataclasses import dataclass, field

@dataclass
class BlockManager:
    num_block:int
    free_block:set=field(init=False)
    used_block:set=field(init=False)

    def __post_init__(self):
        if not self.is_legal_parameters(self.num_block):raise ValueError(f"数值{self.num_block}不合法，数值必须为正整数。")
        self.used_block=set()
        self.free_block=set(range(self.num_block))

    def is_legal_parameters(self,num):
        return type(num)is int and num>0
    
    def allocate(self,n):
        if n<0 or n%1!=0:raise ValueError(f"数值{n}不合法，数值必须为正整数。")
        if len(self.free_block)<n:
            raise MemoryError("内存资源不足。")
        block_ids=[]
        for i in range(n):
            block=self.free_block.pop()
            self.used_block.add(block)
            block_ids.append(block)
        return block_ids

    def free(self,block_ids):
        block_ids=set(block_ids)
        if not block_ids.issubset(self.used_block): raise ValueError(f"已使用块表中不存在{block_ids}。")
        for block_id in block_ids:
            self.used_block.remove(block_id)
            self.free_block.add(block_id)


        

