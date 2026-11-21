from dataclasses import dataclass, field

@dataclass
class BlockTable:
    block_table: dict[int, list[int]] = field(default_factory=dict)

    def add_block(self,request_id, block_id):
        if request_id not in self.block_table:
            self.block_table[request_id]=[]
        if block_id in self.block_table[request_id]:
            raise ValueError(f"物理块{block_id}在请求{request_id}中已存在，不允许重复添加。")
        self.block_table[request_id].append(block_id)
        
    def get_blocks(self,request_id):
        if request_id not in self.block_table: 
            raise ValueError(f"逻辑块表中不存在请求{request_id}的映射。")
        return self.block_table[request_id]

    def remove_request(self,request_id):
        if request_id not in self.block_table: 
            raise ValueError(f"逻辑块表中不存在请求{request_id}的映射。")
        self.block_table.pop(request_id)