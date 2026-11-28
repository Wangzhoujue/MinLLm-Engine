
from math import ceil
from collections import deque
from dataclasses import dataclass, field
from .request import Request
from .cache import BlockManager, BlockTable, KVCache


@dataclass
class Scheduler:
    kv_cache:KVCache
    block_table:BlockTable
    block_manager:BlockManager
    waiting_queue:deque[Request]=field(default_factory=deque)
    running_queue:deque[Request]=field(default_factory=deque)


    def add_request(self,request):
        if request.status != "WAITING":
            raise ValueError(f"请求{request.request_id}不处于WAITING状态。")
        if any(req.request_id==request.request_id  for req in self.waiting_queue):
            raise ValueError(f"请求{request.request_id}已存在于waiting队列。")
        if any(req.request_id==request.request_id  for req in self.running_queue):
            raise ValueError(f"请求{request.request_id}已存在于running队列。")
        self.waiting_queue.append(request)

    def calculate_num_blocks(self, request):
            return ceil((request.prompt_token_ids.shape[-1]+request.max_tokens)/self.kv_cache.block_size)

    def schedule(self):
        if len(self.waiting_queue)==0:
            return
        request=self.waiting_queue.popleft()
        if request.status == "ABORTED":
            return
        num_block=self.calculate_num_blocks(request)
        if num_block > self.block_manager.num_block:
            request.abort()
            request.error = "所需 KV Cache Block 数量超过总容量。"
            return
        try:
            block_ids=self.block_manager.allocate(num_block)
            for block_id in block_ids:
                self.block_table.add_block(request.request_id,block_id)
            request.start()
        except MemoryError:
            self.waiting_queue.appendleft(request)
            return  
        self.running_queue.append(request)

    def get_next_request(self):
        request=self.running_queue.popleft()
        self.running_queue.append(request)
        return request

    def _release_running_request(self, request):
        block_ids=self.block_table.get_blocks(request.request_id)
        for block_id in block_ids:
            self.kv_cache.reset_block(block_id)
        self.block_manager.free(block_ids)
        self.block_table.remove_request(request.request_id)
        self.running_queue.remove(request)

    def release_request(self, request):
        if request.status=="WAITING" and any(req.request_id==request.request_id  for req in self.waiting_queue):
            self.waiting_queue.remove(next(req for req in self.waiting_queue if req.request_id == request.request_id))
        elif request.status=="RUNNING" and any(req.request_id==request.request_id  for req in self.running_queue):
            self._release_running_request(request)
        elif request.status=="ABORTED":
            if any(req.request_id==request.request_id  for req in self.waiting_queue):
                self.waiting_queue.remove(request)
            if any(req.request_id==request.request_id  for req in self.running_queue):
                self._release_running_request(request)

    def finish_request(self,request):
        if request.status == "RUNNING"and any(req.request_id==request.request_id  for req in self.running_queue):
            self.release_request(request)
            request.finish()
        else: raise ValueError("请求状态不合法。") 

    def abort_request(self,request):
        if any(req.request_id==request.request_id  for req in self.running_queue) :
            request.abort()
        if any(req.request_id==request.request_id  for req in self.waiting_queue):
            request.abort()
            self.release_request(request)



