
import torch
from typing import Literal
from dataclasses import dataclass, field

@dataclass
class Request:
    request_id:int
    prompt:str
    max_tokens:int
    prompt_token_ids:torch.Tensor#[batch,slen]
    attention_mask:torch.Tensor
    finish_reason:Literal["stop","length","abort","error"]| None = None
    output_token_ids:list[int]=field(default_factory=list)
    status:Literal["WAITING","RUNNING","ABORTED","FINISHED"]="WAITING"
    is_prefill: bool = True
    error: str | None = None

    @property
    def generated_tokens(self):
        return len(self.output_token_ids)
    
    
    def is_finished(self,token,eos_token):
        if token==eos_token:
            self.finish_reason="stop"
            return True
        elif self.generated_tokens>=self.max_tokens:
            self.finish_reason="length"
            return True
        return False

    def append_token(self, token_id):
        if self.status=="RUNNING":
            self.output_token_ids.append(token_id)
        elif self.status=="WAITING":raise RuntimeError("该请求处于WAITING状态。")
        elif self.status=="FINISHED":raise RuntimeError("该请求已完成。")
        elif self.status=="ABORTED": raise RuntimeError("该请求处于取消状态。")

    def start(self):
        if self.status=="WAITING":
            self.status="RUNNING"

    def abort(self):
        if self.status not in ["ABORTED","FINISHED"]:
            self.status="ABORTED"
            self.finish_reason="abort"

    def finish(self):
        if self.status=="RUNNING":
            self.status="FINISHED"

        