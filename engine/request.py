from dataclasses import dataclass, field

@dataclass
class Request:
    request_id:int
    prompt:str
    max_tokens:int
    prompt_token_ids:list[int]
    output_token_ids:list[int]=field(default_factory=list)
    status:str="WAITING"

    @property
    def generated_tokens(self):
        return len(self.output_token_ids)
    
    @property
    def is_finished(self):
        return self.generated_tokens>=self.max_tokens

    def append_token(self, token_id):
        if self.status=="RUNNING":
            self.output_token_ids.append(token_id)
            if self.is_finished: self.status="FINISHED"
        elif self.status=="WAITING":raise RuntimeError("该请求处于WAITING状态。")
        elif self.status=="FINISHED":raise RuntimeError("该请求已达最大生成token数。")
        elif self.status=="ABORTED": raise RuntimeError("该请求处于取消状态。")

    def start(self):
        if self.status!="RUNNING"and self.status!="FINISHED":
            self.status="RUNNING"

    def abort(self):
        if self.status!="ABORTED" and self.status!="FINISHED":
            self.status="ABORTED"

    def finish(self):
        if self.status=="RUNNING":
            self.status="FINISHED"

        