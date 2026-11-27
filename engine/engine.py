
import torch
from dataclasses import dataclass, field
from .model import ModelRunner
from .scheduler import Scheduler
from .request import Request
from .sampling import Sampler
from .detokenizer import Detokenizer

@dataclass
class Engine:
    model_runner:ModelRunner
    scheduler:Scheduler
    sampler: Sampler = field(default_factory=Sampler)
    request_counter:int=0
    kv_cache :dict=field(default_factory=dict)

    @property
    def has_unfinished_requests(self):
        return bool(self.scheduler.waiting_queue or self.scheduler.running_queue)

    def add_request(self,prompt,max_tokens):
        prompt_ids,attention_mask=self.model_runner.prepare_input(prompt)
        prompt_ids = prompt_ids.cpu()
        attention_mask = attention_mask.cpu()
        request=Request(self.request_counter,prompt,max_tokens,prompt_ids,attention_mask)
        self.scheduler.add_request(request)
        self.request_counter+=1
        return request

    def step(self,temperature=None):
        self.scheduler.schedule()
        if not self.scheduler.running_queue:
            return
        request=self.scheduler.get_next_request()
        if request.is_prefill:
            input_ids=request.prompt_token_ids.to(self.model_runner.device)
            attention_mask=request.attention_mask.to(self.model_runner.device)
            outputs=self.model_runner.forward(input_ids,attention_mask)
        else:
            input_ids= torch.tensor([[request.output_token_ids[-1]]]).to(self.model_runner.device)
            new_mask = request.attention_mask.new_ones((1, 1))
            request.attention_mask=torch.cat([request.attention_mask,new_mask],dim=-1).cpu()
            attention_mask=request.attention_mask.to(self.model_runner.device)
            outputs=self.model_runner.forward(input_ids,attention_mask,self.kv_cache[request.request_id])
        next_token_logits=outputs.logits[:,-1,:]
        if temperature is None:
            next_token=self.sampler.sample(next_token_logits)
        else:
            next_token=self.sampler.sample(next_token_logits,temperature=temperature)
        request.append_token(next_token.item())
        request.is_prefill=False
        self.kv_cache[request.request_id]=outputs.past_key_values
        if next_token.item()==self.model_runner.tokenizer.eos_token_id:
            request.finish_reason="stop"
            self.scheduler.finish_request(request)
            del self.kv_cache[request.request_id]
        elif request.is_finished :
            request.finish_reason="length"
            self.scheduler.finish_request(request)
            del self.kv_cache[request.request_id]


    def generate(self,prompt,max_tokens,temperature=None):
        request=self.add_request(prompt,max_tokens)
        while request.status in ("WAITING", "RUNNING"):
            self.step(temperature)
        if request.status == "ABORTED":
            raise RuntimeError(request.error)
        outputs=self.model_runner.tokenizer.decode(request.output_token_ids,
                                                skip_special_tokens=True)
        return outputs,request.finish_reason,len(request.prompt_token_ids),len(request.output_token_ids)

    def generate_stream(self,prompt,max_tokens):
        detokenizer=Detokenizer(self.model_runner.tokenizer)
        request=self.add_request(prompt,max_tokens)
        num_token=0
        while request.status in ("WAITING", "RUNNING"):
            self.step()
            if len(request.output_token_ids)<=num_token:continue
            token=request.output_token_ids[num_token:]
            num_token+=len(token)
            for token_id in token:
                text=detokenizer.push(token_id)
                if text:
                    yield { "text": text,
                            "completion_tokens": len(request.output_token_ids),
                            "finish_reason":None}
        if request.status == "ABORTED":
            raise RuntimeError(request.error)
        text=detokenizer.flush()
        if text:
            yield {"text": text,
                "finish_reason":None}
        yield { "text": None,
                "completion_tokens": len(request.output_token_ids),
                "finish_reason":request.finish_reason}
        




            







