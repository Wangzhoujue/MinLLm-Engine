import torch
from transformers import AutoModelForCausalLM,AutoTokenizer



class ModelLoader():
    def __init__(self,model_path,device="cuda",dtype=torch.float16):
        self.model_path=model_path
        self.device=device
        self.dtype=dtype


    def load_model(self):
        model=AutoModelForCausalLM.from_pretrained(
        self.model_path,
        dtype=self.dtype,
        device_map=self.device)
            
        print(f"模型加载成功!加载于{next(model.parameters()).device}.")
        model.eval()
        return model
    
    def load_tokenizer(self):
        tokenizer=AutoTokenizer.from_pretrained(
            self.model_path
        )
        tokenizer.padding_side = "left"
        return tokenizer