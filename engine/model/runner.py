import torch
from engine.model.loader import ModelLoader



class ModelRunner():
    def __init__(self,model_path,device="cuda",dtype=torch.float16):
        self.model_path=model_path
        self.device=device
        self.dtype=dtype
        self.model=None
        self.tokenizer=None

        self.loader=ModelLoader(self.model_path,self.device,self.dtype)
        self.model=self.loader.load_model()
        self.tokenizer=self.loader.load_tokenizer()


    def prepare_input(self,prompt):
        if self.tokenizer is not None:
            data=self.tokenizer(prompt,
                                return_tensors="pt",
                                padding=True)
            prepared_input=data["input_ids"].to(self.device)
            attention_mask=data["attention_mask"].to(self.device)
            return prepared_input,attention_mask
        raise RuntimeError("分词器尚未加载。")

    
    def forward(self,input_ids,attention_mask,past_key_values=None):
        if self.model is not None:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=past_key_values
                )
            
            return outputs
        raise RuntimeError("模型尚未加载。")
