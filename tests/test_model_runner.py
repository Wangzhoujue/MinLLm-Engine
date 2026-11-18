import torch
from engine.model.runner import ModelRunner

Modelpath=r"D:\Develop\Models\Qwen3-0.6B"

runner=ModelRunner(Modelpath)
Model=runner.load_model()
Tokenizer=runner.load_tokenizer()

texts = [
    "你好",
    "你好，我正在学习如何开发一个大语言模型推理引擎"
]

input_ids, attention_mask = runner.prepare_input(texts)

print("Input Shape:", input_ids.shape)
print("Attention Mask Shape:", attention_mask.shape)
print("Input IDs:", input_ids)
print("Attention Mask:", attention_mask)
outputs=runner.forward(input_ids,attention_mask)
print("Input Shape:", input_ids.shape)
print("Logits Shape:", outputs.logits.shape)
print("Cache Length:", outputs.past_key_values.get_seq_length())

i=0
while(i<10):
    past_key_values = outputs.past_key_values

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    new_mask = torch.ones(
        (attention_mask.shape[0], 1),
        device=attention_mask.device,
        dtype=attention_mask.dtype
    )
    attention_mask = torch.cat([attention_mask, new_mask], dim=1)

    decode_outputs = runner.forward(
        input_ids=next_token,
        attention_mask=attention_mask,
        past_key_values=past_key_values
    )
    Outputs=decode_outputs
    print("Decode Cache Length:", decode_outputs.past_key_values.get_seq_length())
    i+=1
