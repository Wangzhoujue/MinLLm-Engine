
from dataclasses import dataclass,field


@dataclass
class Detokenizer:
    tokenizer:object
    token_ids :list=field(default_factory=list)
    decoded_text: str = ""
    emitted_text: str = ""


    def push(self, token_id) -> str:
        self.token_ids.append(token_id)
        self.decoded_text=self.tokenizer.decode(self.token_ids)
        if self.decoded_text.startswith(self.emitted_text):
            text=self.decoded_text[len(self.emitted_text):]
            self.emitted_text=self.decoded_text
            return text
        return ""
        
        
        

    def flush(self) -> str:
        text=self.decoded_text[len(self.emitted_text):]
        self.emitted_text=self.decoded_text
        return text