
import torch
import uuid
import time
from fastapi import FastAPI,Request
from contextlib import asynccontextmanager

from engine.engine import Engine
from engine.sampling import Sampler
from engine.scheduler import Scheduler
from engine.model.runner import ModelRunner,ModelLoader
from engine.cache.kv_cache import KVCache
from engine.cache.block_manager import BlockManager
from engine.cache.block_table import BlockTable
from .schemas import ChatMessage,ChatCompletionRequest,ChatCompletionMessage,ChatCompletionChoice,UsageInfo,ChatCompletionResponse

def create_app(model_path: str,device,dtype):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 服务启动时执行

        # 2. 初始化 ModelRunner
        model_runner = ModelRunner(
            model_path=model_path,
            device=device,
            dtype=dtype
        )

        # 3. 初始化 KV Cache
        kv_cache = KVCache(
            num_blocks=128,
            block_size=16,
            num_layers=28,
            num_kv_heads=8,
            head_dim=64,
            dtype=dtype,
            device=device
        )

        # 4. 初始化 Block 管理
        block_manager = BlockManager(num_block=128)
        block_table = BlockTable()

        # 5. 初始化 Scheduler
        scheduler = Scheduler(
            kv_cache=kv_cache,
            block_table=block_table,
            block_manager=block_manager
        )

        #5. 初始化 Sampler
        sampler=Sampler()

        # 7. 初始化 Engine
        engine = Engine(
            model_runner=model_runner,
            scheduler=scheduler,
            sampler=sampler
        )

        app.state.engine = engine

        print("服务启动")

        yield
        # 服务关闭时执行
        print("服务关闭")
    app = FastAPI(lifespan=lifespan)
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        engine = http_request.app.state.engine
        prompt = engine.model_runner.tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True
        )

        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        object = "chat.completion"
        created = int(time.time())
        model_name = "Qwen3-0.6B"


        outputs,finish_reason,prompt_tokens,completion_tokens = engine.generate(
            prompt=prompt,
            max_tokens=request.max_tokens,
        )

        total_tokens = prompt_tokens + completion_tokens
        choices = [
        ChatCompletionChoice(
            index=0,
            message=ChatCompletionMessage(
                role="assistant",
                content=outputs
            ),
            finish_reason=finish_reason
        )
        ]   

        usage = UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens
        )
        chat_completion_response=ChatCompletionResponse(
                                                        id=response_id,
                                                        object=object,
                                                        created=created,
                                                        model=model_name,
                                                        choices=choices,
                                                        usage=usage
                                                        )
        return chat_completion_response

    return app







