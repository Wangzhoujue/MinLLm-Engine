
import torch
import uuid
import time
import json
from fastapi import FastAPI,Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from engine.engine import Engine
from engine.detokenizer import Detokenizer
from engine.sampling import Sampler
from engine.scheduler import Scheduler
from engine.model.runner import ModelRunner,ModelLoader
from engine.cache.kv_cache import KVCache
from engine.cache.block_manager import BlockManager
from engine.cache.block_table import BlockTable
from .schemas import ChatMessage,ChatCompletionRequest,ChatCompletionMessage,ChatCompletionChoice,UsageInfo,ChatCompletionResponse,ChatCompletionDelta,ChatCompletionChunkChoice,ChatCompletionChunk

def create_app(model_path,device,dtype):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 服务启动时执行

        # 初始化 ModelRunner
        model_runner = ModelRunner(
            model_path=model_path,
            device=device,
            dtype=dtype
        )

        # 初始化 Detokenizer
        detokenizer=Detokenizer(
            tokenizer=model_runner.tokenizer
            )

        # 初始化 KV Cache
        kv_cache = KVCache(
            num_blocks=128,
            block_size=16,
            num_layers=28,
            num_kv_heads=8,
            head_dim=64,
            dtype=dtype,
            device=device
        )

        # 初始化 Block 管理
        block_manager = BlockManager(num_block=128)
        block_table = BlockTable()

        # 初始化 Scheduler
        scheduler = Scheduler(
            kv_cache=kv_cache,
            block_table=block_table,
            block_manager=block_manager
        )

        # 初始化 Sampler
        sampler=Sampler()

        # 初始化 Engine
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

        if request.stream:
            include_usage = bool(
                request.stream_options and request.stream_options.include_usage
            )
            return StreamingResponse(
                stream_response(
                    engine, prompt, request.max_tokens,
                    response_id, created, model_name, include_usage
                ),
                media_type="text/event-stream"
            )
            
        else:
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
    
    def stream_response(engine, prompt, max_tokens, response_id, created, model_name, include_usage=False):
        prompt_tokens = len(engine.model_runner.tokenizer.encode(prompt))
        completion_tokens = 0

        delta = ChatCompletionDelta(role="assistant")
        choice = ChatCompletionChunkChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=model_name,
            choices=[choice]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        finish_reason = None

        for event in engine.generate_stream(prompt, max_tokens):
            if event["text"] is not None:
                delta = ChatCompletionDelta(content=event["text"])
                choice = ChatCompletionChunkChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None
                )
                chunk = ChatCompletionChunk(
                    id=response_id,
                    created=created,
                    model=model_name,
                    choices=[choice]
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
            else:
                finish_reason = event["finish_reason"]
                completion_tokens = event["completion_tokens"]

        choice = ChatCompletionChunkChoice(
            index=0,
            delta=ChatCompletionDelta(),
            finish_reason=finish_reason
        )
        chunk = ChatCompletionChunk(
            id=response_id,
            created=created,
            model=model_name,
            choices=[choice]
        )
        yield f"data: {chunk.model_dump_json()}\n\n"

        if include_usage:
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
            chunk = ChatCompletionChunk(
                id=response_id,
                created=created,
                model=model_name,
                choices=[],
                usage=usage
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    return app







