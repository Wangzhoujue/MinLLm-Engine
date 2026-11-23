
import torch

from engine.engine import Engine
from engine.scheduler import Scheduler
from engine.model.runner import ModelRunner,ModelLoader
from engine.cache.kv_cache import KVCache
from engine.cache.block_manager import BlockManager
from engine.cache.block_table import BlockTable


def main():
    # 1. 基础配置
    model_path = r"D:\Develop\Models\Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # 2. 初始化 ModelRunner
    model_runner = ModelRunner(
        model_path=model_path,
        device=device,
        dtype=dtype
    )
    model_loader = ModelLoader(
        model_path=model_path,
            device=device,
            dtype=dtype)
    model_loader.load_model()
    model_loader.load_tokenizer()

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

    # 6. 初始化 Engine
    engine = Engine(
        model_runner=model_runner,
        scheduler=scheduler
    )

    # 7. 执行推理
    prompt = "请介绍一下人工智能。"
    max_tokens = 30

    print(f"Prompt: {prompt}")
    print("正在生成...")

    result = engine.generate(prompt, max_tokens)

    print("\n生成结果：")
    print(result)

    # 8. 检查资源释放
    print("\n========== 资源状态 ==========")
    print(f"Waiting Queue: {len(scheduler.waiting_queue)}")
    print(f"Running Queue: {len(scheduler.running_queue)}")
    print(f"Free Blocks: {len(block_manager.free_block)}")
    print(f"Used Blocks: {len(block_manager.used_block)}")
    print(f"Transformers KV Cache: {len(engine.kv_cache)}")


if __name__ == "__main__":
    main()
