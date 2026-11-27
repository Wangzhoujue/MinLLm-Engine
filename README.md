# MiniLLM-Engine

从 0 手写的大语言模型推理引擎（学习项目，持续开发中）。

目标是吃透 vLLM 的两项核心设计——**PagedAttention（分页 KV 缓存）** 与 **Continuous Batching（连续批处理）**，最终做出一个能同时服务多个请求、并对外暴露 OpenAI 兼容接口的小推理引擎。

当前进度：**单请求端到端生成、流式输出（SSE）、OpenAI 兼容 HTTP 服务都已跑通**。分页 KV 存储、块管理、调度器、采样器均已实现。尚未完成的是把多个请求拼成同一个 batch 的前向计算，以及让自研 `KVCache` 真正参与 attention——这两点也是本项目最终要验证的东西。

## 架构

```
        OpenAI SDK / curl
              │  POST /v1/chat/completions
              ▼
      server/api.py  ──▶ Detokenizer（增量解码，流式输出）
              │
              ▼
           Engine
              │  add_request / step / generate / generate_stream
              ▼
        Scheduler ──▶ BlockManager(物理块分配)
         │     │            │
         │     └────────────┴──▶ BlockTable(请求 → 物理块)
         ▼
    ModelRunner(prefill / decode) ◀──▶ KVCache(分页 KV 存储)
         │
         ▼
      Sampler(greedy / random / top-k / top-p)
```

数据流：`Engine.add_request` 先 tokenize 得到 `prompt_token_ids` 与 `attention_mask`，封装成 `Request` 交给 `Scheduler` 排队。`Engine.step` 每次让调度器放行/轮转一个请求，按预估所需块数向 `BlockManager` 申请物理块并写入 `BlockTable`，随后走一次 prefill 或 decode 前向，取最后一个位置的 logits 交给 `Sampler` 采样出下一个 token。命中 EOS 记 `finish_reason="stop"`，达到 `max_tokens` 记 `"length"`，随后释放块并清掉 `past_key_values`。

流式路径走 `Engine.generate_stream`：每产出一个 token 就交给 `Detokenizer` 增量解码，只把新增的文本片段作为事件吐出，再由 `server/api.py` 包装成 `chat.completion.chunk` 的 SSE。

## 目录结构

```
engine/
├── model/
│   ├── loader.py         # HF 权重 + tokenizer 加载（fp16，tokenizer 左填充）
│   └── runner.py         # ModelRunner：prepare_input / forward（past_key_values 透传）
├── cache/
│   ├── kv_cache.py       # 分页 KV 存储 + 有效位图
│   ├── block_table.py    # 逻辑块表：request_id → 物理块 id 列表
│   └── block_manager.py  # 物理块分配器：free / used 集合
├── request.py            # 请求状态机 + finish_reason
├── scheduler.py          # 等待 / 运行队列、块计算与分配、结束释放
├── sampling/
│   └── sampler.py        # greedy / random / top-k / top-p 采样
├── detokenizer.py        # 增量解码：把逐 token 的输出拼成不重复的文本增量
├── server/
│   ├── __main__.py       # 命令行入口（--model / --device / --dtype / --host / --port）
│   ├── api.py            # FastAPI 应用：/v1/chat/completions（含 SSE 流式）
│   └── schemas.py        # OpenAI 兼容的请求 / 响应 / chunk 模型
└── engine.py             # 引擎主循环：add_request / step / generate / generate_stream
tests/
├── test_engine.py        # 端到端推理冒烟脚本（可用）
└── test_model_runner.py  # 早期 prefill / decode 脚本（接口已过期）
benchmarks/               # 吞吐 / 延迟 / 显存（待实现）
examples/                 # 用法示例（待实现）
```

## 模块状态

已完成：

- [x] `ModelLoader`：HF 权重与 tokenizer 加载，统一 fp16，tokenizer 左填充
- [x] `ModelRunner`：prefill / decode，`past_key_values` 透传复用，`inference_mode` 下前向
- [x] `Request`：四态状态机 + `attention_mask` / `is_prefill` / `error` / `finish_reason`
- [x] `KVCache`：分页读写 + 有效位图，越界与读未写入位置均抛错
- [x] `BlockManager`：物理块 allocate / free，块不足抛 `MemoryError`
- [x] `BlockTable`：请求到物理块的映射，拒绝重复添加同一物理块
- [x] `Scheduler`：waiting / running 双队列、按 `prompt_len + max_tokens` 预估块数、结束与取消时释放块
- [x] `Sampler`：greedy / random / top-k / top-p 四种策略，已由 `Engine.step` 调用
- [x] `Engine`：`add_request` / `step` / `generate`，单请求端到端生成可跑通
- [x] `Detokenizer`：按 token 增量解码，流式场景下不重复、不丢尾
- [x] `Engine.generate_stream`：以生成器形式逐步产出文本增量与 token 计数
- [x] `server`：FastAPI + `/v1/chat/completions`，支持非流式与 SSE 流式、`stream_options.include_usage`

待实现：

- [ ] 真正的连续批处理：`schedule()` 每次只放行一个请求，`step()` 也只对一个请求做前向，尚未把多个请求拼成一个 batch
- [ ] 把自研 `KVCache` 接入前向计算：目前 KV 仍走 HF 的 `past_key_values`，分页缓存只做分配与释放，尚未参与 attention
- [ ] 采样参数打通到请求边界（见「已知事项」第一条）
- [ ] 单元测试补齐（`test_kv_cache.py`、`test_scheduler.py`、`test_concurrency.py` 目前为空）
- [ ] 修好 `test_model_runner.py`（仍调用已不存在的 `runner.load_model()`）
- [ ] 对照测试（vs 参考实现，比对 greedy 输出一致性）
- [ ] Benchmark：吞吐、TTFT、显存占用

## 快速开始

依赖已全部声明在 `pyproject.toml`（torch / torchvision / torchaudio 走 `pytorch-cu134` 索引，另有 transformers、accelerate、fastapi、pydantic、uvicorn），Python 3.11+（见 `.python-version`）：

```bash
uv sync
```

**方式一：端到端脚本**

```bash
uv run python tests/test_engine.py
```

这是当前最直接的冒烟测试：**需要 GPU 和一份 HF 格式的小模型权重**。脚本里写死了 `model_path`（默认 `D:\Develop\Models\Qwen3-0.6B`）以及 KV 形状参数（`num_layers=28`、`num_kv_heads=8`、`head_dim=64`，对应 Qwen3-0.6B），换模型时要一起改。运行后会打印生成结果，以及 waiting / running 队列长度、free / used 块数、KV 缓存条目数，可直接用来检查资源是否泄漏。

**方式二：起 HTTP 服务**

```bash
uv run python -m engine.server --model "D:/Develop/Models/Qwen3-0.6B" --device cuda --dtype float16 --port 8000
```

`--dtype` 可选 `float16` / `float32` / `bfloat16`，`--host` 默认 `0.0.0.0`。服务启动时在 lifespan 里一次性构建好 ModelRunner、KVCache、BlockManager、BlockTable、Scheduler、Sampler 和 Engine，之后每个请求复用同一个 Engine。

非流式：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
```

流式：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"你好"}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

服务端会先用 `tokenizer.apply_chat_template(..., add_generation_prompt=True)` 把 messages 渲染成 prompt，请求体里的 `messages` 因此支持 system / user / assistant / tool 等 role。

## 设计说明

**KVCache 布局**
物理 KV 内存一次性预分配为 `(num_layers, num_blocks, block_size, num_kv_heads, head_dim)`，key / value 各一份。`cache_valid` 是 `(num_layers, num_blocks, block_size)` 的布尔位图，按 token 槽位标记是否已写入；写入前清位、写入后置位，读到未写入的槽位直接抛错。`reset_block` 只清位图、不搬数据，块归还后即可复用——块分配因此不必关心层和头，只做纯粹的槽位簿记。

**BlockManager**
`free_block` / `used_block` 两个集合管理物理块。`allocate(n)` 在空闲块不足时抛 `MemoryError`；`free(block_ids)` 要求传入的块都在已使用集合中，避免重复归还把块"超卖"。返回的是 list 而非 set，保证块顺序稳定、便于与 BlockTable 的追加顺序对应。

**BlockTable**
请求到物理块 id 列表的映射。同一请求内重复添加同一物理块会被拒绝——这是分页复用的不变量：一个请求的块序列里不能自我重叠。

**Request 状态机**
`WAITING → RUNNING → FINISHED / ABORTED`。只有 RUNNING 状态允许追加 token：对 WAITING 追加说明调度流有问题（还没开始就产出了 token），对 FINISHED / ABORTED 追加会污染已结束的序列，两者都直接抛错。`append_token` 只负责追加，是否结束由引擎判断（`is_finished` 或 EOS），避免"达到上限"和"遇到 EOS"两条退出路径互相打架；退出原因写进 `finish_reason`，直接映射到 OpenAI 的 `finish_reason` 字段。

**Scheduler**
`waiting_queue` / `running_queue` 两个 deque。`add_request` 会查重，防止同一 request_id 重复入队。块数按最坏情况预估：`ceil((prompt_len + max_tokens) / block_size)`，这样生成过程中不需要中途重新分配，代价是短请求也会占满其最大可能的块。分配不到块时（`MemoryError`）把请求 `appendleft` 回等待队列，而不是丢弃；所需块数超过总容量这种不可恢复的情况，直接 `abort` 并记录 `error`，让上层能拿到明确的失败原因。`finish_request` 先释放块再切状态，避免释放失败时状态已不可回滚；`get_next_request` 取队首后立刻追加到队尾，于是多个正在运行的请求会轮转推进。

**Sampler**
`sample()` 按 `strategy` 分发到四种实现：`greedy` 直接 argmax；`random` 用 `logits / temperature` 后 softmax，再 `multinomial` 采样；`top-k` 先 `topk` 取前 k 个，其余位置置 `-inf` 再走随机采样；`top-p`（核采样）按概率降序累积，把累积概率超过 p 的位置屏蔽，但强制保留最高概率项（`remove_mask[:, 0] = False`），避免阈值过小时整行被清空。非法策略、`temperature <= 0`、`top_k` 非正整数、`top_p` 越界都会直接抛 `ValueError`。

**Engine**
`step()` 是核心：prefill 阶段直接喂 `prompt_token_ids`（带 attention_mask）；decode 阶段只喂上一个 token、把 mask 尾部追加一位，并复用 `kv_cache[request_id]` 里的 `past_key_values`。每步结束把新的 `past_key_values` 存回字典，请求结束后删除条目，防止显存泄漏。`generate()` 用 `while request.status in ("WAITING", "RUNNING")` 自旋推进，返回 `(文本, finish_reason, prompt_tokens, completion_tokens)` 四元组，恰好够拼一个 OpenAI 响应。

**流式输出与增量解码**
流式不能每个 token 都整段重新 decode 后发出去，否则会重复且产生大量无效字符。`Detokenizer` 记录已解码的全量文本和已发出的文本，`push(token_id)` 只返回"新解码结果比已发出部分多出来的那一段"，`flush()` 在结束时把尾部残差补齐。`generate_stream` 因此能保证：拼起来的事件文本等于最终整段解码结果，且每个事件都非空才发送。

**OpenAI 兼容服务层**
`schemas.py` 用 pydantic 定义了请求与响应模型（`ChatCompletionRequest` / `ChatCompletionResponse` / `ChatCompletionChunk` 等），`api.py` 只做协议转换和 SSE 拼装，业务全部委托给 `Engine`。这样 HTTP 层可替换、可单测，引擎本身不依赖 Web 框架。流式响应遵循 OpenAI 的 chunk 顺序：先一个只含 `role` 的 delta，随后若干 `content` delta，最后一个带 `finish_reason` 的空 delta，若开了 `include_usage` 再补一个只带 `usage` 的 chunk。

## Roadmap

1. `schedule()` 支持多请求组批 + `step()` 一次推进整批（连续批处理的核心）
2. 自研 `KVCache` 接入 attention，替换 HF 的 `past_key_values`
3. 采样参数打通到请求边界（HTTP 的 `temperature` / `top_p` 一路传到 `Sampler.sample`）
4. 对照测试 + Benchmark（吞吐 / TTFT / 显存）
5. Server 层去掉硬编码：模型名、KV 形状都改成从模型配置里读

## 已知事项

- **`temperature` 目前是"通而不达"**：`Engine.step(temperature)` 确实把值传给了 `Sampler.sample`，但`strategy` 仍取默认的 `"greedy"`，而 greedy 分支根本不看 temperature；同时 `server/api.py` 也没把请求体里的 `temperature` 往下传。所以现在无论传不传，实际都是 greedy。
- **server 里有硬编码**：响应里的 `model` 字段固定写成 `"Qwen3-0.6B"`，KV 的 `num_layers=28` / `num_kv_heads=8` / `head_dim=64` 与 `num_blocks=128` / `block_size=16` 也是写死的，换模型必须改代码。
- **`Detokenizer.push` 在字节对不完整时会返回空串**（`decoded_text` 不以 `emitted_text` 开头时直接 `return ""` 且不推进 `emitted_text`），依赖 `flush()` 兜底；对中文等场景通常是安全的，极端情况下会延后一个 token 才对上。
- `Engine.step()` 一次只对一个请求做前向，多个请求只是"轮流推进"，不是同一个 batch——这是距离 vLLM 最远的一步。
- 自研 `KVCache` 仍只做块簿记，真正的 KV 数据还在 HF 的 `past_key_values` 里，显存不受分页管理。
- `ModelRunner` 把整个模型跑在 `device` 上，`attention_mask` 每步都在 CPU / GPU 间搬运，后续可优化。
- `tests/test_model_runner.py` 是重构前写的，调用的 `runner.load_model()` 已不存在，目前无法运行；`tests/test_engine.py` 里也还多此一举地单独 new 了一次 `ModelLoader` 并重复加载权重。
- `benchmarks/`、`examples/` 仍是空文件。

## License

MIT
