# MiniLLM-Engine

从 0 手写一个大语言模型推理引擎（学习项目，持续开发中）。

目标是吃透 vLLM 的两项核心设计——**PagedAttention（分页 KV 缓存）** 与 **Continuous Batching（连续批处理）**，最终做出一个能同时服务多个请求的小推理引擎，并与参考实现做对照测试和吞吐 / 延迟基准。

当前进度：**单请求端到端生成已经跑通**（`Engine.generate()` 能真实输出文本），分页 KV / 块管理 / 调度器骨架、以及 greedy / random / top-k / top-p 采样均已实现；多请求并发组批、自研 KV 接入计算与 HTTP API 仍是下一步。

## 架构

```
            Engine
              │  add_request / step / generate
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

数据流：`Engine.add_request` 先做 tokenize 拿到 `prompt_token_ids` 与 `attention_mask`，封装成 `Request` 交给 `Scheduler` 排队；`Engine.step` 每次让调度器挑一个请求，按预估所需块数向 `BlockManager` 申请物理块并写入 `BlockTable`，随后走一次 prefill 或 decode 前向，取最后一个位置的 logits 交给 `Sampler` 采样出下一个 token，直到达到 `max_tokens` 或命中 EOS 才释放块。

## 目录结构

```
engine/
├── model/
│   ├── loader.py         # HF 权重 + tokenizer 加载（fp16 / cuda，tokenizer 左填充）
│   └── runner.py         # ModelRunner：prepare_input / forward（past_key_values 透传）
├── cache/
│   ├── kv_cache.py       # 分页 KV 存储 + 有效位图
│   ├── block_table.py    # 逻辑块表：request_id → 物理块 id 列表
│   └── block_manager.py  # 物理块分配器：free / used 集合
├── request.py            # 请求状态机：WAITING → RUNNING → FINISHED / ABORTED
├── scheduler.py          # 等待 / 运行队列、块计算与分配、结束释放
├── sampling/
│   └── sampler.py        # greedy / random / top-k / top-p 采样
├── server/
│   └── api.py            # OpenAI 兼容 API（待实现）
└── engine.py             # 引擎主循环：add_request / step / generate
tests/
├── test_engine.py        # 端到端推理冒烟脚本（可用）
└── test_model_runner.py  # 早期 prefill / decode 脚本（接口已过期）
benchmarks/               # 吞吐 / 延迟 / 显存（待实现）
examples/                 # 用法示例（待实现）
```

## 模块状态

已完成：

- [x] `ModelLoader`：HF 权重与 tokenizer 加载，统一 fp16 + CUDA
- [x] `ModelRunner`：prefill / decode，`past_key_values` 透传复用
- [x] `Request`：四态状态机 + `attention_mask` / `is_prefill` / `error` 字段
- [x] `KVCache`：分页读写 + 有效位图，越界与读未写入位置均抛错
- [x] `BlockManager`：物理块 allocate / free，块不足抛 `MemoryError`
- [x] `BlockTable`：请求到物理块的映射，拒绝重复添加同一物理块
- [x] `Scheduler`：waiting / running 双队列、按 `prompt_len + max_tokens` 预估块数、结束与取消时释放块
- [x] `Engine`：`add_request` / `step` / `generate`，单请求端到端生成可跑通
- [x] `Sampler`：greedy / random / top-k / top-p 四种策略，已由 `Engine.step` 调用

待实现：

- [ ] 真正的连续批处理：当前 `schedule()` 每次只放行一个请求、`step()` 每次只推进一步，尚未把多个请求拼成一个 batch
- [ ] 把自研 `KVCache` 接入前向计算：目前 KV 仍走 HF 的 `past_key_values`，分页缓存只做了分配与释放，尚未参与 attention
- [ ] OpenAI 兼容 HTTP API
- [ ] 单元测试补齐（`test_kv_cache.py`、`test_scheduler.py`、`test_concurrency.py` 目前为空）
- [ ] 修好 `test_model_runner.py`（仍调用已不存在的 `runner.load_model()`）
- [ ] 对照测试（vs 参考实现，比对 greedy 输出一致性）
- [ ] Benchmark：吞吐、TTFT、显存占用

## 快速开始

```bash
uv sync
uv run python tests/test_engine.py
```

`tests/test_engine.py` 是当前唯一能跑通的端到端脚本：**需要 GPU 和一份 HF 格式的小模型权重**，脚本里写死了 `model_path`（默认 `D:\Develop\Models\Qwen3-0.6B`）以及 KV 形状参数（`num_layers=28`、`num_kv_heads=8`、`head_dim=64`，对应 Qwen3-0.6B），换模型时要一起改。运行后会打印生成结果，以及 waiting / running 队列、free / used 块数、KV 缓存条目数，可直接用来检查资源是否泄漏。

环境：Python 3.11+（见 `.python-version`）、支持 CUDA 的 PyTorch、transformers、accelerate，依赖用 [uv](https://docs.astral.sh/uv/) 管理。

## 设计说明

**KVCache 布局**
物理 KV 内存一次性预分配为 `(num_layers, num_blocks, block_size, num_kv_heads, head_dim)`，key / value 各一份。`cache_valid` 是 `(num_layers, num_blocks, block_size)` 的布尔位图，按 token 槽位标记是否已写入；写入前清位、写入后置位，读到未写入的槽位直接抛错。`reset_block` 只清位图、不搬数据，块归还后即可复用——块分配因此不必关心层和头，只做纯粹的槽位簿记。

**BlockManager**
`free_block` / `used_block` 两个集合管理物理块。`allocate(n)` 在空闲块不足时抛 `MemoryError`；`free(block_ids)` 要求传入的块都在已使用集合中，避免重复归还把块“超卖”。返回的是 list 而非 set，保证块顺序稳定、便于与 BlockTable 的追加顺序对应。

**BlockTable**
请求到物理块 id 列表的映射。同一请求内重复添加同一物理块会被拒绝——这是分页复用的不变量：一个请求的块序列里不能自我重叠。

**Request 状态机**
`WAITING → RUNNING → FINISHED / ABORTED`。只有 RUNNING 状态允许追加 token：对 WAITING 追加说明调度流有问题（还没开始就产出了 token），对 FINISHED / ABORTED 追加会污染已结束的序列，两者都直接抛错。`append_token` 只负责追加，是否结束由引擎判断（`is_finished` 或 EOS），避免“达到上限”和“遇到 EOS”两条退出路径互相打架。

**Scheduler**
`waiting_queue` / `running_queue` 两个 deque。`add_request` 会查重，防止同一 request_id 重复入队。块数按最坏情况预估：`ceil((prompt_len + max_tokens) / block_size)`，这样生成过程中不需要中途重新分配，代价是短请求也会占满其最大可能的块。分配不到块时（`MemoryError`）把请求 `appendleft` 回等待队列，而不是丢弃；整体容量都不够时直接 `abort` 并记录 `error`，让上层能拿到明确的失败原因。`finish_request` 先释放块再切状态，避免释放失败时状态已不可回滚。

**Sampler**
`sample()` 按 `strategy` 分发到四种实现：`greedy` 直接 argmax；`random` 用 `logits / temperature` 后 softmax，再 `multinomial` 采样；`top-k` 先 `topk` 取前 k 个，其余位置置 `-inf` 再走随机采样；`top-p`（核采样）按概率降序累积，把累积概率超过 p 的位置屏蔽，但强制保留最高概率项（`remove_mask[:, 0] = False`），避免阈值过小时整行被清空。非法策略、`temperature <= 0`、`top_k` 非正整数、`top_p` 越界都会直接抛 `ValueError`。

**Engine**
`step()` 是核心：prefill 阶段直接喂 `prompt_token_ids`（带 attention_mask）；decode 阶段只喂上一个 token、把 mask 尾部追加一位，并复用 `kv_cache[request_id]` 里的 `past_key_values`。每步结束把新的 `past_key_values` 存回字典，请求结束后删除条目，防止显存泄漏。`generate()` 用 `while request.status in ("WAITING", "RUNNING")` 自旋推进，直到该请求结束再解码输出。

## Roadmap

1. `schedule()` 支持多请求组批 + `step()` 一次推进整批（连续批处理的核心）
2. 自研 `KVCache` 接入 attention，替换 HF 的 `past_key_values`
3. 采样参数打通到请求边界（目前 `Engine.step` 固定用默认 greedy，尚未按请求透传 `strategy` / `temperature` / `top_k` / `top_p`）
4. OpenAI 兼容 API
5. 对照测试 + Benchmark（吞吐 / TTFT / 显存）

## 已知事项

- `pyproject.toml` 目前只声明了 `accelerate`，`torch` / `transformers` 需自行安装（待补进依赖并区分 CPU / CUDA 版本）。
- 顶层 `engine/` 缺 `__init__.py`（`cache` / `model` / `sampling` / `server` 子包已有），且 `engine/server/__init__.py` 用的是 `from request import Request` / `from scheduler import Scheduler` / `from engine.engine import Engine` 这类绝对导入，直接当包导入会失败，需要改成相对导入并补齐顶层包结构。
- `tests/test_model_runner.py` 是重构前写的，调用的 `runner.load_model()` 已不存在，目前无法运行。
- `Engine` 的 `step()` 每次只处理一个请求，距离真正的 batch 推理还有一段距离。
- `ModelRunner` 把整个模型跑在 `device` 上，`attention_mask` 每步都在 CPU / GPU 间搬运，后续可优化。

## License

MIT
