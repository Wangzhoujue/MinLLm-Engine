# MiniLLM-Engine

从 0 手写一个大语言模型推理引擎（学习项目，持续开发中）。

目标是吃透 vLLM 的两项核心设计——**PagedAttention（分页 KV 缓存）** 与 **Continuous Batching（连续批处理）**，最终做出一个能同时服务多个请求的小推理引擎，并与参考实现做对照测试和吞吐 / 延迟基准。

## 架构

```
Request ──▶ Scheduler(连续批处理) ──▶ BlockManager(物理块分配)
              │                          │
              │                          ▼
              │                     BlockTable(请求 → 物理块)
              ▼                          │
        ModelRunner(prefill / decode) ◀──┴──▶ KVCache(分页 KV 存储)
              │
              ▼
           Sampler
```

数据流：请求先进入调度器排队，调度器为它分配物理块、建好块表，再把一批请求拼成 batch 交给 `ModelRunner` 前向，采样器产出下一个 token 后回写状态，直到达到 `max_tokens`。

## 目录结构

```
engine/
├── model/
│   ├── loader.py         # HF 权重 + tokenizer 加载（fp16 / cuda，tokenizer 左填充）
│   └── runner.py         # ModelRunner：prepare_input / forward（past_key_values 复用）
├── cache/
│   ├── kv_cache.py       # 分页 KV 存储 + 有效位图
│   ├── block_table.py    # 逻辑块表：request_id → 物理块 id 列表
│   └── block_manager.py  # 物理块分配器：free / used 集合
├── request.py            # 请求状态机：WAITING → RUNNING → FINISHED / ABORTED
├── scheduler.py          # 请求队列与批组装（待实现）
├── sampling/
│   └── sampler.py        # greedy / top-k（待实现）  
├── server/
│   └── api.py            # OpenAI 兼容 API（待实现）
└── engine.py             # 引擎主循环（待实现）
tests/                    # 模块测试（目前仅 test_model_runner.py 有内容）
benchmarks/               # 吞吐 / 延迟 / 显存（待实现）
examples/                 # 用法示例（待实现）
```

## 模块状态

已完成：

- [x] `ModelLoader`：HF 权重与 tokenizer 加载，统一 fp16 + CUDA
- [x] `ModelRunner`：prefill / decode，显式传入 `past_key_values` 复用 KV
- [x] `Request`：四态状态机，非 RUNNING 状态追加 token 直接抛错
- [x] `KVCache`：分页读写 + 有效位图，越界与读未写入位置均抛错
- [x] `BlockManager`：物理块 allocate / free，块不足抛 `MemoryError`
- [x] `BlockTable`：请求到物理块的映射，拒绝重复添加同一物理块

待实现：

- [ ] `Scheduler`：连续批处理调度（在跑请求与等待队列的迭代式组批）
- [ ] `Sampler`：greedy / top-k
- [ ] `Engine`：prefill → decode 主循环
- [ ] OpenAI 兼容 HTTP API
- [ ] 各模块单元测试补齐（`test_kv_cache.py` 等目前为空文件）
- [ ] 对照测试（vs 参考实现，比对 greedy 输出一致性）
- [ ] Benchmark：吞吐、TTFT、显存占用

## 快速开始

```bash
uv sync                       # 创建虚拟环境并安装依赖
uv run python tests/test_model_runner.py
```

`tests/test_model_runner.py` 是手动冒烟脚本：**需要本地 GPU 和一份 HF 格式的小模型权重**，脚本顶部的 `Modelpath` 要改成你自己的路径，运行后会打印 prefill / decode 各步的 logits 形状与 KV 缓存长度。

> ⚠️ 该脚本仍是早期手写版本：尚未接入 pytest，且仍调用了重构前的接口（如 `runner.load_model()`），直接运行前需先与当前 `ModelRunner` 对齐。

环境：Python 3.11+（见 `.python-version`）、支持 CUDA 的 PyTorch、transformers、accelerate，依赖用 [uv](https://docs.astral.sh/uv/) 管理。

## 设计说明

**KVCache 布局**
物理 KV 内存一次性预分配为 `(num_layers, num_blocks, block_size, num_kv_heads, head_dim)`，key / value 各一份。`cache_valid` 是 `(num_layers, num_blocks, block_size)` 的布尔位图，按 token 槽位标记是否已写入；写入前清位、写入后置位，读到未写入的槽位直接抛错。`reset_block` 只清位图、不搬数据，块归还后即可复用——块分配因此不必关心层和头，只做纯粹的槽位簿记。

**BlockManager**
`free_block` / `used_block` 两个集合管理物理块。`allocate(n)` 在空闲块不足时抛 `MemoryError`；`free(block_ids)` 要求传入的块都在已使用集合中，避免重复归还把块"超卖"。

**BlockTable**
请求到物理块 id 列表的映射。同一请求内重复添加同一物理块会被拒绝——这是分页复用的不变量：一个请求的块序列里不能自我重叠。

**Request 状态机**
`WAITING → RUNNING → FINISHED / ABORTED`。只有 RUNNING 状态允许追加 token：对 WAITING 追加说明调度流有问题（还没开始就产出了 token），对 FINISHED / ABORTED 追加会污染已结束的序列，两者都直接抛错。

**ModelRunner**
`prepare_input` 负责 tokenize 并左填充（自回归生成必须左填充，才能保证序列最后一个位置是真实 token）；`forward` 在 `torch.inference_mode()` 下调用 HF 模型并透传 `past_key_values`，避免每步重算历史 KV。目前仍复用 HF 自带的 KV 缓存，后续会换成上面这套自研分页实现。

## Roadmap

大致按依赖顺序：

1. `Sampler`（greedy / top-k）——补上最小可用闭环
2. `Engine` 主循环，跑通单请求端到端生成
3. `Scheduler` 连续批处理，跑通多请求并发
4. `ModelRunner` 接入自研 `KVCache`，脱离 HF 的 `past_key_values`
5. OpenAI 兼容 API + 对照测试 + Benchmark

## 已知事项

- `pyproject.toml` 目前只声明了 `accelerate`，`torch` / `transformers` 需自行安装（待补进依赖并区分 CPU / CUDA 版本）。
- 尚未添加 `__init__.py`，包内模块依赖工作目录导入，后续会补成标准可安装包。

## License

MIT
