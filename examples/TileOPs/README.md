# TileOPs

Standalone NPU benchmark framework, extracted from [TileOPs](https://github.com/tile-ai/TileOPs)
(GPU/TileLang-based) and adapted for NPU backends (Ascend via `torch_npu`).

**No dependency on the TileOPs repository.** All code is self-contained.

## Structure

```
TileOPs/
├── tileops/              # Main package
│   ├── device.py           # Device backend abstraction (GPU→NPU adaptation surface)
│   ├── utils/              # Utilities (str2dtype, etc.)
│   ├── manifest/           # Op manifest (standalone YAML spec)
│   ├── workloads/          # Workload definitions (input generation)
│   ├── ops/                # Op layer (validation, reshape, kernel dispatch, roofline)
│   ├── kernels/            # Kernel layer (NPU implementation)
│   ├── testing/            # Test base (correctness vs PyTorch reference)
│   └── benchmark/          # Benchmark base (latency / TFLOPS / bandwidth)
├── tests/                  # Correctness tests
├── benchmarks/             # Performance benchmarks
├── docs/
│   └── gpu_to_npu_adaptation.md   # GPU→NPU adaptation points
└── pyproject.toml
```

## Quick Start

```bash
cd TileOPs
pip install -e .[dev]

# Run correctness tests
pytest tests/ -v

# Run benchmarks
pytest benchmarks/ -v
```

## Adding a New Op

See `.claude/skills/add-npu-op/SKILL.md` for the step-by-step guide.
