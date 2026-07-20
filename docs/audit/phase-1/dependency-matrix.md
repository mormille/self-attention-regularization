# Phase 1 Preliminary Dependency Matrix

## Confidence levels

- **Confirmed:** directly present in imports or notebook metadata.
- **Strongly indicated:** API usage identifies a major version family, but no lockfile exists.
- **Unknown:** cannot be recovered reliably from the repository alone.

## Historical runtime

| Dependency / platform | Evidence | Estimated historical requirement | Confidence | Notes |
|---|---|---|---|---|
| Python | all three notebook metadata blocks | Python 3.8.2 | Confirmed | The only exact software version recorded in the source. |
| PyTorch | `torch`, `nn.MultiheadAttention`, distributed APIs | likely PyTorch 1.x from the 2020–2022 period | Strongly indicated | Exact version unknown; API and checkpoint compatibility require testing. |
| torchvision | transforms, datasets, models and functional imports | version matched to historical PyTorch | Strongly indicated | Exact version unknown. |
| FastAI | `fastai.vision.all`, `DataBlock`, `Learner`, `to_distributed`, `distrib_ctx`, GAN APIs | FastAI v2 family | Strongly indicated | Exact release is critical because SAM copies and overrides internal methods. |
| fastprogress | direct imports | historical FastAI-compatible release | Confirmed dependency; version unknown | May have arrived transitively through FastAI. |
| NumPy | direct imports | historical compatible release | Confirmed dependency; version unknown | Used broadly, sometimes unnecessarily. |
| SciPy | `scipy.spatial.distance` | historical compatible release | Confirmed dependency; version unknown | Used for Manhattan distances; can later be replaced with vectorized PyTorch. |
| Pillow | `PIL.Image` | historical compatible release | Confirmed dependency; version unknown | Image loading and visualization. |
| requests | direct imports | historical compatible release | Confirmed dependency; version unknown | Mainly notebook and visualization code. |
| matplotlib | direct imports | historical compatible release | Confirmed dependency; version unknown | Visualization. |
| imageio | visualization import | historical compatible release | Confirmed dependency; version unknown | Visualization only. |
| ipywidgets | visualization import | historical compatible release | Confirmed dependency; version unknown | Notebook visualization only. |
| IPython / Jupyter | notebook and display APIs | notebook format 4; Python 3.8.2 kernel | Confirmed | Exact Jupyter versions unknown. |
| pandas | fine-tuning script calls `pd.read_csv` through the FastAI wildcard namespace | historical compatible release | Strongly indicated | Not imported explicitly; reliance on wildcard namespace is fragile. |
| CUDA | explicit CUDA device selection, seeding and fp16 | NVIDIA CUDA environment | Confirmed | Exact toolkit and driver unknown. |
| NCCL | `init_process_group(backend='nccl')` | Linux multi-GPU environment | Confirmed | Several scripts fail without distributed environment variables. |
| Git LFS | `.pth` pointers | required for historical weights | Confirmed | Availability of underlying objects is unknown. |

## Execution assumptions

### Operating system and hardware

The launch scripts assume:

- Linux or another NCCL-capable environment;
- NVIDIA GPUs;
- a `local_rank` supplied by distributed launch;
- CUDA available before model and data setup;
- local file paths rooted at `~/Luiz/`;
- enough memory for batch sizes ranging from 20 to 90.

The papers describe experiments on four NVIDIA GeForce GTX 2080 Ti GPUs, but the repository does not record CUDA, cuDNN, driver, or container versions.

### Distributed training

Distance and region single-layer scripts use the historical PyTorch distributed launcher, set the CUDA device from `local_rank`, and initialize NCCL. Other scripts use FastAI `distrib_ctx()` or `.to_distributed()`. The exact combination of launcher and FastAI distributed APIs must be reconstructed.

### Checkpoint formats

The repository mixes:

- `Learner.export(...)` outputs, commonly named `.pkl`;
- FastAI `load_learner(...)`;
- `.pth` files loaded through `load_learner` and then passed to `load_state_dict`;
- Git LFS checkpoint pointers.

The serialization semantics are unclear and may depend on the historical FastAI release and import paths.

## Future dependency groups

### `arvit`

Target core dependencies:

- Python;
- PyTorch.

Optional: torchvision for preprocessing and examples.

The canonical architecture should not require FastAI, SciPy, Jupyter, or distributed initialization for import or a forward pass.

### Distance regularization

Target core dependencies:

- `arvit`;
- PyTorch.

SciPy should become unnecessary after a vectorized implementation, while remaining documented in the historical environment.

### Region-similarity regularization

Target core dependencies:

- `arvit`;
- PyTorch.

Optional: torchvision for preprocessing.

### SAM

Target core dependencies:

- `arvit`;
- PyTorch.

The maintained training framework is unresolved. It should avoid copied FastAI internals unless historical reproduction requires a separately isolated legacy environment.

## Historical-environment recovery recommendation

Phase 4 should test a compatibility grid rather than claim an unverified fixed environment:

1. Python 3.8 in an isolated container or remote runner;
2. FastAI v2 releases from the 2021–early 2022 period;
3. matching PyTorch and torchvision combinations;
4. CPU imports and synthetic forward pass first;
5. CUDA and distributed execution only after serialization and imports work.

No dependency file should claim exact reproducibility until an environment has actually executed the archived code.
