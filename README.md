# Distillo 🌊

> Turn expensive API models into affordable training data

Generate high-quality training datasets by distilling knowledge from GPT-5, Claude, DeepSeek, GLM, or local models. Train your own models with on-policy RL, PPO, and LoRA fine-tuning.

## Features

- **Dual Pipeline Support**: Off-policy distillation and on-policy RL training modes
- **Multiple Model Backends**: vLLM (local), OpenAI, DeepSeek, Z.AI (GLM), Anthropic, and extensible to more
- **Flexible Data Sources**: HuggingFace datasets with streaming support
- **Multiple Output Formats**: JSONL, JSON, Parquet
- **HuggingFace Integration**: Direct upload to HuggingFace Hub
- **Custom Processors**: Inject custom data filtering/processing logic
- **RL Training**: PPO and Importance Sampling algorithms with LoRA fine-tuning
- **KL Supervision**: Teacher-student training with KL divergence penalty
- **Checkpointing**: Save and resume training with automatic checkpoint management
- **Simple CLI**: Easy-to-use command-line interface
- **REST API**: Full-featured FastAPI server

## Installation

### Client Only

For submitting jobs to a remote server:

```bash
pip install -e ".[client]"
```

### Server Only

For running the Distillo server:

```bash
pip install -e ".[server]"
```

### RL Training (Optional)

For on-policy RL training features:

```bash
pip install -e ".[rl]"
```

### Full Installation

Install everything (client + server + RL + dev tools):

```bash
pip install -e ".[all]"
```

## Quick Start

> 📘 **New to Distillo?** Check out our [Beginner's Guide](docs/BEGINNERS_GUIDE.md) for a step-by-step tutorial!

### 1. Start the Server

```bash
# Start the server on port 9000
uvicorn server.app:app --host 0.0.0.0 --port 9000 --workers 1
```

### 2. Submit a Job

Using the Python client:

```python
from distillo.client import DistilloClient
from distillo.config import AppConfig

# Load configuration
config = AppConfig.load("config/example-vllm.yaml")

# Submit job and wait for completion
with DistilloClient(config=config) as client:
    submission = client.submit_job()
    job_id = submission["job_id"]

    # Poll for completion
    status = client.poll_job(job_id, interval=5.0, wait_for_completion=True)

    if status["status"] == "completed":
        client.download_result(job_id, destination="result.jsonl")
        print(f"✓ Result saved to result.jsonl")
```

Using the CLI:

```bash
# Submit job and wait for completion
distillo submit config/example-vllm.yaml --output result.jsonl

# Check job status
distillo status <job-id>

# Download result
distillo download <job-id> --output result.jsonl

# Cancel a job
distillo cancel <job-id>
```

## Configuration

Distillo uses YAML configuration files to define jobs. Here's a complete example:

```yaml
server:
  base_url: "http://localhost:9000"
  api_key: null
  verify_tls: true
  request_timeout: 30.0

job:
  # Model backend configuration
  model:
    provider: "vllm"  # or "openai", "deepseek", "zai", "anthropic"
    name: "Qwen/Qwen3-VL-8B-Instruct"
    parameters:
      tensor_parallel_size: 1
      gpu_memory_utilization: 0.9
      dtype: "auto"

  # Data source configuration
  source:
    dataset: "HuggingFaceH4/ultrachat_200k"
    split: "train_sft"
    field: "messages"
    streaming: false

  # Generation configuration
  generation:
    duplications: 1
    max_batch_size: 32
    seed: 42
    parameters:
      temperature: 0.7
      top_p: 0.9
      max_tokens: 2048
      stop: ["</s>"]
    on_policy: false  # Enable for on-policy training

  # Output configuration
  output:
    mode: "local_file"  # or "return", "upload_hf"
    local_path: "./output/result.jsonl"
    format: "jsonl"
```

See the `config/` directory for more examples.

## Model Backends

### vLLM (Local Inference)

Fast local inference using vLLM:

```yaml
model:
  provider: "vllm"
  name: "Qwen/Qwen3-VL-8B-Instruct"
  parameters:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    max_model_len: 8192
```

### OpenAI

Use OpenAI models:

```yaml
model:
  provider: "openai"
  name: "gpt-5"
  parameters:
    api_key: "your-key"  # or use OPENAI_API_KEY env var
```

### DeepSeek

Use DeepSeek models (cost-effective Chinese models):

```yaml
model:
  provider: "deepseek"
  name: "deepseek-chat"  # or "deepseek-reasoner"
  parameters:
    api_key: "your-key"  # or use DEEPSEEK_API_KEY env var
```

### Z.AI (GLM)

Use Z.AI GLM models (Chinese AI models):

```yaml
model:
  provider: "zai"  # or "glm"
  name: "glm-4.6"  # Options: glm-4.6, glm-4.5, glm-4.5-flash
  parameters:
    api_key: "your-key"  # or use ZAI_API_KEY env var
```

### Anthropic

Use Claude models:

```yaml
model:
  provider: "anthropic"
  name: "claude-sonnet-4-5-20250929"
  parameters:
    api_key: "your-key"  # or use ANTHROPIC_API_KEY env var
```

## Output Modes

### Local File

Save to a local file:

```yaml
output:
  mode: "local_file"
  local_path: "./output/result.jsonl"
  format: "jsonl"
```

### Return

Return results directly (in-memory):

```yaml
output:
  mode: "return"
  format: "jsonl"
```

### HuggingFace Upload

Upload directly to HuggingFace Hub:

```yaml
output:
  mode: "upload_hf"
  format: "jsonl"
  hf:
    repo_id: "your-username/dataset-name"
    token: "your-hf-token"
    private: true
```

## Custom Processors

You can inject custom processing logic to filter or transform records:

```python
def filter_quality(records):
    """Filter records based on custom criteria"""
    filtered = []
    for record in records:
        # Only keep records with completions longer than 100 chars
        if len(record.get("generated_text", "")) > 100:
            # Add custom fields
            record["quality_score"] = len(record["generated_text"]) / 1000
            filtered.append(record)
    return filtered

# Use the processor
config = AppConfig.load("config.yaml")
with DistilloClient(config=config, processor=filter_quality) as client:
    submission = client.submit_job()
    # ...
```

## On-Policy RL Training

Distillo supports on-policy reinforcement learning with teacher supervision and LoRA fine-tuning:

### Quick Start

```yaml
generation:
  on_policy: true
  on_policy_options:
    teacher: "gpt-5"
    api_key: "your-openai-api-key"
    learning_rate: 5.0e-5
    lora_rank: 16
    loss_fn: "ppo"  # or "importance_sampling"
    kl_penalty_coef: 0.1
    eval_every: 100
    save_every: 500
```

### Algorithms

**PPO (Proximal Policy Optimization)**
- More stable training
- Better for longer training runs
- Industry-standard for RLHF

**Importance Sampling**
- Simpler implementation
- Faster per-step training
- Good for quick experiments

### LoRA Configuration

LoRA enables efficient fine-tuning with minimal GPU memory:

```yaml
lora_rank: 16  # Options: 8 (fast), 16 (balanced), 32 (high quality)
```

- **Rank 8**: ~40% fewer parameters, faster training
- **Rank 16**: Balanced quality/speed (recommended)
- **Rank 32**: Higher quality, more GPU memory

### KL Divergence Penalty

Controls how closely student follows teacher:

```yaml
kl_penalty_coef: 0.1  # Range: 0.01 - 0.5
```

- **Low (0.01-0.05)**: More exploration, student diverges more
- **Medium (0.1-0.2)**: Balanced (recommended)
- **High (0.3-0.5)**: Student closely mimics teacher

### Training Features

- **Automatic Checkpointing**: Save model every N steps
- **Evaluation Metrics**: Track KL divergence and loss
- **Resume Training**: Continue from saved checkpoints
- **Merge Weights**: Export final model with merged LoRA weights

See `config/example-rl-training.yaml` for a complete example.

## API Reference

### Python Client

```python
from distillo.client import DistilloClient
from distillo.config import AppConfig

# Create client
client = DistilloClient.from_config("config.yaml")

# Submit job
submission = client.submit_job()
job_id = submission["job_id"]

# Get status
status = client.get_job_status(job_id)

# Poll until completion
status = client.poll_job(job_id, interval=5.0, wait_for_completion=True)

# Download result
client.download_result(job_id, destination="result.jsonl")

# Cancel job
client.cancel_job(job_id)
```

### CLI

```bash
# Submit a job
distillo submit config.yaml --output result.jsonl [--wait/--no-wait]

# Check status
distillo status <job-id> [--watch]

# Download result
distillo download <job-id> --output result.jsonl

# Cancel job
distillo cancel <job-id>
```

### REST API

#### Submit Job

```http
POST /v1/jobs
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY

{
  "model": { ... },
  "source": { ... },
  "generation": { ... },
  "output": { ... }
}
```

#### Get Job Status

```http
GET /v1/jobs/{job_id}
Authorization: Bearer YOUR_API_KEY
```

#### Download Result

```http
GET /v1/jobs/{job_id}/result
Authorization: Bearer YOUR_API_KEY
```

#### Cancel Job

```http
POST /v1/jobs/{job_id}/cancel
Authorization: Bearer YOUR_API_KEY
```

## Architecture

```
distillo/
├── distillo/                   # Client package
│   ├── config.py              # Configuration models
│   ├── client.py              # HTTP client
│   └── cli.py                 # CLI interface
│
├── server/                    # Server package
│   ├── app.py                # FastAPI application
│   ├── executor.py           # Job execution engine
│   ├── backends.py           # Model backend implementations
│   ├── sources.py            # Data source loaders
│   ├── writers.py            # Output writers
│   ├── uploader.py           # HuggingFace uploader
│   ├── models/               # Model utilities
│   │   └── lora_model.py    # LoRA wrapper
│   └── training/             # RL training
│       ├── kl_divergence.py # KL divergence calculation
│       ├── ppo.py           # PPO algorithm
│       ├── importance_sampling.py
│       └── on_policy_trainer.py
│
└── config/                   # Example configurations
    ├── example-vllm.yaml
    ├── example-openai.yaml
    ├── example-deepseek.yaml
    ├── example-zai.yaml
    ├── example-anthropic.yaml
    ├── example-on-policy.yaml
    └── example-rl-training.yaml
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/waqadir/distillo.git
cd distillo

# Install in development mode
pip install -e ".[all]"

# Run tests
pytest

# Format code
black .
ruff check .
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=distillo --cov=server --cov-report=html

# Run specific test
pytest tests/test_client.py -v
```


## Use Cases

- **Dataset Distillation**: Generate synthetic datasets from powerful teacher models
- **Data Augmentation**: Create variations of existing datasets
- **On-Policy RL**: Train models with online learning and KL supervision
- **Prompt Engineering**: Generate diverse completions for prompt evaluation
- **Model Evaluation**: Compare outputs across different models

## Contributing

Found a bug or have an idea? Feel free to [open an issue](https://github.com/waqadir/distillo/issues) or submit a PR.

## License

MIT License - see [LICENSE](LICENSE) file for details.
