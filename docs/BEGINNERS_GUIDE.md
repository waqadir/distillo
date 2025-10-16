# Distillo Beginner's Guide

Welcome to Distillo! This guide will help you understand what Distillo is, why you'd use it, and how to get started.

## Table of Contents

1. [What is Distillo?](#what-is-distillo)
2. [Why Use Distillo?](#why-use-distillo)
3. [Key Concepts](#key-concepts)
4. [Getting Started](#getting-started)
5. [Your First Job](#your-first-job)
6. [Common Use Cases](#common-use-cases)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)
9. [Next Steps](#next-steps)

## What is Distillo?

Distillo is a framework for creating training data for AI models. Think of it as a factory that takes existing data and uses powerful AI models to create new, high-quality training examples.

### The Problem It Solves

Training AI models requires lots of high-quality examples. Creating these examples manually is:
- **Time-consuming**: Writing thousands of examples takes forever
- **Expensive**: Hiring experts to create examples costs money
- **Inconsistent**: Different people write examples differently

### The Distillo Solution

Distillo automates this process:
1. You provide some example data (prompts, questions, etc.)
2. A powerful "teacher" AI model generates high-quality responses
3. Distillo saves these examples in a format ready for training
4. You use this data to train your own AI models

## Why Use Distillo?

### 1. Save Money

Instead of calling OpenAI's API millions of times, you can:
- Generate a dataset once with GPT-5
- Train a smaller, cheaper model (like Llama or Qwen)
- Run that model yourself for pennies

**Example**:
- 1M GPT-5 API calls = ~$30,000
- Generate dataset once = $300
- Train your own 7B model = $50
- **Savings: 99%+**

### 2. Own Your AI

With Distillo, you can:
- Train models on your own hardware
- Keep your data private
- Customize models for your specific needs
- Not depend on external APIs

### 3. Create Specialized Models

Generate training data for:
- Domain-specific tasks (medical, legal, finance)
- Custom writing styles
- Specific formats (code, poetry, reports)
- Your company's knowledge base

### 4. Advanced Training (RL Mode)

Distillo also supports reinforcement learning to:
- Improve models iteratively
- Align models with human preferences (RLHF)
- Fine-tune models with expert supervision

## Key Concepts

### Off-Policy vs On-Policy

**Off-Policy (Simple Mode)**
```
1. Teacher model generates data
2. Save to file
3. Done! Use file to train later
```

**On-Policy (RL Mode)**
```
1. Student model tries to generate
2. Teacher evaluates and guides
3. Student improves in real-time
4. Repeat until student is good
```

Most beginners start with **off-policy mode**—it's simpler and faster.

### Teacher vs Student

- **Teacher**: The smart model generating training data (GPT-5, Claude, DeepSeek, GLM)
- **Student**: The smaller model you're training (Llama, Qwen)

### Backends

Distillo supports multiple AI providers:

| Backend | Type | Cost | Speed | Best For |
|---------|------|------|-------|----------|
| **vLLM** | Local | Free* | Fast | Running models on your GPU |
| **OpenAI** | API | $$$ | Fast | Using GPT-5 |
| **DeepSeek** | API | $ | Fast | Cost-effective Chinese models |
| **Z.AI (GLM)** | API | $$ | Fast | Chinese GLM models |
| **Anthropic** | API | $$$ | Fast | Using Claude |

*Hardware costs apply

## Getting Started

### Prerequisites

1. **Python 3.10+**
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Install Distillo**
   ```bash
   git clone https://github.com/waqadir/distillo.git
   cd distillo

   # For client only (submitting jobs)
   pip install -e ".[client]"

   # For server (running jobs locally)
   pip install -e ".[server]"
   ```

3. **API Keys** (if using OpenAI/Anthropic)
   ```bash
   export OPENAI_API_KEY="sk-..."
   # or
   export ANTHROPIC_API_KEY="sk-ant-..."
   ```

### Starting the Server

If you're running jobs locally:

```bash
# Start the Distillo server
uvicorn server.app:app --host 0.0.0.0 --port 9000

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:9000
```

Keep this terminal open. The server needs to run while you submit jobs.

## Your First Job

Let's create a simple dataset by distilling GPT-5's knowledge.

### Step 1: Create a Configuration File

Create `my-first-job.yaml`:

```yaml
# Server connection
server:
  base_url: "http://localhost:9000"  # Your local server
  api_key: null  # No auth needed for local

job:
  # Which AI model to use
  model:
    provider: "openai"
    name: "gpt-5"  # High-quality teacher model
    parameters:
      api_key: "your-openai-key-here"

  # Where to get input data
  source:
    dataset: "tatsu-lab/alpaca"  # Free dataset from HuggingFace
    split: "train"
    field: "instruction"  # Use the 'instruction' field

  # How to generate data
  generation:
    duplications: 1  # 1 response per prompt
    max_batch_size: 10  # Process 10 at a time
    parameters:
      temperature: 0.7  # Creativity level (0=boring, 1=creative)
      max_tokens: 512  # Maximum response length

  # Where to save results
  output:
    mode: "local_file"
    local_path: "./my_first_dataset.jsonl"
    format: "jsonl"
```

### Step 2: Submit the Job

**Option A: Using the CLI**
```bash
distillo submit my-first-job.yaml --output my_first_dataset.jsonl
```

**Option B: Using Python**
```python
from distillo.client import DistilloClient

with DistilloClient.from_config("my-first-job.yaml") as client:
    # Submit the job
    submission = client.submit_job()
    job_id = submission["job_id"]
    print(f"Job started: {job_id}")

    # Wait for completion
    status = client.poll_job(job_id, wait_for_completion=True)

    if status["status"] == "completed":
        # Download results
        client.download_result(job_id, destination="my_first_dataset.jsonl")
        print("Dataset created successfully!")
```

### Step 3: Check Your Results

Open `my_first_dataset.jsonl`:

```json
{"instruction": "What is the capital of France?", "generated_text": "The capital of France is Paris...", "tokens": 156}
{"instruction": "Explain photosynthesis", "generated_text": "Photosynthesis is the process...", "tokens": 243}
...
```

**Congratulations!** You've created your first training dataset! 🎉

## Common Use Cases

### Use Case 1: Create a Dataset from Scratch

**Scenario**: You have a list of questions, need answers.

```yaml
source:
  dataset: "your-username/my-questions"  # Your HuggingFace dataset
  field: "question"

model:
  provider: "openai"
  name: "gpt-5"

output:
  mode: "upload_hf"  # Upload back to HuggingFace
  hf:
    repo_id: "your-username/my-answers"
    token: "hf_..."
```

### Use Case 2: Augment Existing Data

**Scenario**: You have 100 examples, want 1000.

```yaml
generation:
  duplications: 10  # Create 10 variations of each example
  parameters:
    temperature: 0.9  # High creativity for variety
```

### Use Case 3: Local Model for Privacy

**Scenario**: Sensitive data, can't use APIs.

```yaml
model:
  provider: "vllm"
  name: "Qwen/Qwen3-VL-8B-Instruct"
  parameters:
    tensor_parallel_size: 1  # Use 1 GPU
    gpu_memory_utilization: 0.9
```

Requires: NVIDIA GPU with 16GB+ VRAM

### Use Case 4: Code Generation Dataset

**Scenario**: Create coding examples.

```yaml
source:
  dataset: "code-problems"
  field: "problem_description"

model:
  provider: "openai"
  name: "gpt-5"

generation:
  parameters:
    temperature: 0.2  # Low temp for code = more deterministic
    max_tokens: 1024
    stop: ["```"]  # Stop at code block end
```

## Understanding the Output

### JSONL Format

Each line is a separate JSON object:

```jsonl
{"prompt": "...", "generated_text": "...", "tokens": 123}
{"prompt": "...", "generated_text": "...", "tokens": 456}
```

**Why JSONL?**
- Easy to stream (process one line at a time)
- Works with big datasets (don't load all in memory)
- Standard format for ML training

### Using Your Dataset

**With HuggingFace Datasets**:
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="my_first_dataset.jsonl")
print(dataset["train"][0])
```

**With Pandas**:
```python
import pandas as pd

df = pd.read_json("my_first_dataset.jsonl", lines=True)
print(df.head())
```

**For Fine-tuning**:
```python
# Most fine-tuning libraries accept JSONL directly
# Example with transformers:
from datasets import load_dataset

dataset = load_dataset("json", data_files="my_first_dataset.jsonl")

# Ready to fine-tune!
trainer = Trainer(
    model=model,
    train_dataset=dataset["train"],
    ...
)
```

## Troubleshooting

### Problem: "Connection refused"

**Cause**: Server isn't running

**Solution**:
```bash
# Start the server in a separate terminal
uvicorn server.app:app --port 9000
```

### Problem: "Module not found"

**Cause**: Missing dependencies

**Solution**:
```bash
# Install all dependencies
pip install -e ".[all]"
```

### Problem: "CUDA out of memory"

**Cause**: Model too big for your GPU

**Solution**:
```yaml
# Reduce GPU memory usage
model:
  parameters:
    gpu_memory_utilization: 0.7  # Lower from 0.9
    max_model_len: 2048  # Reduce max sequence length
```

Or use a smaller model:
```yaml
model:
  name: "Qwen/Qwen3-VL-2B-Instruct"  # Much smaller
```

### Problem: "Rate limit exceeded"

**Cause**: Too many API requests to OpenAI/Anthropic

**Solution**:
```yaml
generation:
  max_batch_size: 5  # Reduce from 10
```

Or add delays between batches (requires code modification).

### Problem: Job stuck at "running"

**Check server logs**:
```bash
# In the terminal where server is running, you'll see errors
# Common issues:
# - API key invalid
# - Dataset doesn't exist
# - Network issues
```

**Check job status**:
```bash
distillo status <job-id>
```

## Next Steps

### Level Up Your Skills

1. **Try Different Models**
   - Experiment with different GPT-5 models
   - Try local models with vLLM
   - Compare quality vs cost

2. **Custom Processing**
   ```python
   def filter_quality(records):
       """Keep only high-quality responses"""
       return [r for r in records if len(r["generated_text"]) > 100]

   client = DistilloClient.from_config("config.yaml", processor=filter_quality)
   ```

3. **Batch Processing**
   - Process large datasets overnight
   - Use `--no-wait` flag to submit and check later
   ```bash
   distillo submit config.yaml --no-wait
   # Returns job_id immediately

   # Check later
   distillo status <job-id>
   ```

4. **Advanced: RL Training**
   - Try on-policy training mode
   - See `config/example-rl-training.yaml`
   - Requires GPU and RL dependencies: `pip install -e ".[rl]"`

### Resources

- **Examples**: Check `config/` folder for more configuration examples
- **API Docs**: http://localhost:9000/docs (when server is running)
- **GitHub Issues**: Report bugs or ask questions
- **Main README**: More technical details

### Join the Community

- **Share Your Datasets**: Upload to HuggingFace Hub
- **Contribute**: Add new backends, features
- **Help Others**: Answer questions on GitHub

## Quick Reference Card

### Installation
```bash
pip install -e ".[client]"  # Client only
pip install -e ".[server]"  # Server
pip install -e ".[all]"     # Everything
```

### Start Server
```bash
uvicorn server.app:app --port 9000
```

### Submit Job
```bash
distillo submit config.yaml --output result.jsonl
```

### Check Status
```bash
distillo status <job-id>
```

### Download Result
```bash
distillo download <job-id> --output result.jsonl
```

### Common Config Template
```yaml
server:
  base_url: "http://localhost:9000"

job:
  model:
    provider: "openai"
    name: "gpt-5"
    parameters:
      api_key: "sk-..."

  source:
    dataset: "your-dataset"
    field: "text"

  generation:
    duplications: 1
    parameters:
      temperature: 0.7
      max_tokens: 512

  output:
    mode: "local_file"
    local_path: "./output.jsonl"
```

---

**Happy distilling!** 🌊

If you get stuck, check the [main README](../README.md) or [open an issue](https://github.com/waqadir/distillo/issues).
