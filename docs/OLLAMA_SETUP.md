# Using NepsisAI with Ollama

**Run Nepsis with local LLMs - no API keys, no costs, full privacy!**

## Why Ollama?

- **Free**: No API costs, unlimited usage
- **Private**: Your data never leaves your machine
- **Fast**: Local inference, no network latency
- **Flexible**: Easy to switch models (llama2, mistral, codellama, etc.)

## Quick Start

### 1. Install Ollama

Visit [ollama.com](https://ollama.com/) and download for your platform:

- **macOS**: `brew install ollama`
- **Linux**: `curl https://ollama.ai/install.sh | sh`
- **Windows**: Download from website

### 2. Start Ollama Server

```bash
ollama serve
```

Keep this running in a separate terminal.

### 3. Pull a Model

```bash
# Recommended for general use
ollama pull llama2

# Other popular models
ollama pull mistral        # Fast, high quality
ollama pull codellama      # Code generation
ollama pull llama3         # Latest Llama
ollama pull phi            # Small, efficient
```

### 4. Run Nepsis Example

```bash
python3 examples/06_ollama_local.py
```

## Usage Examples

### Basic Streaming

```python
from nepsis.stream import govern_completion_stream, Hypothesis

stream = govern_completion_stream(
    prompt="Plan a 3-day trip to Kyoto under $150/day, no flights",
    model="llama2",  # or mistral, codellama, etc.
    hypotheses=[
        Hypothesis("budget_ok", "Stay under budget", prior=0.5),
        Hypothesis("no_flights", "No flights used", prior=0.8),
    ],
    mode="monitor",
    base_url="http://localhost:11434"  # Ollama default
)

for event in stream:
    if event.type == "token":
        print(event.payload["text"], end="")
    elif event.type == "metric":
        ρ = event.payload["contradiction_density"]
        print(f"\n[ρ={ρ:.2f}]")
```

### With Dynamic Constraints

```python
stream = govern_completion_stream(
    prompt="Plan a 3-day trip to Kyoto under $150/day, no flights",
    model="mistral",
    hypotheses=[
        Hypothesis("budget_ok", "Stay under budget", prior=0.5),
        Hypothesis("no_flights", "No flights used", prior=0.8),
    ],
    dynamic_constraints=True,  # Auto-enable trip planner constraint maps
    mode="monitor"
)
```

### Custom Ollama URL

If running Ollama on a different port or remote server:

```python
stream = govern_completion_stream(
    prompt="Your prompt here",
    model="llama2",
    hypotheses=[...],
    base_url="http://192.168.1.100:11434"  # Custom URL
)
```

## Benchmarking

Run latency benchmarks to measure Nepsis overhead:

```bash
python3 benchmarks/streaming_latency_ollama.py
```

**Example output:**
```
======================================================================
Latency Benchmark Results (Ollama)
======================================================================

Metric                         Baseline             Nepsis
----------------------------------------------------------------------
Mean time (s)                  8.234                8.891
Std dev (s)                    0.456                0.512
Min time (s)                   7.654                8.123
Max time (s)                   9.012                9.678

Overhead                       -                              8.0%

✅ PASS: Overhead <15% (acceptable for audit use cases)
```

## Supported Models

Any model available in Ollama works with Nepsis:

| Model | Size | Use Case |
|-------|------|----------|
| **llama2** | 7B | General purpose, good balance |
| **llama3** | 8B | Latest Llama, improved quality |
| **mistral** | 7B | Fast, high quality responses |
| **codellama** | 7B-34B | Code generation, technical tasks |
| **phi** | 2.7B | Small, efficient, fast |
| **vicuna** | 7B-13B | Instruction following |
| **orca** | 3B-13B | Reasoning tasks |

See full list: `ollama list` or [ollama.com/library](https://ollama.com/library)

## Model Selection

Auto-detection based on model name:

```python
# These all use OllamaAdapter automatically
govern_completion_stream(prompt, model="llama2", ...)
govern_completion_stream(prompt, model="mistral", ...)
govern_completion_stream(prompt, model="codellama", ...)

# Explicit ollama/ prefix also works
govern_completion_stream(prompt, model="ollama/llama2", ...)
```

## Troubleshooting

### Ollama Server Not Running

```
❌ Error: Ollama server not available at http://localhost:11434
```

**Fix:** Start Ollama in a separate terminal:
```bash
ollama serve
```

### Model Not Found

```
❌ Error: model 'llama2' not found
```

**Fix:** Pull the model first:
```bash
ollama pull llama2
```

### Connection Refused

**Check if Ollama is running:**
```bash
curl http://localhost:11434/api/tags
```

Should return JSON with available models.

### Slow Responses

**Tips:**
- Use smaller models (phi, llama2-7b vs llama2-70b)
- Ensure sufficient RAM (8GB+ recommended)
- Close other applications
- Consider GPU acceleration if available

## Performance Tips

### 1. Choose Right Model Size

- **Fast (<2s/token)**: phi, llama2-7b, mistral-7b
- **Balanced**: llama3-8b, codellama-13b
- **Quality**: llama2-70b, codellama-34b (requires 64GB+ RAM)

### 2. Adjust Buffer Size

Smaller buffers = more frequent metrics, higher overhead:

```python
stream = govern_completion_stream(
    prompt=prompt,
    model="llama2",
    max_tokens_buffer=200,  # Larger buffer = less overhead
    emit_every_k_tokens=10,  # Less frequent metrics
    ...
)
```

### 3. Monitor Mode Only

Monitor mode has lowest overhead (no intervention logic):

```python
stream = govern_completion_stream(
    prompt=prompt,
    model="llama2",
    mode="monitor",  # Lowest overhead
    ...
)
```

## API Compatibility

Ollama uses the same `govern_completion_stream()` API as OpenAI/Anthropic:

| Provider | Model Example | API Key Required? |
|----------|---------------|-------------------|
| OpenAI | `gpt-4o-mini` | Yes (env var) |
| Anthropic | `claude-3-sonnet` | Yes (env var) |
| **Ollama** | `llama2` | **No** |

Just change the `model` parameter - everything else is identical!

## Advanced: Remote Ollama

Run Ollama on a powerful server, access from laptop:

**Server (GPU machine):**
```bash
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

**Client (laptop):**
```python
stream = govern_completion_stream(
    prompt=prompt,
    model="llama2",
    base_url="http://gpu-server.local:11434",
    ...
)
```

## Next Steps

1. **Try different models**: Compare llama2, mistral, codellama
2. **Run benchmarks**: Measure overhead on your hardware
3. **Build constraint maps**: Customize for your domain
4. **Iterate on prompts**: See how ρ changes with different phrasing

---

**Questions?** Check [main README](../README.md) or [examples/](../examples/)
