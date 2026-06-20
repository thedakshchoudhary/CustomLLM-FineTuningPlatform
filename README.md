#    CoustomLLM Fine Tuning Platform 

## State-of-the-art Parameter-Efficient Fine-Tuning Methods

PEFT (Parameter-Efficient Fine-Tuning) is a library for adapting large pretrained models to downstream tasks by training only a small subset of additional parameters instead of updating every parameter in the model.

This approach significantly reduces computational requirements, GPU memory usage, and storage costs while achieving performance comparable to full fine-tuning.

PEFT is designed to work with modern machine learning workflows and supports efficient training and inference for large language models and diffusion models.

---

## Features

* Parameter-efficient adaptation of large pretrained models
* Reduced GPU memory consumption during training
* Significantly smaller checkpoints compared to full fine-tuning
* Support for multiple PEFT methods, including:

  * LoRA (Low-Rank Adaptation)
  * Prompt Tuning
  * Prefix Tuning
  * IA³
  * Other adapter-based techniques
* Support for distributed training and inference
* Compatibility with quantized models for low-resource environments

---

## Installation

```bash
pip install peft
```

---

## Quick Start

### Training with LoRA

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
import torch

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

model_id = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

model.save_pretrained("qwen2.5-3b-lora")
```

---

### Loading a Trained Adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

model_id = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device
)

model = PeftModel.from_pretrained(
    model,
    "qwen2.5-3b-lora"
)

inputs = tokenizer(
    "Preheat the oven to 350 degrees and place the cookie dough",
    return_tensors="pt"
)

outputs = model.generate(
    **inputs.to(device),
    max_new_tokens=50
)

print(
    tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )
)
```

---

## Why PEFT?

### Reduced Compute Requirements

PEFT trains only a small fraction of model parameters, enabling efficient fine-tuning of large models that would otherwise exceed available hardware resources.

### Reduced Storage

PEFT adapters are typically only a few megabytes in size compared to gigabyte-scale full model checkpoints, making it practical to maintain multiple task-specific adaptations of the same base model.

### High Performance

Despite training only a small number of parameters, PEFT methods achieve performance that is often comparable to full fine-tuning.

### Quantization Support

PEFT can be combined with model quantization techniques to further reduce memory consumption and enable training and inference on consumer-grade hardware.

---

## Supported Integrations

### Transformers

Supports adding, loading, and switching between adapters on transformer-based models.

### Diffusers

Enables memory-efficient training and fine-tuning of diffusion models with significantly smaller checkpoints.

### Accelerate

Provides compatibility with distributed training and inference across various hardware configurations.

### TRL

Supports reinforcement learning workflows and preference optimization techniques for large language models.

---

## Model Support

PEFT supports a wide range of transformer and diffusion architectures. Models can be adapted using built-in PEFT methods or manually configured to enable parameter-efficient fine-tuning.

---

## Core Benefits

* Lower memory requirements
* Faster experimentation
* Smaller checkpoints
* Efficient multi-task adaptation
* Reduced training costs
* Scalable fine-tuning of large models
* Easy deployment of task-specific adapters
