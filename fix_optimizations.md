# Training Optimizations Without CLI Control

The following training optimizations in the SBERT fine-tuning process cannot currently be turned off or modified via CLI arguments:

1. **Normalization in embedding comparisons**
   - Always applied
   - Affects similarity calculation in model evaluation

2. **Model card data**
   - Always set
   - Defines metadata for the model

3. **Evaluation approach**
   - Fixed using HuggingFace's binary classification evaluator
   - Determines how model performance is measured

## Implemented CLI Options

The following optimizations have been implemented and can be controlled via CLI arguments:

1. **Random seed setting** (`--random-seed`)
   - Default: 31337
   - Affects reproducibility of results

2. **Gradient checkpointing** (`--gradient-checkpointing/--no-gradient-checkpointing`)
   - Default: False
   - Reduces memory usage at the cost of increased computation time

3. **Quantization for non-FP16 mode** (`--quantization/--no-quantization`)
   - Default: False
   - Reduces model precision to save memory

4. **Early stopping** (`--patience`)
   - Default: 2
   - Number of evaluation steps without improvement before stopping

5. **Weight decay** (`--weight-decay`)
   - Default: 0.01
   - Controls L2 regularization strength

6. **Warmup ratio** (`--warmup-ratio`)
   - Default: 0.1
   - Affects learning rate schedule

7. **Save/eval strategy** (`--save-strategy` and `--eval-strategy`)
   - Default: "steps"
   - Options: "steps", "epoch", "no"
   - Controls when checkpoints are saved and evaluations are performed