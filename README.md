# PyTorch From Scratch

Small, project-driven PyTorch exercises for learning tensors, autograd, and training loops.

The goal is not to finish a course. The goal is to build enough working knowledge to understand practical machine learning systems and contribute to related engineering work over time.

## Learning Path

1. **Tensors**
   Learn tensor creation, shapes, indexing, broadcasting, matrix multiplication, dtype, device, reshape, permute, squeeze, and unsqueeze.

2. **Autograd And Linear Regression**
   Fit `y = 3x + 2` using manual gradient descent. Understand `requires_grad`, `.backward()`, `.grad`, `.detach()`, and `torch.no_grad()`.

3. **Manual MLP**
   Build a small neural network using raw tensors before using `torch.nn`.

4. **MNIST With `torch.nn`**
   Train a small classifier with `nn.Module`, `DataLoader`, an optimizer, loss function, train loop, eval loop, and model checkpointing.

5. **Bridge To Autograd Internals**
   Reimplement micrograd and compare it with PyTorch autograd.

## Projects

| Folder | Focus |
| --- | --- |
| `01-tensors` | Tensor operations, shapes, broadcasting, devices |
| `02-autograd-linear-regression` | Autograd and manual optimization |
| `03-manual-mlp` | Neural network from raw tensors |
| `04-nn-module-mnist` | Standard PyTorch training workflow |
| `notes` | Short notes, questions, and debugging lessons |

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch Tensor Docs](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch Autograd Docs](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch `torch.nn` Docs](https://pytorch.org/docs/stable/nn.html)
- [PyTorch Data Loading Tutorial](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
- [PyTorch Optimization Tutorial](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [micrograd](https://github.com/karpathy/micrograd)
- [The spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0)
- [Dive into Deep Learning](https://d2l.ai/)
- [fast.ai Practical Deep Learning](https://course.fast.ai/)

## Working Style

For each exercise, write down:

```text
What confused me:
What I learned:
One thing I want to inspect deeper:
```

Useful debugging habits:

- Print tensor shapes often.
- Overfit one tiny batch before training the full dataset.
- Check whether loss decreases.
- Check whether gradients are non-zero.
- Check `.dtype` and `.device` when something breaks.
- Use `model.train()` and `model.eval()` intentionally.

## Setup

Create an environment however you prefer, then install PyTorch from the official selector:

- [PyTorch Get Started](https://pytorch.org/get-started/locally/)

Example with pip on macOS CPU/MPS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio jupyter matplotlib
```

Run notebooks:

```bash
jupyter notebook
```

## Long-Term Direction

After these exercises, continue with:

1. Reimplement `micrograd` from scratch.
2. Add small tensor-like operations to your implementation.
3. Compare your implementation with PyTorch autograd on small examples.
4. Later, study memory layout, strides, kernels, graph lowering, scheduling, and profiling.
