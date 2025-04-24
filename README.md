# mb-gradient-descent

This is a simple mini-batch gradient descent algorithm written in Rust, designed as a test prototype to optimize a single weight parameter by minimizing a mean squared error loss function. It's not tied to any neural networks, just a standalone implementation for learning and experimentation.

The algorithm minimizes the loss function $J(\theta) = \frac{1}{2m} \sum_{i=1}^m (y_i - \theta \cdot x_i)^2$, where $\theta$ is a single weight, $x_i$ is the input, $y_i$ is the target, and $m$ is the mini-batch size. The included synthetic dataset follows $y = 2x + \text{noise}$, where $x$ is sampled from $[-1, 1]$ and noise is Gaussian (means 0, std 0.1). The goal is to converge to $\theta \approx 2$.

## Usage

### Requirements

You need `rustc` and `cargo` to be installed and that's it.

### Building

You can build (and run) this program with:
```bash
git clone https://github.com/CyntexMore/mb-gradient-descent.git
cd mb-gradient-descent/
cargo run --release
```

Example output:
```
Epoch 0, Batch [0-31], Loss: 0.1234, Theta: 0.1500
Epoch 0, Batch [32-63], Loss: 0.0987, Theta: 0.2800
...
Epoch 90, Batch [0-31], Loss: 0.0051, Theta: 1.9876
...
Final theta: 1.9900 (expected ~2.0)
```

The loss should decrease, and $\theta$ should approach $\sim 2$.

### Testing

Modify `generated_dataset` in `src/main.rs` to change the size or noise level. And/or adjust `learning_rate`, `batch_size`, and `num_epochs` in `src/main.rs`.

## License
Feel free to use this code in your projects, no credit needed! It's a simple prototype, but I hope it's useful.

Released under the MIT license.
