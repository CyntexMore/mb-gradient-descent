use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use rand_distr::{Distribution, Normal};

fn main() {
    // Hyperparameters
    // let learning_rate = 0.01;
    let learning_rate = 0.1; // 10x larger than the original
    let batch_size = 32;
    let num_epochs = 100;

    // Generate synthetic dataset: y = 2x + noise
    let dataset = generate_dataset(100);
    let n = dataset.len();

    // Initialize parameter (theta)
    let mut theta = 0.0;

    // Training loop
    for epoch in 0..num_epochs {
        // Shuffle dataset indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut thread_rng());

        // Process mini-batches
        for batch_start in (0..n).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n);
            let batch_indices = &indices[batch_start..batch_end];

            // Compute gradient for the mini-batch
            let mut gradient = 0.0;
            let mut batch_loss = 0.0;
            let m = batch_indices.len() as f64;

            for &i in batch_indices {
                let (x, y) = dataset[i];
                let prediction = theta * x;
                let error = prediction - y;
                gradient += error * x; // Partial derivative: (prediction - y) * x
                batch_loss += error * error;
            }

            // Average gradient and loss
            gradient /= m;
            batch_loss /= 2.0 * m;

            // Update parameter
            theta -= learning_rate * gradient;

            // Print loss for the first and every 10th epoch
            if epoch % 10 == 0 || epoch == 0 {
                println!(
                    "Epoch {}, Batch [{}-{}], Loss: {:.4}, Theta: {:.4}",
                    epoch, batch_start, batch_end - 1, batch_loss, theta
                );
            }
        }
    }

    println!("Final theta: {:.4} (expected ~2.0)", theta);
}

// Generate synthetic dataset: y = 2x + noise
fn generate_dataset(n: usize) -> Vec<(f64, f64)> {
    let mut rng = thread_rng();
    let noise = Normal::new(0.0, 0.1).unwrap(); // Mean 0, std 0.1
    let mut dataset = Vec::new();

    for _ in 0..n {
        let x = rng.gen_range(-1.0..1.0);
        let y = 2.0 * x + noise.sample(&mut rng);
        dataset.push((x, y));
    }

    dataset
}
