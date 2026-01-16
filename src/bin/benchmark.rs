use std::env;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let target = if args.len() > 1 {
        &args[1]
    } else {
        "static/details.png"
    };
    let iterations = if args.len() > 2 {
        args[2].parse().unwrap_or(50)
    } else {
        50
    };

    println!("Building release binary...");
    let build_status = Command::new("cargo")
        .args(&["build", "--release"])
        .status()
        .expect("Failed to run cargo build");

    if !build_status.success() {
        eprintln!("Build failed!");
        return;
    }

    let binary_path = "./target/release/spritefusion-pixel-snapper";
    if !Path::new(binary_path).exists() {
        eprintln!("Binary not found at {}", binary_path);
        return;
    }

    println!(
        "Running benchmark on '{}' for {} iterations...",
        target, iterations
    );
    println!("---------------------------------------------------");

    let mut times = Vec::with_capacity(iterations);

    for i in 0..iterations {
        let start = Instant::now();
        let output = Command::new(binary_path)
            .args(&[target, "--output", "/dev/null"])
            .output()
            .expect("Failed to execute process");
        let duration = start.elapsed();

        if !output.status.success() {
            eprintln!(
                "Run {} failed: {}",
                i + 1,
                String::from_utf8_lossy(&output.stderr)
            );
            continue;
        }

        let secs = duration.as_secs_f64();
        times.push(secs);
        println!("Run {:02}: {:.4}s", i + 1, secs);
    }

    if times.is_empty() {
        println!("No successful runs.");
        return;
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = times.first().unwrap();
    let max = times.last().unwrap();
    let sum: f64 = times.iter().sum();
    let mean = sum / times.len() as f64;

    let median = if times.len() % 2 == 0 {
        let mid = times.len() / 2;
        (times[mid - 1] + times[mid]) / 2.0
    } else {
        times[times.len() / 2]
    };

    let variance: f64 = times
        .iter()
        .map(|value| {
            let diff = mean - *value;
            diff * diff
        })
        .sum::<f64>()
        / times.len() as f64;
    let std_dev = variance.sqrt();

    println!("\n--- Benchmark Results ({}) ---", target);
    println!("Total Runs: {}", times.len());
    println!("Min:        {:.4}s", min);
    println!("Max:        {:.4}s", max);
    println!("Mean:       {:.4}s", mean);
    println!("Median:     {:.4}s", median);
    println!("Std Dev:    {:.4}s", std_dev);
}
