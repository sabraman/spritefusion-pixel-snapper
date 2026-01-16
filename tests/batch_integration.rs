use std::process::Command;
use std::fs;
use std::path::Path;

#[test]
fn test_batch_processing_cli() {
    let input_dir = "tests/data";
    let output_dir = "tests/output";

    // Ensure output dir is clean
    if Path::new(output_dir).exists() {
        fs::remove_dir_all(output_dir).unwrap();
    }
    fs::create_dir_all(output_dir).unwrap();

    // Run the snapper in batch mode
    // We expect the binary to be built already or use cargo run
    let status = Command::new("cargo")
        .args([
            "run", 
            "--release", 
            "--", 
            input_dir, 
            "--output", 
            output_dir
        ])
        .status()
        .expect("Failed to execute snapper");

    assert!(status.success(), "Snapper exited with failure in batch mode");

    // Verify outputs
    let expected_files = vec![
        "character_fixed.png",
        "background_fixed.png",
        "sword_fixed.png",
    ];

    for file in expected_files {
        let path = Path::new(output_dir).join(file);
        assert!(path.exists(), "Output file {} missing", file);
        
        let metadata = fs::metadata(path).unwrap();
        assert!(metadata.len() > 0, "Output file {} is empty", file);
    }
}
