// Tell cargo where to find librefcuda.so + cudart.
fn main() {
    let engine_dir = std::env::var("CUDA_DOJO_ENGINE_DIR").unwrap_or_else(|_| {
        // Default: relative to this crate's manifest dir.
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
        format!("{}/../../cuda/engine", manifest)
    });
    println!("cargo:rustc-link-search=native={}", engine_dir);
    println!("cargo:rustc-link-lib=dylib=refcuda");

    let cuda_lib = std::env::var("CUDA_LIB_DIR")
        .unwrap_or_else(|_| "/usr/local/cuda-13.0/lib64".to_string());
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Tell cargo to also embed the engine_dir into the binary's runtime
    // search path so callers don't need LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", engine_dir);
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib);
}
