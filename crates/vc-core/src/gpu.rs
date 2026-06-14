//! CUDA device discovery for control-thread user interfaces.
//!
//! Enumeration initializes the CUDA driver and may load system DLLs, so callers
//! must keep it away from plugin scan, project restore, and audio callbacks.

#[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
use anyhow::bail;
use anyhow::Result;
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
use anyhow::{anyhow, Context};
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
use libloading::Library;
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
use std::ffi::{c_char, c_int, c_uint, CStr};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDevice {
    pub id: u32,
    pub display_name: String,
}

#[cfg(any(feature = "cuda", feature = "tensorrt"))]
type CuInit = unsafe extern "system" fn(c_uint) -> c_int;
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
type CuDeviceGetCount = unsafe extern "system" fn(*mut c_int) -> c_int;
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
type CuDeviceGetName = unsafe extern "system" fn(*mut c_char, c_int, c_int) -> c_int;

/// Lists CUDA devices using the CUDA driver ordinal namespace consumed by the
/// CUDA and native TensorRT backends.
#[cfg(any(feature = "cuda", feature = "tensorrt"))]
pub fn list_cuda_devices() -> Result<Vec<GpuDevice>> {
    // Keep the library alive until every loaded symbol call has completed.
    let library = unsafe { Library::new(cuda_driver_library_name()) }
        .with_context(|| format!("failed to load {}", cuda_driver_library_name()))?;
    unsafe {
        let init = library
            .get::<CuInit>(b"cuInit\0")
            .context("failed to load cuInit")?;
        let get_count = library
            .get::<CuDeviceGetCount>(b"cuDeviceGetCount\0")
            .context("failed to load cuDeviceGetCount")?;
        let get_name = library
            .get::<CuDeviceGetName>(b"cuDeviceGetName\0")
            .context("failed to load cuDeviceGetName")?;

        check_cuda(init(0), "cuInit")?;
        let mut count = 0;
        check_cuda(get_count(&mut count), "cuDeviceGetCount")?;
        let count = u32::try_from(count).context("CUDA returned a negative device count")?;
        let mut devices = Vec::with_capacity(count as usize);
        for id in 0..count {
            let ordinal = i32::try_from(id).context("CUDA device ID exceeds i32")?;
            let mut name = [0 as c_char; 256];
            check_cuda(
                get_name(name.as_mut_ptr(), name.len() as c_int, ordinal),
                "cuDeviceGetName",
            )?;
            devices.push(GpuDevice {
                id,
                display_name: CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned(),
            });
        }
        Ok(devices)
    }
}

#[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
pub fn list_cuda_devices() -> Result<Vec<GpuDevice>> {
    bail!("CUDA device enumeration is unavailable in this build")
}

#[cfg(any(feature = "cuda", feature = "tensorrt"))]
fn check_cuda(status: c_int, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(anyhow!("{operation} failed with CUDA status {status}"))
    }
}

#[cfg(all(windows, any(feature = "cuda", feature = "tensorrt")))]
fn cuda_driver_library_name() -> &'static str {
    "nvcuda.dll"
}

#[cfg(all(target_os = "linux", any(feature = "cuda", feature = "tensorrt")))]
fn cuda_driver_library_name() -> &'static str {
    "libcuda.so.1"
}

#[cfg(all(
    not(any(windows, target_os = "linux")),
    any(feature = "cuda", feature = "tensorrt")
))]
fn cuda_driver_library_name() -> &'static str {
    "libcuda"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_device_keeps_id_and_display_name() {
        let device = GpuDevice {
            id: 2,
            display_name: "NVIDIA Test GPU".to_string(),
        };
        assert_eq!(device.id, 2);
        assert_eq!(device.display_name, "NVIDIA Test GPU");
    }
}
