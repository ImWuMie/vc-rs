//! Process-wide GPU scheduling priority.
//!
//! Maps the user-facing [`GpuPriority`](super::GpuPriority) onto the Windows
//! kernel graphics scheduler's *per-process* priority class via the `D3DKMT`
//! thunk. This is a different layer from the native-TensorRT CUDA *stream*
//! priority: stream priority only orders work between this process's own
//! streams, whereas this raises the whole process relative to other processes
//! competing for the GPU. Because it acts at the OS scheduler it applies to
//! every backend (Windows ML / DirectML and native TensorRT alike), and the two
//! mechanisms compose — they push the same direction at different granularities
//! and do not conflict.
//!
//! Best-effort hint only: it does not guarantee execution order, does not affect
//! host/device transfer scheduling, and failures are logged and ignored.

use super::GpuPriority;

/// Apply `priority` as this process's GPU scheduling priority class.
///
/// Process-wide and **not** auto-reverted, so only the standalone front-ends
/// call it. The VST3 plugin deliberately does not: it must not bump the whole
/// host DAW process (which would also outlive the plugin instance).
pub fn set_process_gpu_priority(priority: GpuPriority) {
    imp::set_process_gpu_priority(priority);
}

#[cfg(windows)]
mod imp {
    use super::GpuPriority;

    // d3dkmthk.h `D3DKMT_SCHEDULINGPRIORITYCLASS`. We deliberately map `High` to
    // HIGH (4) rather than REALTIME (5): REALTIME requires an elevated privilege
    // and can starve the desktop compositor, while HIGH needs no special rights.
    const D3DKMT_SCHEDULINGPRIORITYCLASS_NORMAL: i32 = 2;
    const D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH: i32 = 4;

    // Stable system exports. `D3DKMTSetProcessSchedulingPriorityClass` takes a
    // process HANDLE and the scheduling-priority-class enum by value and returns
    // an NTSTATUS (negative is failure). `GetCurrentProcess` returns the current
    // process pseudo-handle, which the thunk accepts. Declared directly rather
    // than via `windows-sys` to avoid pulling a feature surface for one call.
    #[link(name = "gdi32")]
    extern "system" {
        fn D3DKMTSetProcessSchedulingPriorityClass(process: isize, priority_class: i32) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
    }

    pub(super) fn set_process_gpu_priority(priority: GpuPriority) {
        let class = match priority {
            GpuPriority::Normal => D3DKMT_SCHEDULINGPRIORITYCLASS_NORMAL,
            GpuPriority::High => D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH,
        };
        // SAFETY: both are stable exports from system DLLs; we pass our own
        // process pseudo-handle and a valid class value, and only read the
        // returned NTSTATUS.
        let status = unsafe { D3DKMTSetProcessSchedulingPriorityClass(GetCurrentProcess(), class) };
        if status < 0 {
            tracing::warn!(
                "failed to set process GPU scheduling priority to {priority:?} \
                 (NTSTATUS {status:#010x}); continuing at default priority"
            );
        } else {
            tracing::info!("process GPU scheduling priority set to {priority:?}");
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::GpuPriority;

    // Non-Windows targets (dev/CI on other platforms) have no D3DKMT scheduler;
    // the priority knob is a no-op there.
    pub(super) fn set_process_gpu_priority(_priority: GpuPriority) {}
}
