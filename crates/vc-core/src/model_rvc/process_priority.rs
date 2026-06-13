//! Process-wide GPU scheduling priority and CPU power-throttling policy.
//!
//! Two process-level Windows knobs used to keep inference performance stable
//! regardless of whether the app window has focus:
//!
//! - [`set_process_gpu_priority`] maps the user-facing
//!   [`GpuPriority`](super::GpuPriority) onto the kernel graphics scheduler's
//!   *per-process* priority class via the `D3DKMT` thunk. This is a different
//!   layer from the native-TensorRT CUDA *stream* priority (which only orders
//!   work between this process's own streams); it raises the whole process
//!   relative to other processes competing for the GPU, so it applies to every
//!   backend (Windows ML / DirectML and native TensorRT alike) and the two
//!   compose without conflict.
//! - [`set_process_power_throttling`] opts the process out of EcoQoS / Power
//!   Throttling so the OS does not move it to efficiency cores or a reduced
//!   clock when the window is not in the foreground. Because it is process-wide
//!   it covers *every* thread doing inference work — the realtime worker, the
//!   ONNX Runtime intra-op thread pool, and the native TensorRT CUDA
//!   orchestration/sync threads — which a per-thread override on the worker
//!   alone would miss.
//!
//! Both are best-effort hints: they do not guarantee execution order, do not
//! prioritize host/device transfers, and failures are logged and ignored.

use super::GpuPriority;

/// Apply `priority` as this process's GPU scheduling priority class.
///
/// Process-wide and **not** auto-reverted, so only the standalone front-ends
/// call it. The VST3 plugin deliberately does not: it must not bump the whole
/// host DAW process (which would also outlive the plugin instance).
pub fn set_process_gpu_priority(priority: GpuPriority) {
    imp::set_process_gpu_priority(priority);
}

/// Opt this process out of CPU power throttling (EcoQoS) when `disable` is true,
/// or restore the OS default when false.
///
/// Disabling keeps inference on performance cores at full clock even while the
/// window is in the background, removing the large foreground/background timing
/// difference. Process-wide and not auto-reverted, so — like
/// [`set_process_gpu_priority`] — only the standalone front-ends call it.
pub fn set_process_power_throttling(disable: bool) {
    imp::set_process_power_throttling(disable);
}

#[cfg(windows)]
mod imp {
    use super::GpuPriority;

    // d3dkmthk.h `D3DKMT_SCHEDULINGPRIORITYCLASS`. We deliberately map `High` to
    // HIGH (4) rather than REALTIME (5): REALTIME requires an elevated privilege
    // and can starve the desktop compositor, while HIGH needs no special rights.
    const D3DKMT_SCHEDULINGPRIORITYCLASS_NORMAL: i32 = 2;
    const D3DKMT_SCHEDULINGPRIORITYCLASS_HIGH: i32 = 4;

    // processthreadsapi.h: PROCESS_INFORMATION_CLASS::ProcessPowerThrottling and
    // the PROCESS_POWER_THROTTLING_STATE payload. Setting EXECUTION_SPEED in the
    // control mask with a zero state mask opts OUT of throttling (always high
    // performance); a zero control mask clears our override (OS default).
    const PROCESS_POWER_THROTTLING: i32 = 4;
    const PROCESS_POWER_THROTTLING_CURRENT_VERSION: u32 = 1;
    const PROCESS_POWER_THROTTLING_EXECUTION_SPEED: u32 = 0x1;

    #[repr(C)]
    struct ProcessPowerThrottlingState {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }

    // Stable system exports. `D3DKMTSetProcessSchedulingPriorityClass` takes a
    // process HANDLE and the scheduling-priority-class enum by value and returns
    // an NTSTATUS (negative is failure). `SetProcessInformation` returns a BOOL
    // (0 is failure). `GetCurrentProcess` returns the current process
    // pseudo-handle, which both thunks accept. Declared directly rather than via
    // `windows-sys` to avoid pulling a feature surface for a couple of calls.
    #[link(name = "gdi32")]
    extern "system" {
        fn D3DKMTSetProcessSchedulingPriorityClass(process: isize, priority_class: i32) -> i32;
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetProcessInformation(
            process: isize,
            information_class: i32,
            information: *const core::ffi::c_void,
            information_size: u32,
        ) -> i32;
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

    pub(super) fn set_process_power_throttling(disable: bool) {
        let control_mask = if disable {
            PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        } else {
            0
        };
        let state = ProcessPowerThrottlingState {
            version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            control_mask,
            // StateMask bit clear under an EXECUTION_SPEED control bit == "do not
            // throttle"; ignored when the control mask is zero (OS default).
            state_mask: 0,
        };
        // SAFETY: stable kernel32 export; we pass our own process pseudo-handle,
        // the matching information class, and a correctly sized POD struct that
        // we own for the duration of the call.
        let ok = unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                PROCESS_POWER_THROTTLING,
                core::ptr::addr_of!(state).cast(),
                core::mem::size_of::<ProcessPowerThrottlingState>() as u32,
            )
        };
        if ok == 0 {
            tracing::warn!(
                "failed to set process power throttling (disable={disable}); \
                 continuing at OS default"
            );
        } else {
            tracing::info!("process power throttling opt-out: disable={disable}");
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::GpuPriority;

    // Non-Windows targets (dev/CI on other platforms) have neither the D3DKMT
    // scheduler nor EcoQoS; both knobs are no-ops there.
    pub(super) fn set_process_gpu_priority(_priority: GpuPriority) {}
    pub(super) fn set_process_power_throttling(_disable: bool) {}
}
