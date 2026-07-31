//! ONNX model wrapper for DPDFNet (CUDA-first, with a CPU fallback).
//!
//! Loads a DPDFNet graph (file or bytes), seeds the recurrent state from the
//! model's custom metadata (`erb_norm_init` + `spec_norm_init`), and runs one hop:
//!   inputs : `spec` `[1,1,481,2]` (f32 interleaved re/im) + `state_in` `[state_size]`
//!   outputs: `spec_e` `[1,1,481,2]` + `state_out` `[state_size]`

use crate::stft::{FREQ_BINS, SPEC_LEN};
use ort::ep::cuda::ConvAlgorithmSearch;
use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
use ort::session::{builder::GraphOptimizationLevel, IoBinding, RunOptions, Session};
use ort::value::{Tensor, TensorRef};
use ort::{ortsys, AsPointer};
use std::ffi::{c_void, CStr};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

pub struct Model {
    // `Session` must be dropped before the custom stream used by its CUDA EP.
    session: Session,
    execution: Execution,
    _cuda_stream: Option<CudaStream>,
    pub state_size: usize,
    pub init_state: Vec<f32>,
}

enum Execution {
    Standard,
    Cuda(Box<CudaIo>),
}

type CudaError = i32;
type CudaStreamRaw = *mut c_void;

type CudaSetDevice = unsafe extern "C" fn(i32) -> CudaError;
type CudaDeviceGetStreamPriorityRange = unsafe extern "C" fn(*mut i32, *mut i32) -> CudaError;
type CudaStreamCreateWithPriority = unsafe extern "C" fn(*mut CudaStreamRaw, u32, i32) -> CudaError;
type CudaStreamDestroy = unsafe extern "C" fn(CudaStreamRaw) -> CudaError;
type CudaStreamSynchronize = unsafe extern "C" fn(CudaStreamRaw) -> CudaError;
type CudaMemcpyAsync =
    unsafe extern "C" fn(*mut c_void, *const c_void, usize, i32, CudaStreamRaw) -> CudaError;
type CudaHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CudaError;
type CudaFreeHost = unsafe extern "C" fn(*mut c_void) -> CudaError;
type CudaGetErrorString = unsafe extern "C" fn(CudaError) -> *const std::ffi::c_char;

const CUDA_SUCCESS: CudaError = 0;
const CUDA_STREAM_NON_BLOCKING: u32 = 1;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

/// Dynamically loaded CUDA Runtime entry points. Loading at runtime preserves
/// the CPU fallback on systems where the CUDA toolkit is not installed.
struct CudaApi {
    set_device: CudaSetDevice,
    device_get_stream_priority_range: CudaDeviceGetStreamPriorityRange,
    stream_create_with_priority: CudaStreamCreateWithPriority,
    stream_destroy: CudaStreamDestroy,
    stream_synchronize: CudaStreamSynchronize,
    memcpy_async: CudaMemcpyAsync,
    host_alloc: CudaHostAlloc,
    free_host: CudaFreeHost,
    get_error_string: CudaGetErrorString,
    _library: libloading::Library,
}

impl CudaApi {
    fn load() -> Result<Arc<Self>, String> {
        let mut errors = Vec::new();
        for soname in ["libcudart.so", "libcudart.so.13", "libcudart.so.12"] {
            // SAFETY: the library is retained in `CudaApi` for at least as long
            // as every copied function pointer.
            let library = match unsafe { libloading::Library::new(soname) } {
                Ok(library) => library,
                Err(e) => {
                    errors.push(format!("{soname}: {e}"));
                    continue;
                }
            };
            // SAFETY: these names and signatures are from the CUDA Runtime C
            // API. Each symbol is copied while `library` remains owned below.
            unsafe {
                macro_rules! symbol {
                    ($name:literal, $ty:ty) => {
                        *library
                            .get::<$ty>(concat!($name, "\0").as_bytes())
                            .map_err(|e| format!("resolve {} from {soname}: {e}", $name))?
                    };
                }
                return Ok(Arc::new(Self {
                    set_device: symbol!("cudaSetDevice", CudaSetDevice),
                    device_get_stream_priority_range: symbol!(
                        "cudaDeviceGetStreamPriorityRange",
                        CudaDeviceGetStreamPriorityRange
                    ),
                    stream_create_with_priority: symbol!(
                        "cudaStreamCreateWithPriority",
                        CudaStreamCreateWithPriority
                    ),
                    stream_destroy: symbol!("cudaStreamDestroy", CudaStreamDestroy),
                    stream_synchronize: symbol!("cudaStreamSynchronize", CudaStreamSynchronize),
                    memcpy_async: symbol!("cudaMemcpyAsync", CudaMemcpyAsync),
                    host_alloc: symbol!("cudaHostAlloc", CudaHostAlloc),
                    free_host: symbol!("cudaFreeHost", CudaFreeHost),
                    get_error_string: symbol!("cudaGetErrorString", CudaGetErrorString),
                    _library: library,
                }));
            }
        }
        Err(format!(
            "could not load CUDA Runtime ({})",
            errors.join("; ")
        ))
    }

    fn check(&self, code: CudaError, operation: &str) -> Result<(), String> {
        if code == CUDA_SUCCESS {
            return Ok(());
        }
        // SAFETY: CUDA owns the returned static error string.
        let error = unsafe {
            let p = (self.get_error_string)(code);
            if p.is_null() {
                format!("CUDA error {code}")
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        Err(format!("{operation}: {error} ({code})"))
    }
}

/// A non-blocking, highest-priority CUDA stream owned for the entire ONNX
/// Runtime session lifetime.
struct CudaStream {
    api: Arc<CudaApi>,
    handle: usize,
}

impl CudaStream {
    fn new() -> Result<Self, String> {
        let api = CudaApi::load()?;
        let code = unsafe { (api.set_device)(0) };
        api.check(code, "select CUDA device 0")?;
        let mut least_priority = 0;
        let mut greatest_priority = 0;
        // CUDA calls the numerically smallest value the greatest priority.
        let code = unsafe {
            (api.device_get_stream_priority_range)(&mut least_priority, &mut greatest_priority)
        };
        api.check(code, "query CUDA stream priority range")?;
        let mut handle = ptr::null_mut();
        let code = unsafe {
            (api.stream_create_with_priority)(
                &mut handle,
                CUDA_STREAM_NON_BLOCKING,
                greatest_priority,
            )
        };
        api.check(code, "create high-priority CUDA stream")?;
        if handle.is_null() {
            return Err("CUDA returned a null stream".to_owned());
        }
        Ok(Self {
            api,
            handle: handle as usize,
        })
    }

    fn raw(&self) -> CudaStreamRaw {
        self.handle as CudaStreamRaw
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the stream, and Model drops its
        // ONNX session before reaching this field.
        unsafe {
            let _ = (self.api.set_device)(0);
            let _ = (self.api.stream_destroy)(self.raw());
        }
    }
}

/// Page-locked host staging memory makes the tiny per-hop transfers truly
/// asynchronous on the same stream as CUDA Graph replay.
struct PinnedBuffer {
    api: Arc<CudaApi>,
    address: usize,
    len: usize,
}

impl PinnedBuffer {
    fn new(api: Arc<CudaApi>, len: usize) -> Result<Self, String> {
        let bytes = len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("CUDA pinned buffer size overflow")?;
        let mut address = ptr::null_mut();
        let code = unsafe { (api.host_alloc)(&mut address, bytes, 0) };
        api.check(code, "allocate CUDA pinned host staging")?;
        if address.is_null() {
            return Err("CUDA returned a null pinned host buffer".to_owned());
        }
        Ok(Self {
            api,
            address: address as usize,
            len,
        })
    }

    fn as_ptr(&self) -> *const c_void {
        self.address as *const c_void
    }

    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.address as *mut c_void
    }

    fn copy_from_slice(&mut self, source: &[f32]) {
        assert_eq!(source.len(), self.len);
        // SAFETY: `address` is a live allocation for exactly `len` f32 values.
        unsafe {
            std::slice::from_raw_parts_mut(self.address as *mut f32, self.len)
                .copy_from_slice(source);
        }
    }

    fn copy_to_slice(&self, destination: &mut [f32]) {
        assert_eq!(destination.len(), self.len);
        // SAFETY: the stream is synchronized before this method is called.
        unsafe {
            destination.copy_from_slice(std::slice::from_raw_parts(
                self.address as *const f32,
                self.len,
            ));
        }
    }
}

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the cudaHostAlloc allocation.
        let _ = unsafe { (self.api.free_host)(self.address as *mut c_void) };
    }
}

/// Two fixed bindings alternate device-resident recurrent-state buffers:
///
/// - binding 0: state A -> state B
/// - binding 1: state B -> state A
///
/// `spec` and `spec_e` use fixed device tensors so CUDA Graph replay sees the
/// same addresses. Page-locked CPU buffers stage two same-stream 3.8 KiB
/// copies; the much larger DPDFNet8 recurrent state never leaves CUDA.
struct CudaIo {
    // Drop bindings before the values and allocator they reference.
    bindings: [IoBinding; 2],
    // Both bindings reference these stable device tensors.
    _spec_input_device: Tensor<f32>,
    _spec_output_device: Tensor<f32>,
    spec_input_address: usize,
    spec_output_address: usize,
    spec_input_host: PinnedBuffer,
    spec_output_host: PinnedBuffer,
    // Keep owned handles to the exact state OrtValues installed in the
    // bindings. `Value::view().try_upgrade()` shares the Arc without copying.
    state_buffers: [Tensor<f32>; 2],
    state_addresses: [usize; 2],
    // Distinct IDs are required because the two captured graphs use opposite
    // recurrent-state addresses.
    run_options: [RunOptions; 2],
    _state_allocator: Allocator,
    cuda: Arc<CudaApi>,
    stream: usize,
    device_thread: std::thread::ThreadId,
    active: usize,
}

impl CudaIo {
    fn new(session: &Session, init_state: &[f32], stream: &CudaStream) -> Result<Self, String> {
        let state_size = init_state.len();
        let state_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .map_err(|e| e.to_string())?;
        let state_allocator = Allocator::new(session, state_info)
            .map_err(|e| format!("create CUDA state allocator: {e}"))?;

        let state_a = Tensor::from_array(([state_size], init_state.to_vec().into_boxed_slice()))
            .map_err(|e| e.to_string())?
            .to(AllocationDevice::CUDA, 0)
            .map_err(|e| format!("upload recurrent state to CUDA: {e}"))?;
        // This buffer is written by binding 0 before binding 1 ever reads it.
        let state_b = Tensor::<f32>::new(&state_allocator, [state_size])
            .map_err(|e| format!("allocate second CUDA state buffer: {e}"))?;

        let spec_input_device = Tensor::<f32>::new(&state_allocator, [1usize, 1, FREQ_BINS, 2])
            .map_err(|e| e.to_string())?;
        let spec_output_device = Tensor::<f32>::new(&state_allocator, [1usize, 1, FREQ_BINS, 2])
            .map_err(|e| e.to_string())?;
        let spec_input_address = spec_input_device.data_ptr() as usize;
        let spec_output_address = spec_output_device.data_ptr() as usize;
        let state_addresses = [state_a.data_ptr() as usize, state_b.data_ptr() as usize];
        let spec_input_host = PinnedBuffer::new(Arc::clone(&stream.api), SPEC_LEN)?;
        let spec_output_host = PinnedBuffer::new(Arc::clone(&stream.api), SPEC_LEN)?;

        let mut bindings = [
            session.create_binding().map_err(|e| e.to_string())?,
            session.create_binding().map_err(|e| e.to_string())?,
        ];
        bindings[0]
            .bind_input("spec", &spec_input_device)
            .map_err(|e| e.to_string())?;
        bindings[1]
            .bind_input("spec", &spec_input_device)
            .map_err(|e| e.to_string())?;
        bindings[0]
            .bind_input("state_in", &state_a)
            .map_err(|e| e.to_string())?;
        bindings[1]
            .bind_input("state_in", &state_b)
            .map_err(|e| e.to_string())?;

        bindings[0]
            .bind_output(
                "spec_e",
                spec_output_device
                    .view()
                    .try_upgrade()
                    .map_err(|_| "could not share CUDA spectrum output".to_owned())?,
            )
            .map_err(|e| e.to_string())?;
        bindings[1]
            .bind_output(
                "spec_e",
                spec_output_device
                    .view()
                    .try_upgrade()
                    .map_err(|_| "could not share CUDA spectrum output".to_owned())?,
            )
            .map_err(|e| e.to_string())?;
        // Each output shares (without copying) the opposite binding's input
        // tensor. No individual run aliases its own state input and output.
        bindings[0]
            .bind_output(
                "state_out",
                state_b
                    .view()
                    .try_upgrade()
                    .map_err(|_| "could not share CUDA state B".to_owned())?,
            )
            .map_err(|e| e.to_string())?;
        bindings[1]
            .bind_output(
                "state_out",
                state_a
                    .view()
                    .try_upgrade()
                    .map_err(|_| "could not share CUDA state A".to_owned())?,
            )
            .map_err(|e| e.to_string())?;

        let mut run_a = RunOptions::new().map_err(|e| e.to_string())?;
        run_a
            .add_config_entry("gpu_graph_id", "1")
            .map_err(|e| e.to_string())?;
        run_a.disable_device_sync().map_err(|e| e.to_string())?;
        let mut run_b = RunOptions::new().map_err(|e| e.to_string())?;
        run_b
            .add_config_entry("gpu_graph_id", "2")
            .map_err(|e| e.to_string())?;
        run_b.disable_device_sync().map_err(|e| e.to_string())?;

        Ok(Self {
            bindings,
            _spec_input_device: spec_input_device,
            _spec_output_device: spec_output_device,
            spec_input_address,
            spec_output_address,
            spec_input_host,
            spec_output_host,
            state_buffers: [state_a, state_b],
            state_addresses,
            run_options: [run_a, run_b],
            _state_allocator: state_allocator,
            cuda: Arc::clone(&stream.api),
            stream: stream.handle,
            device_thread: std::thread::current().id(),
            active: 0,
        })
    }

    fn ensure_device_for_thread(&mut self) -> Result<(), String> {
        let thread = std::thread::current().id();
        if thread != self.device_thread {
            let code = unsafe { (self.cuda.set_device)(0) };
            self.cuda.check(code, "select CUDA device 0")?;
            self.device_thread = thread;
        }
        Ok(())
    }

    fn synchronize_best_effort(&self) {
        let _ = unsafe { (self.cuda.stream_synchronize)(self.stream as CudaStreamRaw) };
    }

    fn run(
        &mut self,
        session: &mut Session,
        spec: &[f32; SPEC_LEN],
        spec_e: &mut [f32; SPEC_LEN],
    ) -> Result<(), String> {
        self.ensure_device_for_thread()?;
        let current = self.active;
        let stream = self.stream as CudaStreamRaw;
        let bytes = SPEC_LEN * std::mem::size_of::<f32>();
        self.spec_input_host.copy_from_slice(spec);
        let code = unsafe {
            (self.cuda.memcpy_async)(
                self.spec_input_address as *mut c_void,
                self.spec_input_host.as_ptr(),
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        if let Err(e) = self.cuda.check(code, "upload spectrum to CUDA") {
            self.synchronize_best_effort();
            return Err(e);
        }

        // The safe ort wrapper fetches and wraps all bound outputs after every
        // run. They are pre-bound here, so invoke RunWithBinding directly and
        // keep the real-time path allocation-free.
        if let Err(e) =
            run_binding_raw(session, &self.bindings[current], &self.run_options[current])
        {
            self.synchronize_best_effort();
            return Err(format!("CUDA binding {current} run: {e}"));
        }
        let code = unsafe {
            (self.cuda.memcpy_async)(
                self.spec_output_host.as_mut_ptr(),
                self.spec_output_address as *const c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if let Err(e) = self
            .cuda
            .check(code, "download enhanced spectrum from CUDA")
        {
            self.synchronize_best_effort();
            return Err(e);
        }
        let code = unsafe { (self.cuda.stream_synchronize)(stream) };
        self.cuda.check(code, "wait for CUDA denoiser hop")?;
        self.spec_output_host.copy_to_slice(spec_e);
        self.active = 1 - current;
        Ok(())
    }

    fn reset(&mut self, init_state: &[f32]) -> Result<(), String> {
        self.ensure_device_for_thread()?;
        if init_state.len() != self.state_buffers[0].shape().num_elements() {
            return Err("CUDA recurrent-state reset size mismatch".to_owned());
        }
        let stream = self.stream as CudaStreamRaw;
        let code = unsafe {
            (self.cuda.memcpy_async)(
                self.state_addresses[0] as *mut c_void,
                init_state.as_ptr().cast(),
                std::mem::size_of_val(init_state),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        };
        if let Err(e) = self.cuda.check(code, "reset CUDA recurrent state") {
            self.synchronize_best_effort();
            return Err(e);
        }
        let code = unsafe { (self.cuda.stream_synchronize)(stream) };
        self.cuda
            .check(code, "wait for CUDA recurrent-state reset")?;
        self.active = 0;
        Ok(())
    }
}

fn run_binding_raw(
    session: &mut Session,
    binding: &IoBinding,
    options: &RunOptions,
) -> ort::Result<()> {
    ortsys![unsafe RunWithBinding(session.ptr_mut(), options.ptr(), binding.ptr())?];
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Cuda,
    Auto,
    Cpu,
}

impl Backend {
    fn from_env() -> Result<Self, String> {
        match std::env::var("HUSHMIC_ONNX_EP")
            .unwrap_or_else(|_| "auto".to_owned())
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "cuda" | "gpu" => Ok(Self::Cuda),
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            value => Err(format!(
                "invalid HUSHMIC_ONNX_EP={value:?}; expected cuda, auto, or cpu"
            )),
        }
    }
}

fn parse_csv_f32(s: &str) -> Vec<f32> {
    s.split(',')
        .filter_map(|t| t.trim().parse::<f32>().ok())
        .collect()
}

/// CUDA-first session options; callers must have a
/// committed runtime environment before touching `Session::builder()` (ort's
/// API bootstrap would otherwise retry the dylib load itself and PANIC on
/// failure) — `crate::runtime::ensure_runtime` guarantees that.
///
/// `HUSHMIC_ONNX_EP=auto` (the default) tries CUDA and falls back to CPU.
/// `cuda` makes CUDA registration and device I/O mandatory, while `cpu` is an
/// explicit recovery path.
fn session_builder() -> Result<
    (
        ort::session::builder::SessionBuilder,
        Backend,
        Option<CudaStream>,
    ),
    String,
> {
    let backend = Backend::from_env()?;
    let cuda_graph = !matches!(
        std::env::var("HUSHMIC_CUDA_GRAPH")
            .unwrap_or_else(|_| "1".to_owned())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "0" | "false" | "off" | "no"
    );
    let cuda_stream = match backend {
        Backend::Cpu => None,
        Backend::Cuda => Some(
            CudaStream::new()
                .map_err(|e| format!("CUDA execution requested but unavailable: {e}"))?,
        ),
        Backend::Auto => match CudaStream::new() {
            Ok(stream) => Some(stream),
            Err(e) => {
                eprintln!("hushmic-denoiser: CUDA Runtime unavailable ({e}); using CPU");
                None
            }
        },
    };
    let cuda = cuda_stream.as_ref().map(|stream| {
        let cuda = ort::ep::CUDA::default()
            .with_device_id(0)
            // Avoid an expensive exhaustive cuDNN autotune during construction.
            .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
            .with_cuda_graph(cuda_graph);
        // SAFETY: Model retains `cuda_stream` until after Session is dropped.
        unsafe { cuda.with_compute_stream(stream.raw().cast::<()>()) }.build()
    });
    let execution_providers = match (backend, cuda) {
        (Backend::Cuda, Some(cuda)) => {
            vec![cuda.error_on_failure(), ort::ep::CPU::default().build()]
        }
        (Backend::Auto, Some(cuda)) => {
            vec![cuda.fail_silently(), ort::ep::CPU::default().build()]
        }
        (Backend::Cpu | Backend::Auto, None) => vec![ort::ep::CPU::default().build()],
        (Backend::Cuda, None) => unreachable!("forced CUDA always has a stream"),
        (Backend::Cpu, Some(_)) => unreachable!("explicit CPU never creates a CUDA stream"),
    };

    let builder = Session::builder()
        .map_err(|e| e.to_string())?
        .with_execution_providers(execution_providers)
        .map_err(|e| e.to_string())?
        .with_intra_threads(1)
        .map_err(|e| e.to_string())?
        .with_inter_threads(1)
        .map_err(|e| e.to_string())?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| e.to_string())?;
    Ok((builder, backend, cuda_stream))
}

impl Model {
    pub fn load(model_path: &Path) -> Result<Model, String> {
        let (mut builder, backend, cuda_stream) = session_builder()?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| format!("commit_from_file({}): {e}", model_path.display()))?;
        Model::from_session(session, backend, cuda_stream)
    }

    pub fn from_memory(model_bytes: &[u8]) -> Result<Model, String> {
        let (mut builder, backend, cuda_stream) = session_builder()?;
        let session = builder
            .commit_from_memory(model_bytes)
            .map_err(|e| format!("commit_from_memory: {e}"))?;
        Model::from_session(session, backend, cuda_stream)
    }

    fn from_session(
        session: Session,
        backend: Backend,
        cuda_stream: Option<CudaStream>,
    ) -> Result<Model, String> {
        let meta = session.metadata().map_err(|e| e.to_string())?;

        // state_size: the model exports it as authoritative custom metadata. We prefer this over
        // introspecting `session.inputs()[1]`'s declared shape -- it is equally authoritative and
        // avoids any ambiguity from a symbolic/dynamic declared dimension.
        let state_size: usize = meta
            .custom("state_size")
            .and_then(|s| s.trim().parse().ok())
            .or_else(|| {
                // Fallback: read the rank-1 size from the `state_in` input's declared shape.
                session
                    .inputs()
                    .get(1)
                    .and_then(|outlet| outlet.dtype().tensor_shape())
                    .and_then(|shape| shape.last().copied())
                    .filter(|&d| d > 0)
                    .map(|d| d as usize)
            })
            .ok_or("could not determine state_size from metadata or input shape")?;

        // Seed init_state from custom metadata (erb_norm_init then spec_norm_init).
        let erb_sz: usize = meta
            .custom("erb_norm_state_size")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(481);
        let spec_sz: usize = meta
            .custom("spec_norm_state_size")
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(96);
        let erb_init = meta
            .custom("erb_norm_init")
            .map(|s| parse_csv_f32(&s))
            .unwrap_or_default();
        let spec_init = meta
            .custom("spec_norm_init")
            .map(|s| parse_csv_f32(&s))
            .unwrap_or_default();
        // `ModelMetadata` borrows `session` and has a Drop impl; release it before moving `session`.
        drop(meta);

        let mut init_state = vec![0f32; state_size];
        if erb_init.len() == erb_sz && erb_sz <= state_size {
            init_state[0..erb_sz].copy_from_slice(&erb_init);
        }
        if spec_init.len() == spec_sz && erb_sz + spec_sz <= state_size {
            init_state[erb_sz..erb_sz + spec_sz].copy_from_slice(&spec_init);
        }

        let execution = match backend {
            Backend::Cpu => Execution::Standard,
            Backend::Cuda => Execution::Cuda(Box::new(
                CudaIo::new(
                    &session,
                    &init_state,
                    cuda_stream.as_ref().ok_or("CUDA stream is missing")?,
                )
                .map_err(|e| format!("CUDA execution requested but unavailable: {e}"))?,
            )),
            Backend::Auto => {
                if let Some(stream) = cuda_stream.as_ref() {
                    match CudaIo::new(&session, &init_state, stream) {
                        Ok(io) => Execution::Cuda(Box::new(io)),
                        Err(e) => {
                            eprintln!(
                                "hushmic-denoiser: CUDA I/O unavailable ({e}); using host I/O"
                            );
                            Execution::Standard
                        }
                    }
                } else {
                    Execution::Standard
                }
            }
        };
        let actual_backend = match (&execution, cuda_stream.is_some()) {
            (Execution::Cuda(_), _) => "CUDA",
            (Execution::Standard, false) => "CPU",
            (Execution::Standard, true) => "host I/O fallback",
        };
        let mut model = Model {
            session,
            execution,
            _cuda_stream: cuda_stream,
            state_size,
            init_state,
        };

        // CUDA/cuDNN initialize kernels and arenas lazily. Pay that cost while
        // constructing the filter, then restore the recurrent seed before the
        // first live audio hop.
        if matches!(&model.execution, Execution::Cuda(_)) {
            model.warm_up()?;
            model.reset_execution_state()?;
        }
        eprintln!("hushmic-denoiser: ONNX backend {actual_backend} ready");
        Ok(model)
    }

    fn warm_up(&mut self) -> Result<(), String> {
        let spec = [0f32; SPEC_LEN];
        let mut spec_e = [0f32; SPEC_LEN];
        let mut state = self.init_state.clone();
        let mut state_out = vec![0f32; self.state_size];
        for _ in 0..3 {
            self.run(&spec, &state, &mut spec_e, &mut state_out)?;
            std::mem::swap(&mut state, &mut state_out);
        }
        Ok(())
    }

    pub(crate) fn reset_execution_state(&mut self) -> Result<(), String> {
        if let Execution::Cuda(io) = &mut self.execution {
            io.reset(&self.init_state)?;
        }
        Ok(())
    }

    pub fn run(
        &mut self,
        spec: &[f32; SPEC_LEN],
        state_in: &[f32],
        spec_e: &mut [f32; SPEC_LEN],
        state_out: &mut Vec<f32>,
    ) -> Result<(), String> {
        if let Execution::Cuda(io) = &mut self.execution {
            return io.run(&mut self.session, spec, spec_e);
        }

        let spec_t = TensorRef::from_array_view(([1usize, 1, FREQ_BINS, 2], spec.as_slice()))
            .map_err(|e| e.to_string())?;
        let state_t =
            TensorRef::from_array_view(([state_in.len()], state_in)).map_err(|e| e.to_string())?;
        let outputs = self
            .session
            .run(ort::inputs! { "spec" => spec_t, "state_in" => state_t })
            .map_err(|e| e.to_string())?;

        // `outputs[..]` (Index) PANICS on a missing name; use `get` so a model
        // with unexpected outputs degrades through the Err-to-silence path.
        let (_, e_slice) = outputs
            .get("spec_e")
            .ok_or("model has no output named 'spec_e'")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        if e_slice.len() != spec_e.len() {
            return Err(format!(
                "model output 'spec_e' has {} elements, expected {}",
                e_slice.len(),
                spec_e.len()
            ));
        }
        spec_e.copy_from_slice(e_slice);
        let (_, s_slice) = outputs
            .get("state_out")
            .ok_or("model has no output named 'state_out'")?
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        state_out.clear();
        state_out.extend_from_slice(s_slice);
        Ok(())
    }
}

#[cfg(all(test, feature = "load-dynamic"))]
mod tests {
    use super::*;
    use crate::stft::SPEC_LEN;
    use std::path::PathBuf;

    /// Development assets provisioned by the repo's asset setup; self-skip
    /// when absent so the suite runs on bare checkouts.
    fn dev_asset(rel: &str) -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(rel);
        if !p.exists() && std::env::var("HUSHMIC_ASSERT_ASSETS").as_deref() == Ok("1") {
            panic!("{} missing but HUSHMIC_ASSERT_ASSETS=1", p.display());
        }
        p.exists().then_some(p)
    }

    #[test]
    fn loads_and_runs_one_frame() {
        let (Some(mp), Some(rt)) = (
            dev_asset("assets/models/dpdfnet8_48khz_hr.onnx"),
            dev_asset("assets/lib/libonnxruntime.so"),
        ) else {
            eprintln!("skipping loads_and_runs_one_frame: assets not provisioned");
            return;
        };
        // AlreadyInitialized is fine — some other test may have won the commit.
        crate::runtime::init_runtime(rt).expect("runtime");
        let mut m = Model::load(&mp).expect("load model");
        // dpdfnet8 state size
        assert_eq!(m.state_size, 90228, "unexpected state size");
        // init_state has exactly 577 nonzero leading elements
        let nonzero = m.init_state.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(
            nonzero, 577,
            "expected 577 metadata-seeded nonzero state elems"
        );

        let spec = [0f32; SPEC_LEN]; // zero (silence) frame is a valid input
        let mut spec_e = [0f32; SPEC_LEN];
        let mut state_out = vec![0f32; m.state_size];
        let state_in = m.init_state.clone();
        m.run(&spec, &state_in, &mut spec_e, &mut state_out)
            .expect("run");
        // running must mutate state (recurrent step happened)
        assert!(state_out != state_in, "state_out did not change after run");
        assert_eq!(state_out.len(), m.state_size);
    }
}
