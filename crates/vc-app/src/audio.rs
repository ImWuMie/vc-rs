use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::AudioBackend;
use vc_core::dsp;

const CPAL_SCRATCH_FALLBACK_SAMPLES: usize = 65_536;
const CPAL_MAX_SCRATCH_SAMPLES: usize = 65_536;

#[cfg(windows)]
#[path = "wasapi_audio.rs"]
pub(crate) mod wasapi_audio;

pub fn print_cpal_devices() -> Result<()> {
    let host = cpal::default_host();

    println!("CPAL input devices:");
    for device in host.input_devices()? {
        println!("  {}", device_name(&device));
    }

    println!();
    println!("CPAL output devices:");
    for device in host.output_devices()? {
        println!("  {}", device_name(&device));
    }

    Ok(())
}

#[cfg(windows)]
pub fn print_wasapi_devices() -> Result<()> {
    wasapi_audio::print_devices()
}

#[cfg(not(windows))]
pub fn print_wasapi_devices() -> Result<()> {
    bail!("WASAPI audio backend is only available on Windows")
}

pub struct RealtimeAudio {
    backend: AudioBackend,
    wasapi_input_exclusive: bool,
    wasapi_output_exclusive: bool,
    input: InputEndpoint,
    output: OutputEndpoint,
    input_sample_rate: u32,
    output_sample_rate: u32,
    input_name: String,
    output_name: String,
}

enum InputEndpoint {
    Cpal {
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
    },
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStreamConfig),
}

enum OutputEndpoint {
    Cpal {
        device: cpal::Device,
        config: cpal::SupportedStreamConfig,
    },
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStreamConfig),
}

impl RealtimeAudio {
    pub fn open(
        backend: AudioBackend,
        wasapi_input_exclusive: bool,
        wasapi_output_exclusive: bool,
        input_name: Option<&str>,
        output_name: Option<&str>,
        wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        if (wasapi_input_exclusive || wasapi_output_exclusive) && backend != AudioBackend::Wasapi {
            bail!("--wasapi-exclusive* options require --audio-backend wasapi");
        }

        match backend {
            AudioBackend::Cpal => Self::open_cpal(input_name, output_name),
            AudioBackend::Wasapi => Self::open_wasapi(
                input_name,
                output_name,
                wasapi_input_exclusive,
                wasapi_output_exclusive,
                wasapi_buffer_ms,
            ),
        }
    }

    fn open_cpal(input_name: Option<&str>, output_name: Option<&str>) -> Result<Self> {
        let input_device = input_device(input_name)?;
        let input_config = default_input_config(&input_device)?;
        let output_device = output_device(output_name)?;
        let output_config = default_output_config(&output_device)?;
        let input_sample_rate = input_config.sample_rate();
        let output_sample_rate = output_config.sample_rate();
        let input_name = device_name(&input_device);
        let output_name = device_name(&output_device);

        Ok(Self {
            backend: AudioBackend::Cpal,
            wasapi_input_exclusive: false,
            wasapi_output_exclusive: false,
            input: InputEndpoint::Cpal {
                device: input_device,
                config: input_config,
            },
            output: OutputEndpoint::Cpal {
                device: output_device,
                config: output_config,
            },
            input_sample_rate,
            output_sample_rate,
            input_name,
            output_name,
        })
    }

    #[cfg(windows)]
    fn open_wasapi(
        input_name: Option<&str>,
        output_name: Option<&str>,
        wasapi_input_exclusive: bool,
        wasapi_output_exclusive: bool,
        wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        let endpoints = wasapi_audio::open_realtime(
            input_name,
            output_name,
            wasapi_input_exclusive,
            wasapi_output_exclusive,
            wasapi_buffer_ms,
        )?;
        let input_name = endpoints.input.device_name.clone();
        let output_name = endpoints.output.device_name.clone();
        let input_sample_rate = endpoints.input_sample_rate;
        let output_sample_rate = endpoints.output_sample_rate;

        Ok(Self {
            backend: AudioBackend::Wasapi,
            wasapi_input_exclusive,
            wasapi_output_exclusive,
            input: InputEndpoint::Wasapi(endpoints.input),
            output: OutputEndpoint::Wasapi(endpoints.output),
            input_sample_rate,
            output_sample_rate,
            input_name,
            output_name,
        })
    }

    #[cfg(not(windows))]
    fn open_wasapi(
        _input_name: Option<&str>,
        _output_name: Option<&str>,
        _wasapi_input_exclusive: bool,
        _wasapi_output_exclusive: bool,
        _wasapi_buffer_ms: u32,
    ) -> Result<Self> {
        bail!("WASAPI audio backend is only available on Windows")
    }

    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            AudioBackend::Cpal => "cpal",
            AudioBackend::Wasapi => {
                match (self.wasapi_input_exclusive, self.wasapi_output_exclusive) {
                    (true, true) => "wasapi-exclusive",
                    (true, false) => "wasapi-input-exclusive",
                    (false, true) => "wasapi-output-exclusive",
                    (false, false) => "wasapi-shared",
                }
            }
        }
    }

    pub fn input_sample_rate(&self) -> u32 {
        self.input_sample_rate
    }

    pub fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn build_input_stream<F>(&self, on_samples: F) -> Result<AudioStream>
    where
        F: FnMut(&[f32]) + Send + 'static,
    {
        match &self.input {
            InputEndpoint::Cpal { device, config } => Ok(AudioStream::Cpal(
                build_cpal_input_stream(device, config, on_samples)?,
            )),
            #[cfg(windows)]
            InputEndpoint::Wasapi(config) => Ok(AudioStream::Wasapi(
                wasapi_audio::build_input_stream(config.clone(), on_samples)?,
            )),
        }
    }

    pub fn build_output_stream<F>(&self, fill: F) -> Result<AudioStream>
    where
        F: FnMut(&mut [f32]) + Send + 'static,
    {
        match &self.output {
            OutputEndpoint::Cpal { device, config } => Ok(AudioStream::Cpal(
                build_cpal_output_stream(device, config, fill)?,
            )),
            #[cfg(windows)]
            OutputEndpoint::Wasapi(config) => Ok(AudioStream::Wasapi(
                wasapi_audio::build_output_stream(config.clone(), fill)?,
            )),
        }
    }
}

pub enum AudioStream {
    Cpal(cpal::Stream),
    #[cfg(windows)]
    Wasapi(wasapi_audio::WasapiStream),
}

impl AudioStream {
    pub fn play(&self) -> Result<()> {
        match self {
            AudioStream::Cpal(stream) => stream.play().context("failed to start CPAL stream"),
            #[cfg(windows)]
            AudioStream::Wasapi(stream) => stream.play(),
        }
    }
}

pub fn input_device(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    find_device(host.input_devices()?, name)
        .or_else(|| host.default_input_device())
        .ok_or_else(|| anyhow!("input device not found"))
}

pub fn output_device(name: Option<&str>) -> Result<cpal::Device> {
    let host = cpal::default_host();
    find_device(host.output_devices()?, name)
        .or_else(|| host.default_output_device())
        .ok_or_else(|| anyhow!("output device not found"))
}

pub fn cpal_device_names() -> Result<(Vec<String>, Vec<String>)> {
    let host = cpal::default_host();
    let inputs = host.input_devices()?.map(|d| device_name(&d)).collect();
    let outputs = host.output_devices()?.map(|d| device_name(&d)).collect();
    Ok((inputs, outputs))
}

fn find_device<I>(devices: I, name: Option<&str>) -> Option<cpal::Device>
where
    I: Iterator<Item = cpal::Device>,
{
    let needle = name?.to_lowercase();
    devices
        .filter_map(|device| {
            let device_name = device_name(&device);
            device_name
                .to_lowercase()
                .contains(&needle)
                .then_some(device)
        })
        .next()
}

pub fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| "<unknown>".to_string())
}

// The engine works in mono, but the stream must be opened with the device's
// native channel count: WASAPI shared mode only accepts the mix-format channel
// count, and since cpal 0.18 `build_*_stream` enforces that via
// `IsFormatSupported` instead of relying on AUTOCONVERTPCM. Channel
// up/downmixing therefore happens in our callbacks, not in the OS.
pub fn default_input_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    device
        .default_input_config()
        .context("failed to get default input config")
}

pub fn default_output_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    device
        .default_output_config()
        .context("failed to get default output config")
}

fn cpal_scratch_frames(config: &cpal::SupportedStreamConfig) -> usize {
    let channels = config.channels().max(1) as usize;
    let frames = match *config.buffer_size() {
        cpal::SupportedBufferSize::Range { max, .. } => max as usize,
        cpal::SupportedBufferSize::Unknown => CPAL_SCRATCH_FALLBACK_SAMPLES,
    };
    // These buffers are moved into CPAL callbacks so sample-format conversion
    // and channel up/downmixing stay allocation-free on the real-time path.
    // Interleaved scratch is frames * channels, so cap the total accordingly.
    frames.clamp(1, (CPAL_MAX_SCRATCH_SAMPLES / channels).max(1))
}

// FormatMessageW cannot resolve AUDCLNT_E_* HRESULTs, so cpal stream errors
// reach the GUI status line as a bare "OS Error -2004287478". Translate the
// codes users actually hit into something actionable. The decimal codes are
// matched against the error text because cpal does not expose the raw HRESULT;
// if the formatting ever changes the hint silently disappears, nothing breaks.
fn with_audclnt_hint<E>(err: E) -> anyhow::Error
where
    E: std::error::Error + Send + Sync + 'static,
{
    let text = err.to_string();
    let hint = [
        // 0x8889000A AUDCLNT_E_DEVICE_IN_USE
        (
            "-2004287478",
            "audio device is in use in exclusive mode by another application",
        ),
        // 0x88890004 AUDCLNT_E_DEVICE_INVALIDATED
        (
            "-2004287484",
            "audio device was removed or its configuration changed",
        ),
        // 0x88890008 AUDCLNT_E_UNSUPPORTED_FORMAT
        ("-2004287480", "audio device rejected the stream format"),
    ]
    .iter()
    .find_map(|(code, hint)| text.contains(code).then_some(*hint));
    match hint {
        Some(hint) => anyhow::Error::new(err).context(hint),
        None => anyhow::Error::new(err),
    }
}

fn build_cpal_input_stream<F>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut on_samples: F,
) -> Result<cpal::Stream>
where
    F: FnMut(&[f32]) + Send + 'static,
{
    let stream_config: cpal::StreamConfig = config.clone().into();
    // CPAL guarantees `data.len()` is a multiple of the channel count, and the
    // chunk size below is too, so every chunk holds whole frames.
    let channels = config.channels().max(1) as usize;
    let frames = cpal_scratch_frames(config);
    let err_fn = |err| tracing::warn!("input stream error: {err}");
    match config.sample_format() {
        cpal::SampleFormat::F32 if channels == 1 => device.build_input_stream(
            stream_config.clone(),
            move |data: &[f32], _| on_samples(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F32 => {
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config.clone(),
                move |data: &[f32], _| {
                    for input in data.chunks(frames * channels) {
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(input, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let mut interleaved = vec![0.0; frames * channels];
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config.clone(),
                move |data: &[i16], _| {
                    for input in data.chunks(frames * channels) {
                        let converted = &mut interleaved[..input.len()];
                        dsp::i16_to_f32_into(input, converted);
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(converted, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut interleaved = vec![0.0; frames * channels];
            let mut mono = vec![0.0; frames];
            device.build_input_stream(
                stream_config.clone(),
                move |data: &[u16], _| {
                    for input in data.chunks(frames * channels) {
                        let converted = &mut interleaved[..input.len()];
                        dsp::u16_to_f32_into(input, converted);
                        let mono = &mut mono[..input.len() / channels];
                        dsp::downmix_to_mono_into(converted, channels, mono);
                        on_samples(mono);
                    }
                },
                err_fn,
                None,
            )
        }
        sample_format => {
            return Err(anyhow!(
                "unsupported input sample format: {sample_format:?}"
            ))
        }
    }
    .map_err(with_audclnt_hint)
    .context("failed to build input stream")
}

fn build_cpal_output_stream<F>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut fill: F,
) -> Result<cpal::Stream>
where
    F: FnMut(&mut [f32]) + Send + 'static,
{
    let stream_config: cpal::StreamConfig = config.clone().into();
    // Same framing guarantee as the input path: chunks hold whole frames.
    let channels = config.channels().max(1) as usize;
    let frames = cpal_scratch_frames(config);
    let err_fn = |err| tracing::warn!("output stream error: {err}");
    match config.sample_format() {
        cpal::SampleFormat::F32 if channels == 1 => device.build_output_stream(
            stream_config.clone(),
            move |data: &mut [f32], _| fill(data),
            err_fn,
            None,
        ),
        cpal::SampleFormat::F32 => {
            let mut mono = vec![0.0; frames];
            device.build_output_stream(
                stream_config.clone(),
                move |data: &mut [f32], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let mono = &mut mono[..output.len() / channels];
                        fill(mono);
                        dsp::upmix_mono_into(mono, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let mut mono = vec![0.0; frames];
            let mut converted = vec![0_i16; frames];
            device.build_output_stream(
                stream_config.clone(),
                move |data: &mut [i16], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let frames_in_chunk = output.len() / channels;
                        let mono = &mut mono[..frames_in_chunk];
                        fill(mono);
                        let converted = &mut converted[..frames_in_chunk];
                        dsp::f32_to_i16_into(mono, converted);
                        dsp::upmix_mono_into(converted, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let mut mono = vec![0.0; frames];
            let mut converted = vec![0_u16; frames];
            device.build_output_stream(
                stream_config.clone(),
                move |data: &mut [u16], _| {
                    for output in data.chunks_mut(frames * channels) {
                        let frames_in_chunk = output.len() / channels;
                        let mono = &mut mono[..frames_in_chunk];
                        fill(mono);
                        let converted = &mut converted[..frames_in_chunk];
                        dsp::f32_to_u16_into(mono, converted);
                        dsp::upmix_mono_into(converted, channels, output);
                    }
                },
                err_fn,
                None,
            )
        }
        sample_format => {
            return Err(anyhow!(
                "unsupported output sample format: {sample_format:?}"
            ))
        }
    }
    .map_err(with_audclnt_hint)
    .context("failed to build output stream")
}
