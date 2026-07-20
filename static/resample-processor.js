// AudioWorkletProcessor: capture mic at native rate, downsample to 16kHz,
// emit 512-sample int16 PCM chunks via port.postMessage to main thread.
//
// Why a worklet and not ScriptProcessorNode?
//   - ScriptProcessor is deprecated and runs on the main thread (jank).
//   - AudioWorklet runs on a dedicated audio thread.
//   - The main thread only sees already-resampled 16kHz chunks ready for WS.
//
// Why resample here?
//   - getUserMedia({audio:{sampleRate:16000}}) is a HINT, not enforced.
//     The AudioContext always runs at its hardware-native rate (commonly
//     44.1k or 48k). Asking for sampleRate:16000 on AudioContext is also
//     ignored on most browsers. We must resample ourselves.

const TARGET_SAMPLE_RATE = 16000;
const CHUNK_SAMPLES = 512; // matches Silero VAD input

class ResampleProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    // Rolling buffer of input samples at native rate
    this.inputBuffer = [];
    // Resampling ratio — set on first process() call once we know native SR
    this.ratio = null;
    this.fractionalOffset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const native = input[0]; // Float32Array at native SR
    if (this.ratio === null) {
      this.ratio = sampleRate / TARGET_SAMPLE_RATE;
    }

    // Append new native samples to the rolling buffer
    for (let i = 0; i < native.length; i++) {
      this.inputBuffer.push(native[i]);
    }

    // Drain the buffer in CHUNK_SAMPLES-sized 16kHz chunks via linear resample.
    // Linear interpolation is sufficient for VAD + ASR quality at 16kHz.
    while (true) {
      const out = new Float32Array(CHUNK_SAMPLES);
      // We need inputBuffer to cover CHUNK_SAMPLES * ratio native samples,
      // starting at the current fractional offset.
      const neededNative = Math.ceil(CHUNK_SAMPLES * this.ratio + this.fractionalOffset);
      if (this.inputBuffer.length < neededNative) {
        break; // not enough yet; wait for next process() call
      }

      for (let i = 0; i < CHUNK_SAMPLES; i++) {
        const srcPos = this.fractionalOffset + i * this.ratio;
        const srcIdx = Math.floor(srcPos);
        const frac = srcPos - srcIdx;
        const s1 = this.inputBuffer[srcIdx];
        const s2 = this.inputBuffer[srcIdx + 1] !== undefined
          ? this.inputBuffer[srcIdx + 1]
          : s1;
        out[i] = s1 + (s2 - s1) * frac;
      }

      // Advance fractional offset past consumed samples
      const consumed = CHUNK_SAMPLES * this.ratio;
      this.fractionalOffset += consumed;
      const consumedInt = Math.floor(this.fractionalOffset);
      this.fractionalOffset -= consumedInt;
      this.inputBuffer.splice(0, consumedInt);

      // Convert float32 [-1, 1] → int16 and post to main thread
      const int16 = new Int16Array(CHUNK_SAMPLES);
      for (let i = 0; i < CHUNK_SAMPLES; i++) {
        const clamped = Math.max(-1, Math.min(1, out[i]));
        int16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      }
      this.port.postMessage(int16.buffer, [int16.buffer]);
    }

    return true;
  }
}

registerProcessor("resample-processor", ResampleProcessor);
