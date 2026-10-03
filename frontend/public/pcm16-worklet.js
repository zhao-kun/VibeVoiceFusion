/**
 * AudioWorklet that turns microphone input into 24 kHz mono PCM16 chunks for the realtime transcription API.
 *
 * Posts { pcm: ArrayBuffer, level: number, final: boolean } to the main thread; a 'flush' message
 * posts the partially filled chunk with final=true.
 */
class PCM16Processor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.targetRate = opts.targetRate || 24000;
    this.chunkSamples = Math.round(this.targetRate * (opts.chunkSeconds || 0.1));
    this.ratio = sampleRate / this.targetRate;
    // Read position in input samples relative to the current block; -1 addresses the last sample of the previous block
    this.position = 0;
    this.previous = 0;
    this.mono = new Float32Array(128);
    this.resetChunk();
    this.port.onmessage = (event) => {
      if (event.data === 'flush') {
        this.postChunk(true);
      }
    };
  }

  resetChunk() {
    this.chunk = new Int16Array(this.chunkSamples);
    this.filled = 0;
    this.sumSquares = 0;
  }

  postChunk(final) {
    const pcm = this.chunk.slice(0, this.filled).buffer;
    const level = this.filled > 0 ? Math.sqrt(this.sumSquares / this.filled) : 0;
    this.port.postMessage({ pcm, level, final }, [pcm]);
    this.resetChunk();
  }

  push(sample) {
    const clamped = Math.max(-1, Math.min(1, sample));
    this.chunk[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    this.sumSquares += clamped * clamped;
    if (this.filled === this.chunkSamples) {
      this.postChunk(false);
    }
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) {
      return true;
    }
    const frames = input[0].length;
    if (this.mono.length !== frames) {
      this.mono = new Float32Array(frames);
    }
    const mono = this.mono;
    mono.fill(0);
    for (let c = 0; c < input.length; c++) {
      const channel = input[c];
      for (let i = 0; i < frames; i++) {
        mono[i] += channel[i] / input.length;
      }
    }

    while (this.position < frames - 1) {
      const index = Math.floor(this.position);
      const frac = this.position - index;
      const a = index < 0 ? this.previous : mono[index];
      const b = mono[index + 1];
      this.push(a + (b - a) * frac);
      this.position += this.ratio;
    }
    this.position -= frames;
    this.previous = mono[frames - 1];
    return true;
  }
}

registerProcessor('pcm16-processor', PCM16Processor);
