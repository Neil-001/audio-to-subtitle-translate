<div align="center">
  <h1>🎙️ Audio → SRT Subtitles (with Translation)</h1>
  <a target="_blank" href="https://colab.research.google.com/github/Neil-001/audio-to-subtitle-translate/blob/main/speech_to_srt_translate.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a target="_blank" href="https://molab.marimo.io/github/Neil-001/audio-to-subtitle-translate/blob/main/speech_to_srt_translate.py">
    <img src="https://marimo.io/molab-shield.svg" alt="Open in molab"/>
  </a>
</div>

A Google Colab notebook that transcribes audio into timed SRT subtitles using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), then optionally translates them into another language.

An equivalent marimo notebook is also available at [speech_to_srt_translate.py](speech_to_srt_translate.py) and can be opened in molab with the badge above.

Runs on a **free-tier T4 GPU** (15 GB VRAM).

## What it does

1. **Transcribes** audio with [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
2. **Aligns** word-level timestamps via [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
3. **Generates** an `.srt` subtitle file with grouped, timed segments
4. **Translates** (optional) using Gemini or Google Translate:

| Method | Quality | Cost | Notes |
|---|---|---|---|
| **Gemini 3.6 Flash** | Best | Free (API key, rate limited) | Requires a [Google AI Studio](https://aistudio.google.com/app/api-keys) key |
| **Google Translate** | Good | Free | Via `deep-translator`, no key needed |

## Quick start

1. **Open in Colab** — upload the notebook or use "Open in Colab" from GitHub
2. **Select a T4 GPU** runtime (`Runtime → Change runtime type → T4 GPU`)
3. **Upload your audio file** (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, etc.)
4. **Configure** the first code cell:
   - `AUDIO_PATH` — filename of your uploaded audio
   - `SOURCE_LANGUAGE` / `TARGET_LANGUAGE` — e.g. `"Japanese"` → `"English"` (default)
   - Toggle which translation methods to run
5. **Run all cells** — SRT files are generated and downloaded automatically

## Configuration

All settings live in the **CONFIG** section (cells 0–1):

```python
AUDIO_PATH = "ja_audio.mp3"

SOURCE_LANGUAGE = "Japanese"
TARGET_LANGUAGE = "English"

# Toggle translation backends
TRANSLATE_USING_GEMINI = True
TRANSLATE_USING_GT     = True
```

### Technical parameters

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SEC` | `200` | Audio chunk length in seconds. Lower = less VRAM |
| `MAX_INFERENCE_BATCH_SIZE` | `32` | ASR batch size. Reduce to `1` if you hit OOM |
| `GEMINI_BATCH_SIZE` | `100` | Subtitle lines per Gemini API call. Larger = fewer requests and faster translation, up to prompt limits |

## Supported languages

The notebook ships with ISO 639-1 codes for 29 languages (Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese). Add more by extending the `LANG_CODES` dict.

## How it fits in 15 GB VRAM

The ASR model (~3.4 GB) and ForcedAligner (~1.2 GB) are loaded **sequentially** — each is freed before the next is loaded. Audio is split into configurable-length chunks to avoid O(n²) attention blowup. All models use `bfloat16` and `flash_attention_2`.

Speed wins that do not change the pipeline structure:

1. Increase `GEMINI_BATCH_SIZE` if the prompt still fits comfortably. That reduces API calls.
2. Keep only the translation backend you actually need. Removing unused branches lowers notebook runtime and setup cost.
3. If GPU memory allows it, test a larger `CHUNK_SEC` or `MAX_INFERENCE_BATCH_SIZE` on your own audio. Those are the main knobs for ASR throughput.

## Output files

| File | Contents |
|---|---|
| `{name}_{src}.srt` | Source-language subtitles with timestamps |
| `{name}_{tgt}_gemini.srt` | Gemini translation |
| `{name}_{tgt}_gtranslate.srt` | Google Translate translation |

A side-by-side comparison table is also displayed in the notebook.

## Requirements

- Google Colab with a T4 GPU (free tier works)
- (Optional) A [Google AI Studio API key](https://aistudio.google.com/app/api-keys) for Gemini translation — add it as a Colab secret named `GOOGLE_API_KEY`

Dependencies are installed automatically by the notebook:

```
qwen-asr  transformers  sentencepiece  deep-translator  google-genai  flash-attn  librosa
```

## License

MIT
