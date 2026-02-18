# 🎙️ Audio → SRT Subtitles (with Translation)

A Google Colab notebook that transcribes audio into timed SRT subtitles using [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), then optionally translates them into another language.

Runs on a **free-tier T4 GPU** (15 GB VRAM).

## What it does

1. **Transcribes** audio with [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
2. **Aligns** word-level timestamps via [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
3. **Generates** an `.srt` subtitle file with grouped, timed segments
4. **Translates** (optional) using up to three backends:

| Method | Quality | Cost | Notes |
|---|---|---|---|
| **Gemini 3 Flash** | Best | Free (API key, rate limited) | Requires a [Google AI Studio](https://aistudio.google.com/app/api-keys) key |
| **Google Translate** | Good | Free | Via `deep-translator`, no key needed |
| **opus-mt** | Basic | Free | On-device [Helsinki-NLP/opus-mt](https://huggingface.co/Helsinki-NLP) model; not all language pairs available |

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
TRANSLATE_USING_OPUS   = False
```

### Technical parameters

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SEC` | `200` | Audio chunk length in seconds. Lower = less VRAM |
| `MAX_INFERENCE_BATCH_SIZE` | `32` | ASR batch size. Reduce to `1` if you hit OOM |
| `GEMINI_BATCH_SIZE` | `100` | Subtitle lines per Gemini API call |

## Supported languages

The notebook ships with ISO 639-1 codes for 29 languages (Arabic, Chinese, Czech, Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, Vietnamese). Add more by extending the `LANG_CODES` dict.

> **Note:** Qwen3-ASR supports many languages for transcription, but opus-mt has limited language pair coverage.
> Gemini and Google Translate support virtually all pairs.

## How it fits in 15 GB VRAM

The ASR model (~3.4 GB) and ForcedAligner (~1.2 GB) are loaded **sequentially** — each is freed before the next is loaded. Audio is split into configurable-length chunks to avoid O(n²) attention blowup. All models use `bfloat16` and `flash_attention_2`.

## Output files

| File | Contents |
|---|---|
| `{name}_{src}.srt` | Source-language subtitles with timestamps |
| `{name}_{tgt}_gemini.srt` | Gemini translation |
| `{name}_{tgt}_gtranslate.srt` | Google Translate translation |
| `{name}_{tgt}_opus.srt` | opus-mt translation |

A side-by-side comparison table is also displayed in the notebook.

## Requirements

- Google Colab with a T4 GPU (free tier works)
- (Optional) A [Google AI Studio API key](https://aistudio.google.com/app/api-keys) for Gemini translation — add it as a Colab secret named `GOOGLE_API_KEY`

Dependencies are installed automatically by the notebook:

```
qwen-asr  transformers  sentencepiece  sacremoses  deep-translator  google-genai  flash-attn  librosa
```

## License

MIT
