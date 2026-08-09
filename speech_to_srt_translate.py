# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "marimo",
#     "numpy",
#     "torch==2.10.*",
#     "transformers",
#     "sentencepiece",
#     "qwen-asr",
#     "deep-translator",
#     "google-genai",
#     "imageio-ffmpeg",
#     "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎙️ Audio → SRT Subtitles (with Translation)

    This notebook:
    1. **Transcribes** uploaded audio using [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) with word-level timestamps via the ForcedAligner
    2. **Generates** an SRT subtitle file from the transcription
    3. **Translates** the SRT to a target language using Gemini or Google Translate

    Upload audio and choose the language and translation settings in the controls below.

    **Requirements:** A CUDA-enabled runtime with a **T4 GPU** (free tier works).

    > ⚠️ Molab installs the packages declared in this notebook, including the
    > Python 3.12 / CUDA 12.8 / PyTorch 2.10 prebuilt FlashAttention wheel.
    """)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · CONFIG

    General Config
    """)


@app.cell
def _(mo):
    # ISO 639-1 language codes — used for file naming and translation APIs.
    lang_codes = {
        "Arabic": "ar",
        "Chinese": "zh",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Finnish": "fi",
        "French": "fr",
        "German": "de",
        "Greek": "el",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hungarian": "hu",
        "Indonesian": "id",
        "Italian": "it",
        "Japanese": "ja",
        "Korean": "ko",
        "Malay": "ms",
        "Norwegian": "no",
        "Polish": "pl",
        "Portuguese": "pt",
        "Romanian": "ro",
        "Russian": "ru",
        "Spanish": "es",
        "Swedish": "sv",
        "Thai": "th",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Vietnamese": "vi",
    }
    languages = list(lang_codes)
    audio_file = mo.ui.file(
        filetypes=["audio/*", ".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        kind="area",
        max_size=1_000_000_000,  # 1 GB
        label="Audio file",
    )
    source_language = mo.ui.dropdown(
        options=languages, value="Japanese", label="Source language"
    )
    target_language = mo.ui.dropdown(
        options=languages, value="English", label="Target language"
    )
    translate_using_gemini = mo.ui.checkbox(value=True, label="Translate with Gemini")
    translate_using_gt = mo.ui.checkbox(
        value=True, label="Translate with Google Translate"
    )
    gemini_api_key = mo.ui.text(
        kind="password",
        label="Gemini API key (optional; can also use GOOGLE_API_KEY)",
    )
    config = mo.vstack(
        [
            mo.md("### General settings"),
            audio_file,
            mo.hstack([source_language, target_language]),
            mo.hstack([translate_using_gemini, translate_using_gt]),
            gemini_api_key,
        ]
    )
    config
    return (
        audio_file,
        gemini_api_key,
        lang_codes,
        source_language,
        target_language,
        translate_using_gemini,
        translate_using_gt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Technical Parameters
    """)


@app.cell
def _(mo):
    # Chunk length (seconds) — Each audio chunk is processed separately
    # to fit in GPU memory. Shorter = less VRAM but more chunks.
    # 20 s works on a free-tier T4 (15 GB). Increase if possible.
    chunk_sec = mo.ui.slider(
        20, 600, value=200, step=10, label="Chunk length (seconds)"
    )

    # Maximum batch size for the ASR Model
    max_inference_batch_size = mo.ui.slider(
        1, 256, value=32, label="Maximum ASR inference batch size"
    )

    # Gemini translation batch size — Number of subtitle lines sent
    # per API call. Larger = fewer calls but longer prompts.
    gemini_batch_size = mo.ui.slider(
        1, 200, value=100, label="Gemini translation batch size"
    )
    technical_config = mo.vstack(
        [
            mo.md("### Technical settings"),
            chunk_sec,
            max_inference_batch_size,
            gemini_batch_size,
        ]
    )
    technical_config
    return chunk_sec, gemini_batch_size, max_inference_batch_size


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Prepare audio

    The uploaded audio is saved temporarily for `ffmpeg`. Select a file in the
    configuration controls above to continue.
    """)


@app.cell
def _(audio_file, lang_codes, source_language, target_language, mo):
    import tempfile
    from pathlib import Path as _Path

    mo.stop(
        not audio_file.value, mo.md("Upload an audio file to begin.").callout("info")
    )
    uploaded_name = audio_file.name()
    suffix = _Path(uploaded_name).suffix if uploaded_name else ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as uploaded:
        uploaded.write(audio_file.contents() or b"")
        audio_path = uploaded.name
    source_language_value = source_language.value
    target_language_value = target_language.value
    src_code = lang_codes[source_language_value]
    tgt_code = lang_codes[target_language_value]
    return (
        audio_path,
        uploaded_name,
        source_language_value,
        src_code,
        target_language_value,
        tgt_code,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Transcribe with Qwen3-ASR-1.7B

    To fit on a T4 (15 GB VRAM), we run ASR and alignment as **two separate steps** so both models are never loaded at the same time.
    """)


@app.cell
def _(audio_path, chunk_sec, max_inference_batch_size):
    import gc
    import os as _os
    import subprocess

    import imageio_ffmpeg
    import numpy as np
    import torch
    from qwen_asr import Qwen3ASRModel

    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Help PyTorch reuse freed VRAM fragments
    SR = 16000

    def load_audio(path: str, sr: int = SR) -> np.ndarray:
        """Decode audio to a mono float32 numpy array using ffmpeg.
        Supports any format ffmpeg handles (wav, mp3, m4a, flac, ogg, …)
        without deprecated fallback paths.
        """  # qwen-asr expects 16 kHz
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-loglevel",
            "error",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        return np.frombuffer(proc.stdout, dtype=np.float32)

    print("Loading uploaded audio …")
    full_wav = load_audio(audio_path)
    total_dur = len(full_wav) / SR
    print(f"Duration: {total_dur:.1f} s ({total_dur / 60:.1f} min)")
    chunk_samples = chunk_sec.value * SR
    audio_chunks = []
    for start in range(0, len(full_wav), chunk_samples):
        chunk = full_wav[start : start + chunk_samples]
        if len(chunk) < SR // 2:
            continue
        audio_chunks.append((float(start) / SR, chunk))  # raw 32-bit float PCM
    print(f"Split into {len(audio_chunks)} chunks of ≤{chunk_sec.value} s each.")
    print("\nLoading Qwen3-ASR-1.7B …")  # mono
    asr_model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
        max_inference_batch_size=max_inference_batch_size.value,
        max_new_tokens=4096,
    )
    # --- Load and split audio manually ---
    # The audio tower's attention is O(n²) on sequence length, so we split
    # into short segments and feed each one individually.
    # --- Load ASR model ---
    print(
        "✅ ASR model loaded."
    )  # resample to target sample rate  # pipe to stdout  # skip tiny tail < 0.5 s
    return SR, asr_model, audio_chunks, gc, torch


@app.cell
def _(SR, asr_model, audio_chunks, gc, source_language_value, torch):
    # Transcribe each chunk individually to stay within T4 VRAM
    all_texts = []
    for _i, (_offset, _chunk_wav) in enumerate(audio_chunks):
        print(
            f"  Chunk {_i + 1}/{len(audio_chunks)}  [{_offset:.1f}s – {_offset + len(_chunk_wav) / SR:.1f}s] …",
            end=" ",
        )
        with torch.inference_mode():
            r = asr_model.transcribe(
                audio=(_chunk_wav, SR),
                language=source_language_value,
                return_time_stamps=False,
            )
        _text = r[0].text.strip()
        all_texts.append(_text)
        print(_text[:80])
    transcribed_text = "".join(all_texts)
    print(f"\n{'─' * 60}")
    print(f"Full transcription ({len(transcribed_text)} chars):\n{transcribed_text}")
    del asr_model
    gc.collect()
    torch.cuda.empty_cache()
    # Free ASR model before loading the aligner
    print("\n✅ ASR model unloaded — GPU memory freed.")
    return (all_texts,)


@app.cell
def _(SR, all_texts, audio_chunks, gc, source_language_value, torch):
    # --- Step 2: Forced Aligner for word-level timestamps ---
    from dataclasses import replace

    from qwen_asr import Qwen3ForcedAligner

    print("Loading Qwen3-ForcedAligner-0.6B …")
    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    print("✅ Aligner loaded.")
    print("Aligning timestamps …")
    time_stamps = []
    for _i, (_offset, _chunk_wav) in enumerate(audio_chunks):
        chunk_text = all_texts[_i]
        if not chunk_text.strip():
            continue
        print(f"  Aligning chunk {_i + 1}/{len(audio_chunks)} …")
        # Align each chunk separately (same chunking as ASR) and shift timestamps
        with torch.inference_mode():
            alignment = aligner.align(
                audio=(_chunk_wav, SR), text=chunk_text, language=source_language_value
            )
        for stamp in alignment[0]:
            shifted = replace(
                stamp,
                start_time=stamp.start_time + _offset,
                end_time=stamp.end_time + _offset,
            )
            time_stamps.append(shifted)
        del alignment
    print(f"\nTimestamp segments: {len(time_stamps)}")
    if time_stamps:
        print(
            f"First: {time_stamps[0].text} [{time_stamps[0].start_time:.2f}s – {time_stamps[0].end_time:.2f}s]"
        )
    del aligner
    gc.collect()
    torch.cuda.empty_cache()
    # Free aligner
    print("\n✅ Aligner unloaded — GPU memory freed.")
    return (time_stamps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Generate SRT Subtitles
    """)


@app.cell
def _(audio_name, source_language_value, src_code, time_stamps):
    from datetime import timedelta
    from pathlib import Path

    def format_srt_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = total_seconds % 3600 // 60
        secs = total_seconds % 60
        millis = int(td.microseconds / 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def group_timestamps_to_subtitles(
        stamps,
        max_chars: int = 40,
        max_duration: float = 7.0,
        gap_threshold: float = 0.6,
    ):
        """
        Group word-level timestamps into subtitle segments.

        Args:
            stamps: list of timestamp objects with .text, .start_time, .end_time
            max_chars: max characters per subtitle line
            max_duration: max duration (seconds) per subtitle
            gap_threshold: silence gap (seconds) that forces a new subtitle
        """
        if not stamps:
            return []
        subtitles = []
        current_text = ""
        current_start = stamps[0].start_time
        current_end = stamps[0].end_time
        for _i, stamp in enumerate(stamps):
            start_new = False
            if _i == 0:
                current_text = stamp.text
                current_start = stamp.start_time
                current_end = stamp.end_time
                continue
            gap = (
                stamp.start_time - current_end
            )  # Decide whether to start a new subtitle
            new_duration = stamp.end_time - current_start
            new_len = len(current_text) + len(stamp.text)
            if (
                gap > gap_threshold
                or new_duration > max_duration
                or new_len > max_chars
            ):
                start_new = True
            if start_new:
                subtitles.append((current_start, current_end, current_text.strip()))
                current_text = stamp.text
                current_start = (
                    stamp.start_time
                )  # Check gap between previous and current word
                current_end = stamp.end_time
            else:
                current_text += stamp.text
                current_end = stamp.end_time
        if current_text.strip():
            subtitles.append((current_start, current_end, current_text.strip()))
        return subtitles

    def build_srt(subtitles) -> str:
        """Build SRT string from list of (start, end, text) tuples."""
        lines = []
        for idx, (start, end, _text) in enumerate(subtitles, 1):
            lines.append(str(idx))
            lines.append(f"{format_srt_time(start)} --> {format_srt_time(end)}")
            lines.append(_text)
            lines.append("")
        return "\n".join(lines)  # Don't forget the last segment

    subtitles_src = group_timestamps_to_subtitles(time_stamps)
    srt_src = build_srt(subtitles_src)
    base_name = Path(audio_name or "audio").stem
    output_dir = Path.cwd() / "srt_output"
    output_dir.mkdir(exist_ok=True)
    srt_src_path = output_dir / f"{base_name}_{src_code}.srt"
    with open(srt_src_path, "w", encoding="utf-8") as _f:
        _f.write(srt_src)
    print(f"✅ {source_language_value} SRT saved to: {srt_src_path}")
    print(f"   {len(subtitles_src)} subtitle segments\n")
    print("--- Preview (first 10 segments) ---")
    # Build source-language SRT
    # Save
    print("\n".join(srt_src.split("\n")[:40]))  # blank line separator
    return base_name, build_srt, format_srt_time, srt_src_path, subtitles_src


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Translate Subtitles with Gemini (free via Google AI Studio)

    Get an API key at https://aistudio.google.com/app/api-keys. By default, it uses
    a free tier, where billing isn't set up, so you shouldn't worry about getting
    charged. You can check the rate limits at https://aistudio.google.com/rate-limit?timeRange=last-28-days, and change the model accordingly by setting `GEMINI_MODEL` in the cell below.

    Gemini 3.6 Flash is the default translation model here.
    """)


@app.cell
def _(
    gemini_api_key,
    gemini_batch_size,
    source_language_value,
    target_language_value,
    tgt_code,
    translate_using_gemini,
    base_name,
    build_srt,
    srt_src_path,
    subtitles_src,
):
    import os as _os
    import time

    subtitles_tgt_gemini = []
    srt_tgt_gemini_path = None
    if translate_using_gemini.value:
        import json

        from google import genai

        api_key = gemini_api_key.value or _os.environ.get("GOOGLE_API_KEY", "")
        assert api_key, (
            "Enter a Gemini API key in the configuration controls or set GOOGLE_API_KEY."
        )
        client = genai.Client(api_key=api_key)
        GEMINI_MODEL = "gemini-3.6-flash"
        SYSTEM_PROMPT = f"Translate numbered {source_language_value} subtitle lines into {target_language_value}. Return only a JSON array of strings, one translation per line, in the same order. Keep subtitles natural, concise, and easy to read on screen. Preserve meaning, tone, speaker intent, and punctuation where useful. Do not add numbering, explanations, or extra text. Do not merge or split lines unless clarity requires it. Use plain dialogue and keep sound effects or non-speech text concise."

        def translate_with_gemini(
            texts: list[str], batch_size: int = gemini_batch_size.value
        ) -> list[str]:
            """Translate texts using Gemini in batches."""
            all_translations = []
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                batch_num = batch_start // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                print(
                    f"  Batch {batch_num}/{total_batches} ({len(batch)} lines) …",
                    end=" ",
                )
                numbered = "\n".join((f"{_i + 1}. {t}" for _i, t in enumerate(batch)))
                prompt = f"{SYSTEM_PROMPT}\n\nSubtitle lines:\n{numbered}"
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model=GEMINI_MODEL, contents=prompt
                        )
                        raw = (response.text or "").strip()
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[1]
                            raw = raw.rsplit("```", 1)[0]
                        translations = json.loads(raw)
                        assert isinstance(translations, list) and len(
                            translations
                        ) == len(batch)
                        all_translations.extend(translations)
                        print("✅")
                        break
                    except (json.JSONDecodeError, AssertionError, Exception) as e:
                        if attempt < 2:
                            print(f"⚠️ retry ({e.__class__.__name__}) …", end=" ")
                            time.sleep(2**attempt)
                        else:
                            print(f"❌ fallback (kept {source_language_value})")
                            all_translations.extend(batch)
                if batch_start + batch_size < len(texts):
                    time.sleep(1)
            return all_translations

        src_texts_gemini = [_text for _, _, _text in subtitles_src]
        print(
            f"Translating {len(src_texts_gemini)} subtitles with Gemini ({GEMINI_MODEL}) …\n"
        )
        tgt_texts_gemini = translate_with_gemini(src_texts_gemini)
        print("\n✅ Gemini translation complete.")
        subtitles_tgt_gemini = [
            (start, end, tgt_text)
            for (start, end, _), tgt_text in zip(subtitles_src, tgt_texts_gemini)
        ]
        srt_tgt_gemini = build_srt(subtitles_tgt_gemini)
        srt_tgt_gemini_path = srt_src_path.parent / f"{base_name}_{tgt_code}_gemini.srt"
        with open(srt_tgt_gemini_path, "w", encoding="utf-8") as _f:
            _f.write(srt_tgt_gemini)
        print(f"✅ Gemini {target_language_value} SRT saved to: {srt_tgt_gemini_path}")
        print(f"   {len(subtitles_tgt_gemini)} subtitle segments\n")
        print("--- Preview (first 10 segments) ---")
        print("\n".join(srt_tgt_gemini.split("\n")[:40]))
    else:
        print("⏭️  Gemini translation skipped")
    return srt_tgt_gemini_path, subtitles_tgt_gemini, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Translate Subtitles with Google Translate (free)

    Uses the free Google Translate API via `deep-translator`. No API key needed. Quality sits below Gemini.
    """)


@app.cell
def _(
    src_code,
    tgt_code,
    translate_using_gt,
    base_name,
    build_srt,
    srt_src_path,
    subtitles_src,
    time,
):
    subtitles_tgt_gt = []
    srt_tgt_gt_path = None
    if translate_using_gt.value:
        from deep_translator import GoogleTranslator

        gtranslator = GoogleTranslator(source=src_code, target=tgt_code)

        def translate_with_google(texts: list[str], batch_size: int = 50) -> list[str]:
            """Translate texts using Google Translate (free).

            deep-translator's GoogleTranslator.translate_batch() has a ~5000 char
            limit per request, so we send in small batches.
            """
            all_translations = []
            for batch_start in range(0, len(texts), batch_size):
                batch = texts[batch_start : batch_start + batch_size]
                batch_num = batch_start // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                print(
                    f"  Batch {batch_num}/{total_batches} ({len(batch)} lines) …",
                    end=" ",
                )
                try:
                    translated = gtranslator.translate_batch(batch)
                    all_translations.extend(translated)
                    print("✅")
                except Exception as e:
                    print(
                        f"⚠️ batch failed ({e.__class__.__name__}), trying one-by-one …"
                    )
                    for t in batch:
                        try:
                            all_translations.append(gtranslator.translate(t))
                        except Exception:
                            all_translations.append(t)
                        time.sleep(0.3)
                if batch_start + batch_size < len(texts):
                    time.sleep(0.5)
            return all_translations

        src_texts_gt = [_text for _, _, _text in subtitles_src]
        print(f"Translating {len(src_texts_gt)} subtitles with Google Translate …\n")
        tgt_texts_gt = translate_with_google(src_texts_gt)
        print("\n✅ Google Translate complete.")
        subtitles_tgt_gt = [
            (start, end, tgt_text)
            for (start, end, _), tgt_text in zip(subtitles_src, tgt_texts_gt)
        ]
        srt_tgt_gt = build_srt(subtitles_tgt_gt)
        srt_tgt_gt_path = srt_src_path.parent / f"{base_name}_{tgt_code}_gtranslate.srt"
        with open(srt_tgt_gt_path, "w", encoding="utf-8") as _f:
            _f.write(srt_tgt_gt)
        print(f"✅ Google Translate SRT saved to: {srt_tgt_gt_path}")
        print(f"   {len(subtitles_tgt_gt)} subtitle segments\n")
        print("--- Preview (first 10 segments) ---")
        print("\n".join(srt_tgt_gt.split("\n")[:40]))
    else:
        print("⏭️  Google Translate skipped")
    return srt_tgt_gt_path, subtitles_tgt_gt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Side-by-Side Comparison
    """)


@app.cell
def _(
    mo,
    source_language_value,
    format_srt_time,
    subtitles_src,
    subtitles_tgt_gemini,
    subtitles_tgt_gt,
):
    cols = [(source_language_value, subtitles_src)]
    if subtitles_tgt_gemini:
        cols.append(("Gemini 3.6 Flash", subtitles_tgt_gemini))
    if subtitles_tgt_gt:
        cols.append(("Google Translate", subtitles_tgt_gt))
    ncols = 2 + len(cols)
    header_cells = "".join((f"<th>{name}</th>" for name, _ in cols))
    header = f"<tr><th>#</th><th>Time</th>{header_cells}</tr>"
    rows = []
    for _i, (s, e, src) in enumerate(subtitles_src):
        time_str = f"{format_srt_time(s)} → {format_srt_time(e)}"
        data_cells = ""
        for _, subs in cols:
            _text = subs[_i][2] if _i < len(subs) else ""
            data_cells += f"<td>{_text}</td>"
        rows.append(
            f"<tr><td>{_i + 1}</td><td style='white-space:nowrap'>{time_str}</td>{data_cells}</tr>"
        )
        if _i >= 29:
            remaining = len(subtitles_src) - 30
            if remaining > 0:
                rows.append(
                    f"<tr><td colspan='{ncols}'><i>… and {remaining} more segments</i></td></tr>"
                )
            break
    html = (
        "<table border='1' cellpadding='4' style='border-collapse:collapse;font-size:13px'>"
        + header
        + "\n".join(rows)
        + "</table>"
    )
    mo.Html(html)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Download SRT Files
    """)


@app.cell
def _(
    mo,
    srt_src_path,
    srt_tgt_gemini_path,
    srt_tgt_gt_path,
):
    downloads = [
        mo.download(
            srt_src_path.read_bytes(),
            filename=srt_src_path.name,
            mimetype="text/plain",
            label="Download source SRT",
        )
    ]
    if srt_tgt_gemini_path:
        downloads.append(
            mo.download(
                srt_tgt_gemini_path.read_bytes(),
                filename=srt_tgt_gemini_path.name,
                mimetype="text/plain",
                label="Download Gemini SRT",
            )
        )
    if srt_tgt_gt_path:
        downloads.append(
            mo.download(
                srt_tgt_gt_path.read_bytes(),
                filename=srt_tgt_gt_path.name,
                mimetype="text/plain",
                label="Download Google Translate SRT",
            )
        )
    mo.hstack(downloads)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Cleanup (Optional)

    Free GPU memory if you want to run other things in this session.
    """)


@app.cell
def _(gc, torch):
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ GPU memory freed.")


if __name__ == "__main__":
    app.run()
