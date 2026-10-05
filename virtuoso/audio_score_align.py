"""Prototype: align a real audio recording to a score via chroma + DTW,
instead of blind note transcription + symbolic MIDI alignment.

Chroma features collapse octaves (pitch class only), so pedal-resonance
phantom octave doublings don't corrupt the match. DTW enforces a
monotonic, order-preserving path, so it can't jump backward/forward
across the piece the way the symbolic aligner did.
"""
from pathlib import Path
import numpy as np
import librosa
import pretty_midi


SR = 22050
HOP_LENGTH = 512


def synthesize_score_audio(score_midi_path, sr=SR):
    pm = pretty_midi.PrettyMIDI(str(score_midi_path))
    return pm.synthesize(fs=sr)


def load_performance_audio(audio_path, sr=SR):
    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    return y


def compute_chroma(y, sr=SR, hop_length=HOP_LENGTH):
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    # avoid all-zero (silent) frames producing NaN cosine distances
    return chroma + 1e-6


def align(score_midi_path, performance_audio_path, sr=SR, hop_length=HOP_LENGTH):
    score_audio = synthesize_score_audio(score_midi_path, sr=sr)
    perf_audio = load_performance_audio(performance_audio_path, sr=sr)

    score_chroma = compute_chroma(score_audio, sr=sr, hop_length=hop_length)
    perf_chroma = compute_chroma(perf_audio, sr=sr, hop_length=hop_length)

    # D[i, j] = cumulative cost; wp = warping path (frame index pairs),
    # ordered from the end of the piece back to the start
    D, wp = librosa.sequence.dtw(X=score_chroma, Y=perf_chroma, metric="cosine")
    wp = wp[::-1]  # chronological order: start of piece -> end

    score_times = librosa.frames_to_time(wp[:, 0], sr=sr, hop_length=hop_length)
    perf_times = librosa.frames_to_time(wp[:, 1], sr=sr, hop_length=hop_length)

    return {
        "score_times": score_times,
        "perf_times": perf_times,
        "score_duration": len(score_audio) / sr,
        "perf_duration": len(perf_audio) / sr,
        "cost": float(D[-1, -1] / len(wp)),
    }


def score_time_to_perf_time(alignment, score_time):
    """Interpolate the warping path to map an arbitrary score timestamp
    to its corresponding estimated performance timestamp."""
    return float(np.interp(score_time, alignment["score_times"], alignment["perf_times"]))


def map_notes(alignment, xml_notes):
    """Map each score note (with .note_duration.time_position in seconds,
    from the mechanical score render) to its estimated real-performance time."""
    mapped = []
    for note in xml_notes:
        score_t = note.note_duration.time_position
        perf_t = score_time_to_perf_time(alignment, score_t)
        mapped.append(perf_t)
    return mapped


def local_tempo_ratio(alignment, score_time, window=4.0, clip=(0.2, 5.0)):
    """Estimate how much faster/slower the performance is moving relative
    to the score's mechanical tempo at a given score timestamp. >1 = performance
    plays that passage slower than the mechanical score; <1 = faster.

    A wider window than a single note-to-note gap is used, and the result is
    clipped to a physically plausible range, because the warping path can have
    short near-flat "staircase" plateaus (sustained/held passages) where a
    naive local finite-difference slope blows up or collapses toward zero --
    not because tempo actually did that, but because the window is too narrow
    relative to the plateau.
    """
    t0 = score_time_to_perf_time(alignment, score_time - window / 2)
    t1 = score_time_to_perf_time(alignment, score_time + window / 2)
    ratio = (t1 - t0) / window
    return float(np.clip(ratio, clip[0], clip[1]))


def loudness_at(perf_audio, sr, perf_time, window=0.15):
    start = max(0, int((perf_time - window / 2) * sr))
    end = min(len(perf_audio), int((perf_time + window / 2) * sr))
    if end <= start:
        return 0.0
    return float(np.sqrt(np.mean(perf_audio[start:end] ** 2)))


def compute_onset_strength(perf_audio, sr=SR, hop_length=HOP_LENGTH):
    """Spectral-flux onset strength envelope: how much new energy appears
    at each instant. A much better proxy for "how hard was this note struck"
    than raw RMS, since RMS also picks up sustain/pedal resonance from
    earlier notes rather than just the attack transient."""
    return librosa.onset.onset_strength(y=perf_audio, sr=sr, hop_length=hop_length)


def compute_rms_envelope(perf_audio, sr=SR, hop_length=HOP_LENGTH):
    return librosa.feature.rms(y=perf_audio, hop_length=hop_length)[0]


def sustain_fraction(rms_env, sr, hop_length, onset_time, next_onset_time,
                      peak_window=0.05, decay_threshold=0.15):
    """How long a note's energy stays audible, as a fraction of the gap
    until the next note's onset. ~1.0 = sound fills the whole gap (legato /
    pedaled); ~0.0 = sound decays almost immediately (detached / staccato).

    Proxy for VirtuosoNet's articulation feature (held-duration / notated-
    duration), since audio alone can't cleanly separate overlapping/pedaled
    notes into individual durations the way aligned MIDI can.
    """
    gap = next_onset_time - onset_time
    if gap <= 0.03:
        return None  # near-simultaneous onsets (e.g. a chord) -- not meaningful
    onset_frame = int(round(onset_time * sr / hop_length))
    next_frame = int(round(next_onset_time * sr / hop_length))
    peak_frames = max(1, int(round(peak_window * sr / hop_length)))
    if onset_frame >= len(rms_env):
        return None
    peak_region = rms_env[onset_frame: min(onset_frame + peak_frames, len(rms_env))]
    if len(peak_region) == 0 or peak_region.max() <= 1e-6:
        return None
    peak = peak_region.max()
    window = rms_env[onset_frame: min(next_frame, len(rms_env))]
    below = np.where(window < peak * decay_threshold)[0]
    if len(below) == 0:
        decay_frame_offset = len(window)  # never decays within the gap: full sustain
    else:
        decay_frame_offset = below[0]
    decay_time = onset_time + decay_frame_offset * hop_length / sr
    return float(np.clip((decay_time - onset_time) / gap, 0.0, 1.0))


def onset_strength_at(onset_env, sr, hop_length, perf_time, window=0.1):
    """Peak onset strength in a short window around perf_time, so a slightly
    imprecise time-map still lands near the true attack rather than missing it."""
    center = librosa.time_to_frames(perf_time, sr=sr, hop_length=hop_length)
    half = max(1, int(round(window / 2 * sr / hop_length)))
    start = max(0, center - half)
    end = min(len(onset_env), center + half + 1)
    if end <= start:
        return 0.0
    return float(np.max(onset_env[start:end]))
