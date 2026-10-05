"""Turn a chroma+DTW audio-to-score alignment into an accuracy percentage,
by comparing the performer's actual tempo/dynamics curve (derived from the
real recording) against VirtuosoNet's predicted "ideal" composer-style
interpretation of the same score.
"""
import numpy as np
import torch

from .audio_score_align import (
    align, local_tempo_ratio, load_performance_audio, SR, HOP_LENGTH,
    compute_onset_strength, onset_strength_at,
    compute_rms_envelope, sustain_fraction, score_time_to_perf_time,
)
from .inference import InferenceModel
from .pyScoreParser.data_class import ScoreData


def _ideal_features(model: InferenceModel, xml_path, composer, qpm_primo=None):
    score, input_t, edges, note_locations = model.get_input_from_xml(xml_path, composer, qpm_primo)
    with torch.no_grad():
        outputs, _, _, _ = model.model(input_t, None, edges, note_locations, initial_z="zero")
    outputs_scaled = model.scale_model_prediction_to_original(outputs)
    return score, model.model_prediction_to_feature(outputs_scaled)


def _zscore(arr):
    arr = np.asarray(arr, dtype=float)
    std = arr.std()
    if std < 1e-8:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def load_model(checkpoint_path, device="cpu"):
    return InferenceModel(checkpoint_path=checkpoint_path, device=device,
                           output_path="test_result", midi_decode_options={})


def score_audio_performance(xml_path, score_midi_path, composer, performance_audio_path,
                             model=None, checkpoint_path=None, device="cpu"):
    if model is None:
        if checkpoint_path is None:
            raise ValueError("pass either a pre-loaded `model` or a `checkpoint_path`")
        model = load_model(checkpoint_path, device=device)

    ideal_score, ideal_features = _ideal_features(model, xml_path, composer)
    alignment = align(score_midi_path, performance_audio_path)
    perf_audio = load_performance_audio(performance_audio_path)

    notated_qpm = np.array([n.state_fixed.qpm for n in ideal_score.xml_notes])
    score_time = np.array([n.note_duration.time_position for n in ideal_score.xml_notes])

    # ideal (model) tempo as a ratio relative to the score's own notated tempo
    ideal_qpm = 10 ** ideal_features["beat_tempo"]
    ideal_tempo_ratio = ideal_qpm / notated_qpm

    # performer's actual tempo ratio relative to notated tempo, from chroma+DTW:
    # local_tempo_ratio = (real seconds elapsed) / (mechanical-score seconds elapsed),
    # i.e. performer_qpm = notated_qpm / local_tempo_ratio
    perf_local_ratio = np.array([local_tempo_ratio(alignment, t) for t in score_time])
    perf_local_ratio = np.clip(perf_local_ratio, 1e-3, None)
    perf_tempo_ratio = 1.0 / perf_local_ratio

    # shape-match only, like dynamics below: the score's own notated tempo
    # (used as the common denominator for both ratios) is community-sourced
    # and can be miscalibrated, which would otherwise show up as a constant
    # multiplicative offset between ideal and performed ratios having nothing
    # to do with actual rubato accuracy. z-normalizing each curve independently
    # cancels that out and compares "where do you speed up/slow down", not
    # "what absolute tempo did you choose".
    ideal_tempo_z = _zscore(np.log(ideal_tempo_ratio))
    perf_tempo_z = _zscore(np.log(perf_tempo_ratio))
    tempo_diff = ideal_tempo_z - perf_tempo_z
    tempo_score = float(100 * np.exp(-np.median(tempo_diff ** 2) / 2))

    # dynamics: shape-match using onset strength (spectral-flux attack energy)
    # rather than raw RMS -- RMS also picks up sustain/pedal resonance from
    # earlier notes, which is not "how hard this note was struck". Onset
    # strength isolates the attack transient, a much closer analogue to
    # MIDI velocity. Still not absolute-scale comparable to velocity (mic
    # gain/distance is arbitrary), so this remains a shape-only comparison.
    perf_times = np.array([np.interp(t, alignment["score_times"], alignment["perf_times"]) for t in score_time])
    onset_env = compute_onset_strength(perf_audio, sr=SR, hop_length=HOP_LENGTH)
    perf_loudness = np.array([onset_strength_at(onset_env, SR, HOP_LENGTH, t) for t in perf_times])

    ideal_dyn_z = _zscore(ideal_features["velocity"])
    perf_dyn_z = _zscore(perf_loudness)
    dyn_diff = ideal_dyn_z - perf_dyn_z
    dynamics_score = float(100 * np.exp(-np.median(dyn_diff ** 2) / 2))

    # articulation: fraction of the gap to the next distinct onset that stays
    # audible. Notes sharing a score onset (chords) share the same "next
    # onset"; the last onset in the piece has no next onset and is excluded.
    unique_score_times = np.unique(score_time)
    next_onset_lookup = dict(zip(unique_score_times[:-1], unique_score_times[1:]))
    rms_env = compute_rms_envelope(perf_audio, sr=SR, hop_length=HOP_LENGTH)

    perf_sustain, ideal_artic_matched = [], []
    valid_idx = []
    for i, st in enumerate(score_time):
        next_st = next_onset_lookup.get(st)
        if next_st is None:
            continue
        next_perf_t = score_time_to_perf_time(alignment, next_st)
        frac = sustain_fraction(rms_env, SR, HOP_LENGTH, perf_times[i], next_perf_t)
        if frac is None:
            continue
        perf_sustain.append(frac)
        ideal_artic_matched.append(ideal_features["articulation"][i])
        valid_idx.append(i)

    perf_sustain = np.array(perf_sustain)
    ideal_artic_matched = np.array(ideal_artic_matched)

    if len(perf_sustain) > 10:
        ideal_artic_z = _zscore(ideal_artic_matched)
        perf_artic_z = _zscore(perf_sustain)
        artic_diff = ideal_artic_z - perf_artic_z
        articulation_score = float(100 * np.exp(-np.median(artic_diff ** 2) / 2))
    else:
        articulation_score = None
        artic_diff = None

    if articulation_score is not None:
        overall = float(0.45 * tempo_score + 0.30 * dynamics_score + 0.25 * articulation_score)
    else:
        overall = float(0.6 * tempo_score + 0.4 * dynamics_score)
    measure = np.array([n.measure_number for n in ideal_score.xml_notes])

    artic_measure = measure[valid_idx] if articulation_score is not None else np.array([])

    return {
        "overall_score": overall,
        "tempo_score": tempo_score,
        "dynamics_score": dynamics_score,
        "articulation_score": articulation_score,
        "num_notes": len(score_time),
        "num_articulation_notes": len(perf_sustain),
        "note_level": {
            "score_time": score_time,
            "perf_time": perf_times,
            "ideal_tempo_ratio": ideal_tempo_ratio,
            "perf_tempo_ratio": perf_tempo_ratio,
            "ideal_velocity": ideal_features["velocity"],
            "perf_loudness": perf_loudness,
            "measure": measure,
            "tempo_diff_z": tempo_diff,
            "dyn_diff_z": dyn_diff,
            "ideal_articulation": ideal_artic_matched,
            "perf_sustain_fraction": perf_sustain,
            "artic_measure": artic_measure,
            "artic_diff_z": artic_diff,
        },
        "feedback": generate_feedback(measure, tempo_diff, dyn_diff, artic_measure, artic_diff),
    }


def generate_feedback(measure, tempo_diff_z, dyn_diff_z, artic_measure=None, artic_diff_z=None,
                       top_n=10, min_severity=1.0):
    """Aggregate per-note shape-mismatch z-scores to measure level and turn
    the worst ones into plain-language feedback lines."""
    measures = np.unique(measure)
    reports = []
    for m in measures:
        mask = measure == m
        reports.append({
            "measure": int(m), "feature": "tempo",
            "mean_z": float(np.mean(tempo_diff_z[mask])),
        })
        reports.append({
            "measure": int(m), "feature": "dynamics",
            "mean_z": float(np.mean(dyn_diff_z[mask])),
        })
    if artic_measure is not None and artic_diff_z is not None and len(artic_measure) > 0:
        for m in np.unique(artic_measure):
            mask = artic_measure == m
            reports.append({
                "measure": int(m), "feature": "articulation",
                "mean_z": float(np.mean(artic_diff_z[mask])),
            })
    reports.sort(key=lambda r: -abs(r["mean_z"]))

    lines = []
    for r in reports:
        if abs(r["mean_z"]) < min_severity:
            break
        if r["feature"] == "tempo":
            phrase = "rushing ahead of" if r["mean_z"] > 0 else "lagging behind"
            detail = "the reference interpretation's pacing shape"
        elif r["feature"] == "dynamics":
            phrase = "playing louder than" if r["mean_z"] > 0 else "playing softer than"
            detail = "the reference interpretation's dynamic shaping"
        else:
            phrase = "more legato/sustained than" if r["mean_z"] > 0 else "more detached/staccato than"
            detail = "the reference interpretation's articulation"
        lines.append(f"Measure {r['measure']}: {phrase} {detail} (|z|={abs(r['mean_z']):.1f})")
        if len(lines) >= top_n:
            break
    return lines
