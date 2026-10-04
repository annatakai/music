"""Compare a user's MIDI performance against VirtuosoNet's predicted
composer-style interpretation of the same score, note-by-note.

MVP scope: direct MIDI input only (no audio/video transcription yet).
"""
from pathlib import Path
import numpy as np
import torch

from .inference import InferenceModel

# Pedal features are excluded: the current checkpoint's inverse-sigmoid
# decode for pedal_at_start/pedal_at_end/soft_pedal produces NaN for this
# checkpoint's output range (virtuoso/inference.py:503), and a MIDI-keyboard
# "user performance" has no pedal signal to compare against anyway.
COMPARISON_FEATURES = ["beat_tempo", "velocity", "onset_deviation", "articulation"]

FEATURE_LABELS = {
    "beat_tempo": "tempo",
    "velocity": "dynamics (loudness)",
    "onset_deviation": "note timing (rubato placement)",
    "articulation": "articulation (legato/staccato length)",
}


def _direction_phrase(feature, z_diff):
    if feature == "beat_tempo":
        return "rushing ahead of" if z_diff > 0 else "dragging behind"
    if feature == "velocity":
        return "playing louder than" if z_diff > 0 else "playing softer than"
    if feature == "onset_deviation":
        return "placing notes later than" if z_diff > 0 else "placing notes earlier than"
    if feature == "articulation":
        return "holding notes longer (more legato) than" if z_diff > 0 else "cutting notes shorter (more detached) than"
    return "deviating from"


class PerformanceComparator:
    def __init__(self, checkpoint_path, device="cpu"):
        self.model = InferenceModel(
            checkpoint_path=checkpoint_path,
            device=device,
            output_path="test_result",
            midi_decode_options={},
        )
        self.device = device

    def _ideal_features(self, xml_path, composer, qpm_primo=None):
        score, input_t, edges, note_locations = self.model.get_input_from_xml(
            xml_path, composer, qpm_primo
        )
        with torch.no_grad():
            outputs, _, _, _ = self.model.model(
                input_t, None, edges, note_locations, initial_z="zero"
            )
        outputs_scaled = self.model.scale_model_prediction_to_original(outputs)
        features = self.model.model_prediction_to_feature(outputs_scaled)
        return score, features, note_locations

    def _user_features(self, xml_path, perf_midi_path, composer):
        score, x, y, edges, note_locations = self.model.get_score_perf_pair_data(
            Path(xml_path), perf_midi_path, composer, save_data=True
        )
        outputs_scaled = self.model.scale_model_prediction_to_original(y)
        features = self.model.model_prediction_to_feature(outputs_scaled)
        return score, features, note_locations

    def compare(self, xml_path, composer, perf_midi_path, qpm_primo=None, top_n=8):
        xml_path = Path(xml_path)
        ideal_score, ideal_features, _ = self._ideal_features(xml_path, composer, qpm_primo)
        user_score, user_features, note_locations = self._user_features(
            xml_path, perf_midi_path, composer
        )

        stats = self.model.model.stats["stats"]
        measure_number = np.array([n.measure_number for n in user_score.xml_notes])

        per_note_z = {}
        for feat in COMPARISON_FEATURES:
            ideal_v = ideal_features[feat]
            user_v = user_features[feat]
            std = stats[feat]["stds"] or 1.0
            z = (user_v - ideal_v) / std
            per_note_z[feat] = z

        overall_abs_z = np.mean(
            [np.abs(per_note_z[f]) for f in COMPARISON_FEATURES], axis=0
        )
        overall_score = float(100 * np.exp(-np.mean(overall_abs_z**2) / 2))

        per_feature_score = {}
        for feat in COMPARISON_FEATURES:
            mean_abs_z = float(np.mean(np.abs(per_note_z[feat])))
            per_feature_score[feat] = {
                "label": FEATURE_LABELS[feat],
                "score": float(100 * np.exp(-mean_abs_z**2 / 2)),
                "mean_abs_z": mean_abs_z,
            }

        measures = np.unique(measure_number)
        measure_reports = []
        for feat in COMPARISON_FEATURES:
            z = per_note_z[feat]
            for m in measures:
                m_mask = measure_number == m
                m_mean_z = float(np.mean(z[m_mask]))
                measure_reports.append(
                    {
                        "measure": int(m),
                        "feature": feat,
                        "label": FEATURE_LABELS[feat],
                        "mean_z": m_mean_z,
                        "severity": abs(m_mean_z),
                    }
                )
        measure_reports.sort(key=lambda r: -r["severity"])

        feedback = []
        seen_measures = set()
        for report in measure_reports:
            if report["severity"] < 1.0:
                break
            key = (report["measure"], report["feature"])
            if key in seen_measures:
                continue
            seen_measures.add(key)
            phrase = _direction_phrase(report["feature"], report["mean_z"])
            feedback.append(
                f"Measure {report['measure']}: {phrase} the reference interpretation's "
                f"{report['label']} (|z|={report['severity']:.1f})"
            )
            if len(feedback) >= top_n:
                break

        return {
            "overall_score": overall_score,
            "per_feature_score": per_feature_score,
            "measure_reports": measure_reports[:top_n],
            "feedback": feedback,
        }
