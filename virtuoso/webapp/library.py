"""Curated library of pieces the web app can score a performance against.

Each entry has been manually verified (full two-hand piano score, not a
melody-only reduction; sane measure/note counts for the piece) before being
added here -- see the project session notes on why automated score lookup
for arbitrary pieces isn't reliable enough to do unsupervised yet.
"""
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class LibraryPiece:
    id: str
    title: str
    composer: str
    xml_path: Path
    score_midi_path: Path


LIBRARY = [
    LibraryPiece(
        id="beethoven_sonata17_mvt1",
        title='Piano Sonata No. 17 "Tempest", Mvt. I',
        composer="Beethoven",
        xml_path=REPO_ROOT / "test_pieces/bps_17_1/musicxml_cleaned.musicxml",
        score_midi_path=REPO_ROOT / "test_pieces/bps_17_1/score.mid",
    ),
    LibraryPiece(
        id="beethoven_sonata32_mvt1",
        title="Piano Sonata No. 32, Op. 111, Mvt. I",
        composer="Beethoven",
        xml_path=REPO_ROOT / "virtuoso/pyScoreParser/test_examples/Beethoven/32-1/musicxml_cleaned.musicxml",
        score_midi_path=REPO_ROOT / "virtuoso/pyScoreParser/test_examples/Beethoven/32-1/midi_cleaned.mid",
    ),
    LibraryPiece(
        id="chopin_nocturne_48_1",
        title="Nocturne in C minor, Op. 48 No. 1",
        composer="Chopin",
        xml_path=REPO_ROOT / "test_pieces/chopin_noct_48_1/score.musicxml",
        score_midi_path=REPO_ROOT / "test_pieces/chopin_noct_48_1/score.mid",
    ),
    LibraryPiece(
        id="debussy_clair_de_lune",
        title="Clair de Lune (Suite bergamasque)",
        composer="Debussy",
        xml_path=REPO_ROOT / "test_pieces/debussy_clair_de_lune/score.musicxml",
        score_midi_path=REPO_ROOT / "test_pieces/debussy_clair_de_lune/score.mid",
    ),
    LibraryPiece(
        id="chopin_fantaisie_impromptu",
        title="Fantaisie-Impromptu, Op. 66",
        composer="Chopin",
        xml_path=REPO_ROOT / "test_pieces/chopin_fantaisie_impromptu/score.musicxml",
        score_midi_path=REPO_ROOT / "test_pieces/chopin_fantaisie_impromptu/score.mid",
    ),
    LibraryPiece(
        id="beethoven_moonlight_mvt3",
        title='Piano Sonata No. 14 "Moonlight", Mvt. III (Presto agitato)',
        composer="Beethoven",
        xml_path=REPO_ROOT / "test_pieces/beethoven_moonlight_mvt3/score.musicxml",
        score_midi_path=REPO_ROOT / "test_pieces/beethoven_moonlight_mvt3/score.mid",
    ),
    LibraryPiece(
        id="mozart_k331_mvt1",
        title="Piano Sonata No. 11, K. 331, Mvt. I (Andante grazioso)",
        composer="Mozart",
        xml_path=REPO_ROOT / "test_pieces/mozart_k331_mvt1/score.musicxml",
        score_midi_path=REPO_ROOT / "test_pieces/mozart_k331_mvt1/score.mid",
    ),
]

LIBRARY_BY_ID = {p.id: p for p in LIBRARY}
