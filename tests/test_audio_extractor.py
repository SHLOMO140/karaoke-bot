from types import SimpleNamespace

from karaoke.audio_extractor import transcode_to_mp3
from karaoke.legacy_media import split_artist_and_title


def test_split_artist_and_title_splits_dash_separated_titles():
    artist, title = split_artist_and_title("אייל גולן - בית מזכוכית")

    assert artist == "אייל גולן"
    assert title == "בית מזכוכית"


def test_transcode_to_mp3_writes_title_and_artist_metadata(tmp_path, monkeypatch):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.mp3"
    input_path.write_bytes(b"wav")
    captured: dict[str, object] = {}

    def fake_run(cmd, capture_output=True, timeout=300):
        captured["cmd"] = cmd
        output_path.write_bytes(b"mp3")
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr("karaoke.audio_extractor.subprocess.run", fake_run)

    transcode_to_mp3(
        str(input_path),
        str(output_path),
        title="בית מזכוכית",
        artist="אייל גולן",
    )

    cmd = captured["cmd"]
    assert "-metadata" in cmd
    assert "title=בית מזכוכית" in cmd
    assert "artist=אייל גולן" in cmd
