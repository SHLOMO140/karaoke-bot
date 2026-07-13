"""Tests for the download-link registry and aiohttp app."""

from karaoke.file_server import LinkRegistry, make_app


def test_register_returns_token_and_resolves(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    reg = LinkRegistry(ttl_seconds=100, now=lambda: 0.0)
    token = reg.register(str(f))
    assert reg.resolve(token) == str(f)


def test_expired_token_resolves_none(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    clock = {"t": 0.0}
    reg = LinkRegistry(ttl_seconds=10, now=lambda: clock["t"])
    token = reg.register(str(f))
    clock["t"] = 999
    assert reg.resolve(token) is None


def test_unknown_token_resolves_none():
    reg = LinkRegistry(ttl_seconds=10, now=lambda: 0.0)
    assert reg.resolve("nope") is None


def test_sweep_removes_expired_file(tmp_path):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    clock = {"t": 0.0}
    reg = LinkRegistry(ttl_seconds=10, now=lambda: clock["t"])
    reg.register(str(f))
    clock["t"] = 999
    reg.sweep()
    assert not f.exists()


def test_make_app_has_routes():
    reg = LinkRegistry(ttl_seconds=10)
    app = make_app(reg)
    paths = {r.resource.canonical for r in app.router.routes()}
    assert "/" in paths
    assert "/d/{token}" in paths
