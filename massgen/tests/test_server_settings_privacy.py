from massgen.server.settings import ServerSettings


def test_server_settings_private_defaults(monkeypatch):
    monkeypatch.delenv("MASSGEN_SERVER_HOST", raising=False)
    monkeypatch.delenv("MASSGEN_PRIVATE_MODE", raising=False)
    monkeypatch.delenv("MASSGEN_ALLOW_REMOTE_ACCESS", raising=False)
    monkeypatch.delenv("MASSGEN_ALLOW_UNSAFE_SHARING", raising=False)
    monkeypatch.delenv("MASSGEN_SERVER_TOKEN", raising=False)

    settings = ServerSettings.from_env()

    assert settings.host == "127.0.0.1"
    assert settings.private_mode is True
    assert settings.allow_remote_access is False
    assert settings.allow_unsafe_sharing is False
    assert settings.server_token
