import pytest
from yarp.exceptions.runtime import EmbeddingProviderNotFoundException
from yarp.runtime import preflight_checks


def test_is_package_installed_found(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    assert preflight_checks.is_package_installed("sentence_transformers") is True


def test_is_package_installed_not_found(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert preflight_checks.is_package_installed("sentence_transformers") is False


def test_check_embedding_provider_installed(monkeypatch):
    monkeypatch.setattr(preflight_checks, "is_package_installed", lambda name: True)
    # Should not raise
    preflight_checks.check_embedding_provider()


def test_check_embedding_provider_not_installed(monkeypatch):
    monkeypatch.setattr(preflight_checks, "is_package_installed", lambda name: False)
    with pytest.raises(EmbeddingProviderNotFoundException):
        preflight_checks.check_embedding_provider()


def test_check_required_packages_success(monkeypatch):
    monkeypatch.setattr(preflight_checks, "is_package_installed", lambda name: True)
    # Should not raise
    preflight_checks.check_required_packages()


def test_check_required_packages_failure(monkeypatch):
    monkeypatch.setattr(preflight_checks, "is_package_installed", lambda name: False)
    with pytest.raises(EmbeddingProviderNotFoundException):
        preflight_checks.check_required_packages()
