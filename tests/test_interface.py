"""Teste de utilização via AppTest; uploads técnicos existem apenas em memória."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


@pytest.fixture
def app_temporario(tmp_path):
    """Copia o entrypoint para isolar modelos/resultados dos arquivos do usuário."""
    origem = Path(__file__).resolve().parents[1] / "app.py"
    destino = tmp_path / "app.py"
    destino.write_text(origem.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "modelos").mkdir()
    return destino


def test_app_abre_sem_upload_e_sem_modelo(app_temporario):
    app = AppTest.from_file(str(app_temporario), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Detecção de Resíduos Plásticos em Praias e Rios"
    assert any("Modelo de detecção ainda não treinado" in item.value for item in app.info)
    assert any("microplásticos" in item.value for item in app.markdown)
    assert not app.metric


def test_upload_invalido_mostra_erro_e_mantem_app(monkeypatch, app_temporario):
    arquivo = SimpleNamespace(name="texto.png", getvalue=lambda: b"conteudo invalido")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: arquivo)
    app = AppTest.from_file(str(app_temporario), default_timeout=20).run()
    assert not app.exception
    assert app.error
    assert any("não é uma imagem" in item.value for item in app.error)


def test_upload_respeita_limite_da_hospedagem(monkeypatch, app_temporario):
    original_get_option = st.get_option
    monkeypatch.setattr(st, "get_option", lambda chave: 4 if chave == "server.maxUploadSize"
                        else original_get_option(chave))
    arquivo = SimpleNamespace(name="marcador_tecnico.png",
                              getvalue=lambda: b"x" * (4 * 1024 * 1024 + 1))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: arquivo)
    app = AppTest.from_file(str(app_temporario), default_timeout=20).run()
    assert not app.exception
    assert any("até 4 MB nesta versão" in item.value for item in app.error)
    assert any("até 4 MB e 24 megapixels" in item.value for item in app.caption)


def test_fluxo_processamento_parametros_e_histograma(monkeypatch, app_temporario, png_bytes):
    arquivo = SimpleNamespace(name="matriz_tecnica.png", getvalue=lambda: png_bytes)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: arquivo)
    app = AppTest.from_file(str(app_temporario), default_timeout=20).run()
    assert not app.exception
    assert not app.error
    assert any("ela não identifica plástico" in item.value for item in app.warning)
    assert any("Modelo de detecção ainda não treinado" in item.value for item in app.info)
    assert not app.metric
    app.select_slider[0].set_value(7)
    app.checkbox[0].uncheck()
    app.selectbox[0].set_value("abertura")
    app.checkbox[1].check()
    app.run()
    assert not app.exception
    assert not app.error
    assert app.slider[0].disabled is False
    assert app.select_slider[1].disabled is False
    app.slider[0].set_value(150).run()
    assert not app.exception
    assert app.select_slider[0].value == 7
