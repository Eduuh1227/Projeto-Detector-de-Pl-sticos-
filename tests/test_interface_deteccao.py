"""Fluxo visual com doubles de contrato, sem pesos ou detecções científicas.

O entrypoint, o marcador .pt e qualquer saída permanecem em tmp_path. O detector
é substituído antes da execução: estes testes nunca carregam YOLO ou treinam.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from interface import deteccao as interface_deteccao
from modelo.detector import Deteccao, ErroModelo, ResultadoDeteccao


@pytest.fixture
def app_com_marcador(tmp_path, monkeypatch, png_bytes):
    origem = Path(__file__).resolve().parents[1] / "app.py"
    destino = tmp_path / "app.py"
    destino.write_text(origem.read_text(encoding="utf-8"), encoding="utf-8")
    modelos = tmp_path / "modelos"
    modelos.mkdir()
    (modelos / "marcador_teste.pt").write_bytes(b"nao-contem-pesos-reais")
    arquivo = SimpleNamespace(name="matriz_tecnica.png", getvalue=lambda: png_bytes)
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: arquivo)
    return destino


def botao_detectar(app):
    return next(botao for botao in app.button if botao.label == "Detectar resíduos")


class DetectorVisualDouble:
    """Fornece um contrato controlado em memória; não representa um modelo."""

    def __init__(self, caminho):
        assert caminho.name == "marcador_teste.pt"

    def inferir(self, imagem, **parametros):
        item = Deteccao(0, "plastico", 0.42, (2.0, 2.0, 12.0, 12.0))
        return ResultadoDeteccao(imagem.copy(), [item], {"origem": "double_unitario"})


def test_modelo_habilita_botao_e_nova_confianca_invalida_resultado(
        app_com_marcador, monkeypatch):
    monkeypatch.setattr(interface_deteccao, "DetectorPlastico", DetectorVisualDouble)
    app = AppTest.from_file(str(app_com_marcador), default_timeout=20).run()
    assert not app.exception
    assert not app.error
    assert botao_detectar(app).disabled is False
    assert not app.metric

    botao_detectar(app).click().run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "1"
    tabela = app.dataframe[0].value
    assert tabela.iloc[0]["Classe"] == "PLASTICO"
    assert tabela.iloc[0]["Confiança"] == "42.0%"

    confianca = next(slider for slider in app.slider if slider.label == "Confiança mínima")
    confianca.set_value(0.5).run()
    assert not app.exception
    assert not app.error
    assert not app.metric
    assert "resultado_deteccao" not in app.session_state
    assert not (app_com_marcador.parent / "resultados").exists()


def test_zero_deteccoes_exibe_limite_sem_afirmar_ausencia(app_com_marcador, monkeypatch):
    class DetectorVazioDouble(DetectorVisualDouble):
        def inferir(self, imagem, **parametros):
            return ResultadoDeteccao(imagem.copy(), [], {"origem": "double_unitario_vazio"})

    monkeypatch.setattr(interface_deteccao, "DetectorPlastico", DetectorVazioDouble)
    app = AppTest.from_file(str(app_com_marcador), default_timeout=20).run()
    botao_detectar(app).click().run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "0"
    assert any("Isso não comprova ausência de plástico" in item.value for item in app.info)
    assert not app.dataframe
    assert not (app_com_marcador.parent / "resultados").exists()


def test_falha_ao_carregar_pesos_remove_resultado_anterior(app_com_marcador, monkeypatch):
    monkeypatch.setattr(interface_deteccao, "DetectorPlastico", DetectorVisualDouble)
    app = AppTest.from_file(str(app_com_marcador), default_timeout=20).run()
    botao_detectar(app).click().run()
    assert app.metric[0].value == "1"

    def falhar(caminho):
        raise ErroModelo("Falha de pesos simulada no teste unitário.")

    monkeypatch.setattr(interface_deteccao, "DetectorPlastico", falhar)
    botao_detectar(app).click().run()
    assert not app.exception
    assert any("Falha de pesos simulada" in item.value for item in app.error)
    assert not app.metric
    assert "resultado_deteccao" not in app.session_state
    assert any("microplásticos" in item.value for item in app.markdown)


def test_hospedado_oferece_zip_sem_salvar_no_servidor(app_com_marcador, monkeypatch):
    monkeypatch.setenv("APS_HOSPEDADO", "1")
    monkeypatch.setattr(interface_deteccao, "DetectorPlastico", DetectorVisualDouble)

    def nao_salvar(*args, **kwargs):
        raise AssertionError("A versão hospedada não pode salvar resultados no disco.")

    monkeypatch.setattr(interface_deteccao, "salvar_resultado", nao_salvar)
    app = AppTest.from_file(str(app_com_marcador), default_timeout=20).run()
    botao_detectar(app).click().run()
    assert not app.exception
    assert not app.error
    assert not any(botao.label.startswith("Salvar imagens") for botao in app.button)
    downloads = [item.proto.label for item in app.get("download_button")]
    assert "Baixar todas as imagens e informações (ZIP)" in downloads
    assert "Baixar informações JSON" in downloads
    assert any("a sessão é temporária" in item.value for item in app.caption)
    assert not (app_com_marcador.parent / "resultados").exists()


def test_hospedado_bloqueia_download_grande_sem_alterar_resultado(app_com_marcador, monkeypatch):
    monkeypatch.setenv("APS_HOSPEDADO", "1")
    monkeypatch.setattr(interface_deteccao, "DetectorPlastico", DetectorVisualDouble)
    marcador_grande = b"x" * (4 * 1024 * 1024 + 1)
    # Marcadores simulam somente tamanho de transporte; nunca são oferecidos como arquivos válidos.
    monkeypatch.setattr(interface_deteccao, "empacotar_resultado", lambda *args: marcador_grande)
    monkeypatch.setattr(interface_deteccao, "codificar_png", lambda *args: marcador_grande)
    app = AppTest.from_file(str(app_com_marcador), default_timeout=20).run()
    botao_detectar(app).click().run()
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == "1"
    downloads = [item.proto.label for item in app.get("download_button")]
    assert downloads == ["Baixar informações JSON"]
    assert sum("ultrapassa o limite de 4 MB" in item.value for item in app.info) == 2
    assert not (app_com_marcador.parent / "resultados").exists()
