"""Resultados de doubles em tmp_path verificam persistência, não métricas/detecções reais."""

import hashlib
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from modelo import resultados as modulo
from modelo.detector import Deteccao, ResultadoDeteccao
from modelo.resultados import codificar_png, empacotar_resultado, montar_relatorio, salvar_resultado
from processamento.preprocessamento import carregar_imagem


def test_png_preserva_pixels(imagem_bgr):
    assert np.array_equal(carregar_imagem(codificar_png(imagem_bgr)), imagem_bgr)


def test_relatorio_preserva_proveniencia_do_double(imagem_bgr, png_bytes):
    resultado = ResultadoDeteccao(imagem_bgr, [Deteccao(0, "plastico", 0.63, (1, 2, 3, 4))],
                                  {"origem": "double de teste; não é resultado científico"})
    relatorio = montar_relatorio(resultado, png_bytes, "pasta/fonte.png", {"otsu": True})
    assert relatorio["arquivo_original"] == "fonte.png"
    assert relatorio["sha256_arquivo_original"] == hashlib.sha256(png_bytes).hexdigest()
    assert relatorio["quantidade"] == 1
    assert relatorio["deteccoes"][0]["confianca"] == 0.63
    assert relatorio["processamento_comparativo"] == {"otsu": True}


def test_salvamento_unico_zero_objetos_e_sem_sobrescrita(tmp_path, imagem_bgr, png_bytes):
    resultado = ResultadoDeteccao(imagem_bgr, [], {"origem": "double de teste"})
    argumentos = (resultado, {"gaussiano": imagem_bgr}, png_bytes, "teste.png", {"kernel_filtros": 3}, tmp_path)
    primeiro = salvar_resultado(*argumentos)
    conteudo = primeiro.read_bytes()
    segundo = salvar_resultado(*argumentos)
    assert primeiro != segundo
    assert primeiro.read_bytes() == conteudo
    for caminho in (primeiro, segundo):
        relatorio = json.loads(caminho.read_text(encoding="utf-8"))
        assert relatorio["quantidade"] == 0
        assert relatorio["deteccoes"] == []
        assert (caminho.parent / "original.png").read_bytes() == png_bytes
        assert (caminho.parent / "deteccoes.png").is_file()
        assert (caminho.parent / relatorio["imagens_processadas"][0]).resolve().is_file()


def test_rejeita_nome_de_imagem_fora_da_lista(tmp_path, imagem_bgr, png_bytes):
    resultado = ResultadoDeteccao(imagem_bgr, [], {})
    with pytest.raises(ValueError):
        salvar_resultado(resultado, {"../fora": imagem_bgr}, png_bytes, "teste.png", {}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_falha_no_salvamento_remove_apenas_arquivos_da_chamada(monkeypatch, tmp_path, imagem_bgr, png_bytes):
    anterior = tmp_path / "preservar.txt"
    anterior.write_text("resultado anterior", encoding="utf-8")
    resultado = ResultadoDeteccao(imagem_bgr, [], {})

    def falhar(_):
        raise OSError("falha simulada de codificação")

    monkeypatch.setattr(modulo, "codificar_png", falhar)
    with pytest.raises(OSError):
        salvar_resultado(resultado, {"mediana": imagem_bgr}, png_bytes, "teste.png", {}, tmp_path)
    assert anterior.read_text(encoding="utf-8") == "resultado anterior"
    assert not list(tmp_path.rglob("*.png"))
    assert not list(tmp_path.rglob("*.json"))


def test_zip_em_memoria_preserva_arquivos_e_referencias_sem_escrever_disco(
        monkeypatch, imagem_bgr, png_bytes):
    """Empacotamento de contrato com pixels técnicos, sem conteúdo científico."""
    resultado = ResultadoDeteccao(imagem_bgr, [], {"origem": "double_unitario"})

    def escrita_proibida(*args, **kwargs):
        raise AssertionError("O download hospedado não deve abrir arquivos no disco.")

    with monkeypatch.context() as contexto:
        contexto.setattr(Path, "open", escrita_proibida)
        contexto.setattr("builtins.open", escrita_proibida)
        dados = empacotar_resultado(resultado, {"gaussiano": imagem_bgr}, png_bytes,
                                   "../nome_original.png", {"otsu": True})
    with ZipFile(BytesIO(dados)) as pacote:
        assert pacote.testzip() is None
        assert set(pacote.namelist()) == {"original.png", "deteccoes.png", "deteccoes.json", "filtros/gaussiano.png"}
        assert pacote.read("original.png") == png_bytes
        assert np.array_equal(carregar_imagem(pacote.read("deteccoes.png")), imagem_bgr)
        relatorio = json.loads(pacote.read("deteccoes.json"))
        assert relatorio["quantidade"] == 0
        assert relatorio["arquivo_original"] == "nome_original.png"
        assert relatorio["sha256_arquivo_original"] == hashlib.sha256(png_bytes).hexdigest()
        assert relatorio["imagens_processadas"] == ["filtros/gaussiano.png"]
        assert all(caminho in pacote.namelist() for caminho in relatorio["imagens_processadas"])


@pytest.mark.parametrize("nome", ["../fora", "subpasta/imagem", ""])
def test_zip_rejeita_nomes_nao_permitidos(nome, imagem_bgr, png_bytes):
    resultado = ResultadoDeteccao(imagem_bgr, [], {})
    with pytest.raises(ValueError):
        empacotar_resultado(resultado, {nome: imagem_bgr}, png_bytes, "fixture.png", {})
