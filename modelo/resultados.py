"""Persistência de uma inferência executada. Não cria dados de demonstração."""

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import cv2

from processamento.preprocessamento import Imagem, validar_imagem
from .detector import ResultadoDeteccao


def codificar_png(imagem: Imagem) -> bytes:
    validar_imagem(imagem)
    sucesso, dados = cv2.imencode(".png", imagem)
    if not sucesso:
        raise OSError("Não foi possível codificar a imagem PNG.")
    return dados.tobytes()


def montar_relatorio(resultado: ResultadoDeteccao, arquivo_original: bytes,
                     nome_original: str, parametros: dict) -> dict:
    """Associa dados reais do detector à fonte e ao processamento visual aplicado."""
    return {"arquivo_original": Path(nome_original).name,
            "sha256_arquivo_original": hashlib.sha256(arquivo_original).hexdigest(),
            "quantidade": len(resultado.deteccoes),
            "deteccoes": [item.como_dict() for item in resultado.deteccoes],
            "inferencia": resultado.metadados,
            "processamento_comparativo": parametros,
            "observacao": "Confiança do modelo não é acurácia. Segmentação não identifica plástico."}


def _validar_processadas(processadas: dict[str, Imagem]) -> None:
    permitidas = {"original_comparacao", "gaussiano", "mediana", "segmentacao"}
    if not set(processadas).issubset(permitidas) or not processadas:
        raise ValueError("Informe as imagens produzidas pela etapa de processamento.")


def empacotar_resultado(resultado: ResultadoDeteccao, processadas: dict[str, Imagem],
                       arquivo_original: bytes, nome_original: str, parametros: dict) -> bytes:
    """Cria um ZIP em memória para download, sem gravar no disco do servidor.

    O JSON fica na raiz do pacote e seus caminhos de imagens são relativos a
    essa raiz. Os nomes de entrada são controlados, nunca derivados do upload.
    """
    _validar_processadas(processadas)
    extensao = ".png" if arquivo_original.startswith(b"\x89PNG") else ".jpg"
    relatorio = montar_relatorio(resultado, arquivo_original, nome_original, parametros)
    relatorio["imagens_processadas"] = [f"filtros/{nome}.png" for nome in processadas]
    dados_json = json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as pacote:
        pacote.writestr(f"original{extensao}", arquivo_original)
        pacote.writestr("deteccoes.png", codificar_png(resultado.imagem_bgr))
        for nome, imagem in processadas.items():
            pacote.writestr(f"filtros/{nome}.png", codificar_png(imagem))
        pacote.writestr("deteccoes.json", dados_json)
    return buffer.getvalue()


def salvar_resultado(resultado: ResultadoDeteccao, processadas: dict[str, Imagem],
                     arquivo_original: bytes, nome_original: str, parametros: dict,
                     diretorio: Path) -> Path:
    """Salva imagens e JSON sem sobrescrever; inclusive inferências reais sem objetos."""
    _validar_processadas(processadas)
    identificador = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:10]
    deteccoes_dir = diretorio / "deteccoes" / identificador
    filtros_dir = diretorio / "filtros" / identificador
    criados: list[Path] = []
    pastas: list[Path] = []

    def gravar(caminho: Path, dados: bytes) -> None:
        with caminho.open("xb") as arquivo:
            criados.append(caminho)
            arquivo.write(dados)

    try:
        for pasta in (deteccoes_dir, filtros_dir):
            pasta.mkdir(parents=True, exist_ok=False)
            pastas.append(pasta)
        extensao = ".png" if arquivo_original.startswith(b"\x89PNG") else ".jpg"
        gravar(deteccoes_dir / f"original{extensao}", arquivo_original)
        gravar(deteccoes_dir / "deteccoes.png", codificar_png(resultado.imagem_bgr))
        for nome, img in processadas.items():
            gravar(filtros_dir / f"{nome}.png", codificar_png(img))
        relatorio = montar_relatorio(resultado, arquivo_original, nome_original, parametros)
        relatorio["imagens_processadas"] = [f"../../filtros/{identificador}/{nome}.png" for nome in processadas]
        destino = deteccoes_dir / "deteccoes.json"
        gravar(destino, json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
        return destino
    except Exception:
        # Só remove arquivos criados por esta chamada, nunca resultados anteriores.
        for caminho in reversed(criados):
            caminho.unlink(missing_ok=True)
        for pasta in reversed(pastas):
            pasta.rmdir()
        raise
