"""Apresentação e estado de uma inferência, separados do adaptador YOLO."""

import hashlib
import json
import os
from pathlib import Path

import streamlit as st

from modelo.detector import DetectorPlastico, ErroModelo
from modelo.resultados import codificar_png, empacotar_resultado, montar_relatorio, salvar_resultado
from processamento.preprocessamento import Imagem
from .componentes import mostrar_bgr


def _oferecer_download(rotulo: str, conteudo: bytes | str, *, arquivo: str,
                      mime: str, hospedado: bool) -> None:
    """Evita anunciar downloads que ultrapassariam o limite da hospedagem."""
    tamanho = len(conteudo.encode("utf-8") if isinstance(conteudo, str) else conteudo)
    if hospedado and tamanho > 4 * 1024 * 1024:
        st.info(f"O arquivo {arquivo} ultrapassa o limite de 4 MB desta versão online. "
                "Baixe as outras opções disponíveis ou reduza a resolução da fotografia "
                "para gerar o pacote completo. A versão local mantém os arquivos sem esse limite.")
        return
    st.download_button(rotulo, conteudo, file_name=arquivo, mime=mime)


def mostrar_deteccao(imagem: Imagem, dados: bytes, nome: str,
                     processadas: dict, parametros: dict,
                     pesos: list[Path], raiz: Path) -> None:
    """Só mantém um resultado enquanto imagem, pesos e parâmetros forem os mesmos."""
    caminho = st.selectbox("Modelo personalizado", pesos, format_func=lambda p: p.name)
    st.caption("Os pesos devem ter sido treinados para a classe plastico. A presença do arquivo "
               "não comprova seu desempenho; isso exige avaliação independente.")
    c1, c2, c3 = st.columns(3)
    confianca = c1.slider("Confiança mínima", 0.05, 1.0, 0.25, 0.05)
    iou = c2.slider("IoU para pós-processamento do YOLO", 0.05, 1.0, 0.70, 0.05)
    imgsz = c3.selectbox("Resolução interna (imgsz)", [320, 640, 960, 1280], index=1)
    assinatura = (hashlib.sha256(dados).hexdigest(), str(caminho), caminho.stat().st_mtime_ns,
                  caminho.stat().st_size, confianca, iou, imgsz)
    if st.session_state.get("assinatura_deteccao") != assinatura:
        st.session_state.pop("resultado_deteccao", None)
        st.session_state.pop("arquivo_salvo", None)
        st.session_state["assinatura_deteccao"] = assinatura
    if st.button("Detectar resíduos", type="primary"):
        st.session_state.pop("resultado_deteccao", None)
        st.session_state.pop("arquivo_salvo", None)
        try:
            with st.spinner("Carregando pesos locais e executando a detecção..."):
                detector = DetectorPlastico(caminho)
                st.session_state["resultado_deteccao"] = detector.inferir(
                    imagem, confianca=confianca, iou=iou, imgsz=imgsz, dispositivo="cpu")
        except (ErroModelo, ValueError, OSError) as exc:
            st.error(str(exc))
    resultado = st.session_state.get("resultado_deteccao")
    if resultado is None:
        return
    mostrar_bgr(resultado.imagem_bgr, "Resultado da inferência na imagem original")
    st.metric("Resíduos detectados", len(resultado.deteccoes))
    if not resultado.deteccoes:
        st.info("Nenhum resíduo foi detectado com os parâmetros atuais. "
                "Isso não comprova ausência de plástico na fotografia.")
    else:
        st.dataframe([{"Detecção": numero, "Classe": item.classe.upper(),
                       "Confiança": f"{item.confianca:.1%}",
                       "Caixa (x1, y1, x2, y2)": [round(v, 2) for v in item.caixa_xyxy]}
                      for numero, item in enumerate(resultado.deteccoes, start=1)],
                     hide_index=True, use_container_width=True)
    st.caption("Confiança é o escore retornado pelo modelo para esta detecção; não é acurácia.")
    hospedado = os.getenv("APS_HOSPEDADO") == "1"
    if hospedado:
        try:
            pacote = empacotar_resultado(resultado, processadas, dados, nome, parametros)
            _oferecer_download("Baixar todas as imagens e informações (ZIP)", pacote,
                                arquivo="resultados_plasticos.zip", mime="application/zip",
                                hospedado=hospedado)
        except (OSError, ValueError) as exc:
            st.error(f"Não foi possível preparar o download dos resultados: {exc}")
    else:
        if st.button("Salvar imagens e informações em resultados/"):
            try:
                caminho_json = salvar_resultado(resultado, processadas, dados, nome, parametros, raiz / "resultados")
                st.session_state["arquivo_salvo"] = str(caminho_json.relative_to(raiz))
            except (OSError, ValueError) as exc:
                st.error(f"Não foi possível salvar os resultados: {exc}")
        if st.session_state.get("arquivo_salvo"):
            st.success(f"Resultado salvo em: {st.session_state['arquivo_salvo']}")
    relatorio = montar_relatorio(resultado, dados, nome, parametros)
    esquerda, direita = st.columns(2)
    with esquerda:
        _oferecer_download("Baixar imagem com detecções", codificar_png(resultado.imagem_bgr),
                            arquivo="deteccoes.png", mime="image/png", hospedado=hospedado)
    with direita:
        _oferecer_download("Baixar informações JSON",
                            json.dumps(relatorio, ensure_ascii=False, indent=2, allow_nan=False),
                            arquivo="deteccoes.json", mime="application/json", hospedado=hospedado)
