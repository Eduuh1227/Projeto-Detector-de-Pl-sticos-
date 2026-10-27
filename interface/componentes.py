"""Componentes Streamlit. Não contém lógica de treinamento ou inferência."""

import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from processamento.filtros import filtro_gaussiano, filtro_mediana
from processamento.preprocessamento import Imagem, carregar_imagem, converter_cores, redimensionar
from processamento.segmentacao import segmentar


def mostrar_bgr(imagem: Imagem, legenda: str) -> None:
    if os.getenv("APS_HOSPEDADO") == "1":
        # Só a prévia é reduzida. Inferência e exportação usam os arrays originais.
        previa = redimensionar(imagem, largura=1280, altura=1280)
        st.image(converter_cores(previa), caption=legenda, use_container_width=True,
                 output_format="JPEG")
    else:
        st.image(converter_cores(imagem), caption=legenda, use_container_width=True)


def entrada_imagem() -> tuple[Imagem, bytes, str] | None:
    """Valida upload e retorna imagem BGR, bytes originais e nome do arquivo."""
    st.header("Etapa 1 — Upload")
    limite_mb = min(20, int(st.get_option("server.maxUploadSize")))
    arquivo = st.file_uploader("Selecione uma fotografia de praia ou margem de rio",
                               type=["jpg", "jpeg", "png"])
    st.caption(f"JPG, JPEG ou PNG; até {limite_mb} MB e 24 megapixels. Transparência PNG é exibida sobre branco.")
    if arquivo is None:
        st.info("Envie uma fotografia para explorar as etapas de processamento.")
        return None
    try:
        dados = arquivo.getvalue()
        if len(dados) > limite_mb * 1024 * 1024:
            raise ValueError(f"Use uma imagem com até {limite_mb} MB nesta versão.")
        imagem = carregar_imagem(dados)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return None
    mostrar_bgr(imagem, "Imagem original")
    if os.getenv("APS_HOSPEDADO") == "1":
        st.caption("A prévia de exibição é limitada a 1280 pixels por lado; a imagem enviada é preservada para análise.")
    st.caption(f"Resolução: {imagem.shape[1]} × {imagem.shape[0]} pixels")
    return imagem, dados, arquivo.name


def mostrar_processamento(imagem: Imagem) -> tuple[dict[str, Imagem], dict]:
    """Compara filtros na mesma imagem; retorna imagens e parâmetros para exportação."""
    st.header("Etapa 2 — Processamento")
    with st.expander("Parâmetros de processamento", expanded=True):
        c1, c2, c3 = st.columns(3)
        kernel = c1.select_slider("Kernel dos filtros", options=[3, 5, 7, 9, 11], value=5)
        otsu = c2.checkbox("Limiar automático (Otsu)", value=True)
        limiar = c2.slider("Limiar manual", 0, 255, 127, disabled=otsu)
        morfologia = c3.selectbox("Operação morfológica", ["nenhuma", "abertura", "fechamento"])
        kernel_morf = c3.select_slider("Kernel morfológico", options=[3, 5, 7], value=3,
                                       disabled=morfologia == "nenhuma")
    # Limita apenas a comparação visual para manter a interface responsiva.
    base = redimensionar(imagem, largura=1600, altura=1600)
    gaussiano = filtro_gaussiano(base, kernel)
    mediana = filtro_mediana(base, kernel)
    mascara, limiar_usado = segmentar(base, limiar, otsu=otsu,
                                      morfologia=morfologia, kernel=kernel_morf)
    colunas = st.columns(3)
    for coluna, img, titulo in zip(colunas, (base, gaussiano, mediana),
                                  ("Imagem original", "Filtro Gaussiano", "Filtro de Mediana")):
        with coluna:
            mostrar_bgr(img, titulo)
    st.caption(f"Comparação limitada a 1600 pixels no maior lado, mantendo proporção. "
               f"Dimensões desta comparação: {base.shape[1]} × {base.shape[0]}. "
               "A detecção usa a imagem original, sem esses filtros.")
    st.subheader("Segmentação tradicional")
    st.warning("Esta máscara separa intensidades de luz; ela não identifica plástico. "
               "Areia, reflexos, vegetação e sombras podem produzir regiões semelhantes.")
    esquerda, direita = st.columns(2)
    esquerda.image(converter_cores(base, "GRAY"), caption="Escala de cinza",
                   use_container_width=True, clamp=True)
    direita.image(mascara, caption=f"Limiarização — limiar utilizado: {limiar_usado:g}",
                  use_container_width=True, clamp=True)
    if st.checkbox("Exibir histograma de intensidades"):
        fig, ax = plt.subplots(figsize=(8, 2.7))
        cinza = converter_cores(base, "GRAY")
        contagens = np.bincount(cinza.ravel(), minlength=256)
        ax.plot(np.arange(256), contagens, color="#226c63")
        ax.axvline(limiar_usado, color="#bd482d", label=f"Limiar: {limiar_usado:g}")
        ax.set(xlim=(0, 255), xlabel="Intensidade", ylabel="Número de pixels")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    imagens = {"original_comparacao": base, "gaussiano": gaussiano,
               "mediana": mediana, "segmentacao": mascara}
    parametros = {"kernel_filtros": kernel, "sigma_gaussiano": 0, "otsu": otsu,
                  "limiar_usado": limiar_usado, "morfologia": morfologia,
                  "kernel_morfologia": kernel_morf,
                  "resolucao_comparacao": [base.shape[1], base.shape[0]]}
    return imagens, parametros


def informacoes_academicas() -> None:
    st.header("Etapa 4 — Informações acadêmicas")
    if os.getenv("APS_HOSPEDADO") == "1":
        st.caption("Nesta versão online, a sessão é temporária. Baixe os resultados antes de "
                   "fechar ou recarregar a página.")
    st.write("O processamento digital prepara as imagens para análise, enquanto o modelo "
             "de visão computacional é responsável pela detecção dos resíduos plásticos.")
    st.write("O sistema identifica apenas resíduos visualmente perceptíveis em fotografias "
             "convencionais e não foi projetado para detecção de microplásticos.")
    st.caption("O desempenho depende dos dados anotados e precisa ser medido em imagens "
               "independentes. Confiança de uma detecção não representa acurácia do sistema.")
    with st.expander("Equipe e responsabilidades"):
        st.markdown("- **Eduardo:** processamento das imagens e integração.\n"
                    "- **Ryan:** dataset, configuração, treinamento e avaliação do YOLO.\n"
                    "- **Bárbara:** interface, testes de utilização e apresentação dos resultados.")
