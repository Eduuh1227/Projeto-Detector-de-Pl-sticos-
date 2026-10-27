"""Execute com: streamlit run app.py."""

from pathlib import Path

try:
    import streamlit as st
except ImportError:
    raise SystemExit("Streamlit não está instalado. Execute: python -m pip install -r requirements.txt")

st.set_page_config(page_title="APS — Resíduos Plásticos", page_icon="♻", layout="wide")

try:
    from interface.componentes import entrada_imagem, informacoes_academicas, mostrar_processamento
except ImportError as exc:
    st.error(f"Biblioteca necessária ausente ({exc.name}). Execute: python -m pip install -r requirements.txt")
    st.stop()

RAIZ = Path(__file__).resolve().parent
MENSAGEM_SEM_MODELO = (
    "Modelo de detecção ainda não treinado. As funções de processamento de imagem estão "
    "disponíveis, porém a detecção de resíduos será habilitada após a inclusão do modelo treinado."
)


def etapa_deteccao(imagem, dados: bytes, nome: str, processadas: dict, parametros: dict) -> None:
    """A integração com o detector é isolada para não bloquear o processamento."""
    st.header("Etapa 3 — Detecção")
    pesos = sorted((RAIZ / "modelos").glob("*.pt"))
    pesos = [p for p in pesos if p.is_file()]
    st.button("Atualizar lista de modelos")
    if not pesos:
        st.info(MENSAGEM_SEM_MODELO)
        st.caption("Coloque os pesos personalizados em modelos/best.pt e atualize a lista.")
        return
    # Importação adiada: a aplicação sem modelo não exige carregar PyTorch/YOLO.
    from interface.deteccao import mostrar_deteccao
    mostrar_deteccao(imagem, dados, nome, processadas, parametros, pesos, RAIZ)


def main() -> None:
    st.title("Detecção de Resíduos Plásticos em Praias e Rios")
    entrada = entrada_imagem()
    if entrada is not None:
        imagem, dados, nome = entrada
        try:
            processadas, parametros = mostrar_processamento(imagem)
            etapa_deteccao(imagem, dados, nome, processadas, parametros)
        except ImportError as exc:
            st.error(f"Biblioteca ausente ({exc.name}). Instale as dependências de requirements.txt.")
        except (ValueError, OSError, RuntimeError) as exc:
            st.error(f"Não foi possível concluir a operação: {exc}")
    else:
        st.header("Etapa 2 — Detecção")
        if not list((RAIZ / "modelos").glob("*.pt")):
            st.info(MENSAGEM_SEM_MODELO)
    informacoes_academicas(etapa=4 if entrada is not None else 3)


if __name__ == "__main__":
    main()
