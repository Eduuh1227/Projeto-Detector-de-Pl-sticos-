"""Testes técnicos da conversão, sem fotografias nem treino simulados."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from treinamento.preparacao import categorias_selecionadas, converter_caixa
from treinamento.preparacao import extrair_fotos


class TestPreparacaoTaco(unittest.TestCase):
    def test_interseccao_antes_normalizar(self):
        caixa, ajustada = converter_caixa([10, -1, 20, 11], 100, 100)
        self.assertTrue(ajustada)
        self.assertEqual(caixa, [0.2, 0.05, 0.2, 0.1])

    def test_metal_excluido_e_plastico_raro_incluido(self):
        coco = {'categories':[{'id':27,'name':'Plastic lid','supercategory':'Lid'},
                              {'id':28,'name':'Metal lid','supercategory':'Lid'},
                              {'id':48,'name':'Plastic glooves','supercategory':'Plastic glooves'}]}
        self.assertEqual(set(categorias_selecionadas(coco)), {27,48})

    def test_caixa_fora_da_imagem_rejeitada(self):
        with self.assertRaises(ValueError):
            converter_caixa([200, 0, 10, 10], 100, 100)

    def test_zip_preserva_bytes_e_ignora_codigo_e_labels(self):
        # Marcador técnico de transporte; não é fotografia ou dado de treinamento.
        with TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            zip_path = raiz / 'entrada.zip'
            with ZipFile(zip_path, 'w') as pacote:
                pacote.writestr('dataset/images/train/taco_1.jpg', b'marcador-tecnico')
                pacote.writestr('dataset/labels/train/taco_1.txt', 'nao importar label antigo')
                pacote.writestr('executar.py', 'raise RuntimeError()')
            saida = extrair_fotos(zip_path, raiz / 'saida')
            self.assertEqual((saida / 'images/train/taco_1.jpg').read_bytes(), b'marcador-tecnico')
            self.assertFalse((saida / 'labels').exists())
            self.assertFalse((saida / 'executar.py').exists())

    def test_zip_com_caminho_externo_rejeitado_antes_de_escrever(self):
        with TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            zip_path = raiz / 'entrada.zip'
            with ZipFile(zip_path, 'w') as pacote:
                pacote.writestr('../fora.jpg', b'marcador-tecnico')
            with self.assertRaises(ValueError):
                extrair_fotos(zip_path, raiz / 'saida')
            self.assertFalse((raiz / 'saida').exists())
