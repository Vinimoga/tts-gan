def arquivos_sao_identicos(caminho1, caminho2):
    try:
        with open(caminho1, 'r', encoding='utf-8') as f1, open(caminho2, 'r', encoding='utf-8') as f2:
            conteudo1 = f1.read()
            conteudo2 = f2.read()
            return conteudo1 == conteudo2
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return False

arquivo1 = 'log_pytorch.txt'
arquivo2 = 'log_lightning.txt'

if arquivos_sao_identicos(arquivo1, arquivo2):
    print("Os arquivos são idênticos.")
else:
    print("Os arquivos são diferentes.")

