import fitz  # PyMuPDF
import ollama
import base64
import os

pdf_path = "manual_sgu_hospital.pdf" # Confirme se o nome está exato
pasta_imagens = "imagens_extraidas"

# Cria a pasta para salvar as imagens se ela não existir
if not os.path.exists(pasta_imagens):
    os.makedirs(pasta_imagens)

print("Abrindo o PDF...")
documento = fitz.open(pdf_path)

# Vamos olhar especificamente a página 4 (índice 3, pois começa no 0)
# que é onde está a tela de Login do SGU
pagina = documento[4] 
lista_imagens = pagina.get_images(full=True)

if not lista_imagens:
    print("Nenhuma imagem encontrada nesta página.")
else:
    print(f"Encontrei {len(lista_imagens)} imagem(ns) na página 5!")
    
    # Pega a primeira imagem da página
    img_info = lista_imagens[0]
    xref = img_info[0]
    imagem_extraida = documento.extract_image(xref)
    bytes_imagem = imagem_extraida["image"]
    
    # Salva a imagem no computador
    caminho_imagem = f"{pasta_imagens}/login_sgu.png"
    with open(caminho_imagem, "wb") as f:
        f.write(bytes_imagem)
    print(f"Imagem salva em: {caminho_imagem}")
    
    # ==========================================
    # LIGAÇÃO COM O LLAVA (VISÃO)
    # ==========================================
    print("\nPedindo para o LLaVA analisar a imagem (pode demorar um pouco na 1ª vez)...")
    
    # O Ollama precisa da imagem em formato Base64 para ler
    imagem_base64 = base64.b64encode(bytes_imagem).decode('utf-8')

    resposta = ollama.chat(
        model='llava',
        messages=[{
            'role': 'user',
            'content': 'Descreva detalhadamente o que você vê nesta captura de tela de um sistema de software. Quais botões e campos de texto existem nela?',
            'images': [imagem_base64]
        }]
    )

    print("\n👁️ Descrição do LLaVA:")
    print(resposta['message']['content'])

documento.close()