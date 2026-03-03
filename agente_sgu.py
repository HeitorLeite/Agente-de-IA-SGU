from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import os

# ==========================================
# CLASSES DE COMPATIBILIDADE (FIX DO LANGCHAIN)
# ==========================================
# O `langchain` 1.x removeu o antigo pacote `langchain.chains`; o código de
# exemplo que originalmente importava `create_retrieval_chain` e
# `create_stuff_documents_chain` só funciona com as versões 0.x. Em vez disso,
# implementamos substitutos mínimos aqui para que o script possa rodar com a
# instalação atual.

def create_stuff_documents_chain(llm, prompt):
    """Retorna um pequeno empacotador (wrapper) que formata o prompt e chama o LLM.

    O assistente original produzia um objeto ``chain`` com um método ``invoke``
    que retornava ``{"output_text": resposta}``. Nós replicamos esse comportamento
    para que o resto do script permaneça inalterado.
    """
    class StuffChain:
        def __init__(self, llm, prompt):
            self.llm = llm
            self.prompt = prompt

        def invoke(self, inputs: dict):
            # Espera-se que ``inputs`` contenha pelo menos as chaves ``input`` e ``context``
            prompt_value = self.prompt.invoke(inputs)
            output = self.llm.invoke(prompt_value)
            return {"output_text": output}

    return StuffChain(llm, prompt)


def create_retrieval_chain(retriever, question_answer_chain):
    """Implementação muito simples de RAG que espelha o assistente antigo.

    O objeto retornado possui um método ``invoke`` que aceita ``{"input": ...}``
    e retorna ``{"answer": <texto>}``.
    """
    class RetrievalChain:
        def __init__(self, retriever, qa_chain):
            self.retriever = retriever
            self.qa_chain = qa_chain

        def invoke(self, inputs: dict):
            question = inputs.get("input")
            # O atual `VectorStoreRetriever` não expõe mais o método
            # `get_relevant_documents`. Em vez disso, acessamos o
            # banco vetorial diretamente ou chamamos seu método de busca público.
            if hasattr(self.retriever, "vectorstore"):
                # A maioria dos bancos vetoriais implementa similarity_search
                docs = self.retriever.vectorstore.similarity_search(
                    question,
                    **getattr(self.retriever, "search_kwargs", {}),
                )
            else:
                # Alternativa (fallback) para o método privado
                try:
                    docs = self.retriever._get_relevant_documents(
                        question, run_manager=None 
                    )
                except Exception:
                    raise AttributeError(
                        "Não foi possível recuperar documentos do retriever; "
                        "inspecione a API para encontrar o método correto."
                    )

            # Junta o conteúdo dos documentos encontrados
            context = "\n\n".join(d.page_content for d in docs)
            result = self.qa_chain.invoke({"input": question, "context": context})
            
            # A cadeia (chain) interna retorna um dicionário com a chave "output_text";
            # repassamos isso para a chave externa "answer" (resposta).
            answer = result.get("output_text") if isinstance(result, dict) else result
            return {"answer": answer}

    return RetrievalChain(retriever, question_answer_chain)


# ==========================================
# INGESTÃO DO PDF
# ==========================================
print("Lendo os arquivos...")
pdf_path = "manuais/manual_sgu_hospital.pdf"  # Substitua pelo nome do seu PDF

if not os.path.isfile(pdf_path):
    raise FileNotFoundError(
        f"O arquivo {pdf_path!r} não foi encontrado. "
        "Coloque o PDF na mesma pasta do script ou ajuste o caminho."
    )

loader = PyMuPDFLoader(pdf_path)
documentos = loader.load()

# Quebra o texto em pedaços de 1000 caracteres, com 200 de sobreposição 
# (para não cortar uma frase no meio do raciocínio)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
pedacos = text_splitter.split_documents(documentos)


# ==========================================
# BANCO DE DADOS VETORIAL LOCAL
# ==========================================
print("Criando o banco de dados...")
# Salva os pedaços no Chroma usando o modelo nomic-embed-text
vectorstore = Chroma.from_documents(
    documents=pedacos, 
    embedding=OllamaEmbeddings(model="nomic-embed-text")
)
# Cria o "motor de busca" que vai puxar a informação
retriever = vectorstore.as_retriever()


# ==========================================
# CONFIGURANDO A IA E A REGRA DO JOGO
# ==========================================
# Conecta com o modelo que você baixou. O servidor Ollama precisa estar
# rodando em segundo plano com esse modelo específico.
try:
    llm = OllamaLLM(model="llama3.2")
except Exception as exc:  
    print("Erro ao inicializar o modelo de IA (LLM):", exc)
    raise

# Teste rápido para garantir que a IA está "acordada" e pronta para uso
try:
    _ = llm.invoke("Responda apenas 'ok' para confirmar que você está online.")
except Exception as exc:
    print("Falha ao comunicar com o modelo de IA:", exc)
    print("Verifique se o modelo está baixado e o Ollama está rodando no computador.")
    raise

# O System Prompt é a regra absoluta da IA. 
system_prompt = (
    "Você é um assistente especializado no sistema SGU. "
    "Use APENAS o contexto fornecido abaixo para responder à pergunta. "
    "Se a resposta não estiver no contexto, diga 'Eu não sei com base no manual fornecido'. "
    "Não invente informações em hipótese alguma."
    "\n\n"
    "Contexto encontrado nos manuais:\n"
    "{context}"
)

# Monta o template da conversa
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Junta a IA, o prompt e o buscador em uma única engrenagem
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# ==========================================
# A HORA DA VERDADE (CHAT CONTÍNUO)
# ==========================================
print("\n" + "="*50)
print("Agente SGU Online! Digite 'sair' para encerrar o chat.")
print("="*50)

while True:
    # O programa pausa aqui e espera você digitar sua pergunta
    pergunta = input("\nVocê: ")

    # Se a pessoa digitar 'sair', o 'break' quebra o loop e o programa acaba
    if pergunta.strip().lower() in ['sair', 'exit', 'quit']:
        print("Agente SGU: Encerrando os sistemas. Bom trabalho!")
        break
    
    # Se a pessoa só apertar Enter sem querer, o 'continue' volta pro começo do loop
    if not pergunta.strip():
        continue

    print("Agente SGU está analisando os manuais...")

    try:
        # Executa a busca nos PDFs e gera a resposta
        resposta = rag_chain.invoke({"input": pergunta})
        
        print(f"\nAgente SGU:\n{resposta['answer']}")
        
    except Exception as erro:
        print(f"\nOps! Deu um erro ao tentar processar sua pergunta: {erro}")