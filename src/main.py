import streamlit as st
from crewai import Crew, Process, Agent, Task
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import BaseCallbackHandler
from utils import text
import os
from dotenv import load_dotenv

load_dotenv()

# Obter a chave da API da OpenAI do arquivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")

# Inicializar o modelo OpenAI com a chave da API
llm = ChatOpenAI(openai_api_key=openai_api_key)

# Avatar para o agente
avators = {"RAG Agent": "https://cdn-icons-png.flaticon.com/512/320/320336.png"}

# Definir o handler customizado para o agente
class MyCustomHandler(BaseCallbackHandler):
    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        """Iniciar o processamento da cadeia."""
        st.session_state.messages.append({"role": "assistant", "content": inputs['input']})
        st.chat_message("assistant").write(inputs['input'])
    
    def on_chain_end(self, outputs: dict, **kwargs):
        """Finalizar o processamento da cadeia."""
        st.session_state.messages.append({"role": self.agent_name, "content": outputs['output']})
        st.chat_message(self.agent_name, avatar=avators[self.agent_name]).write(outputs['output'])

# Definir o agente que responderá com base em documentos
class RAGAgent(Agent):
    def __init__(self, chunks):
        # Evitar o erro de Pydantic ao definir atributos dinamicamente
        self.__dict__["qa_chain"] = None

        # Criar os embeddings e o vetor de busca
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Inicializar a cadeia de perguntas e respostas (RAG)
        self.__dict__["qa_chain"] = RetrievalQA.from_chain_type(
            llm=llm,  # Usar o modelo de linguagem que foi carregado anteriormente
            retriever=retriever
        )

    def respond(self, prompt: str) -> str:
        """Executar o agente com uma pergunta do usuário."""
        response = self.qa_chain({'query': prompt})
        if response and response['result']:
            return response['result']
        else:
            return "Não tenho acesso a essa informação."

# Definir o processo de CrewAI
def create_rag_process(prompt: str, chunks):
    agent = RAGAgent(chunks)  # Passando os chunks processados para o agente

    task = Task(
        description=f"Responda a pergunta: {prompt}",
        agent=agent,
        expected_output="Uma resposta detalhada com base nos documentos fornecidos"
    )

    # CrewAI com apenas um agente (RAGAgent)
    project_crew = Crew(
        tasks=[task], 
        agents=[agent],
        manager_llm=llm,
        process=Process.hierarchical  # Definindo processo hierárquico
    )

    final_response = project_crew.kickoff()
    return final_response


# Interface do Streamlit
st.title("💬 Zoppy-Mind - RAG Agent")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Pergunte-me algo relacionado ao documento."}]

# Carregar e processar o documento
pdf = ['C:/Users/Lenovo/Documents/zoppy/LangChain-RAG-Chatbot/docs/qg_zoppy.pdf']
all_files = text.process_files(pdf)
chunks = text.create_text_chunks(all_files)

# Exibir as mensagens anteriores
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Capturar a entrada do usuário
if prompt := st.chat_input("Digite sua pergunta aqui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Criar e executar o processo de RAG para responder
    result = create_rag_process(prompt, chunks)

    # Adicionar a resposta ao histórico de mensagens
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.chat_message("assistant").write(result)
