import streamlit as st
from pathlib import Path
import tempfile, os

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

st.set_page_config(page_title="RAG Document Assistant", page_icon="🤖", layout="wide")

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stChatMessage { border-radius: 10px; margin-bottom: 10px; }
.source-box { background: #f0f2f6; border-left: 3px solid #4CAF50;
              padding: 8px 12px; border-radius: 4px; font-size: 0.85em; }
</style>""", unsafe_allow_html=True)

# ─── En-tête ──────────────────────────────────────────────────────────────────
st.title("🤖 RAG Document Assistant")
st.caption("Uploadez un PDF et posez vos questions en langage naturel")

# ─── Vérification clé API ─────────────────────────────────────────────────────
api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input(
    "🔑 Clé OpenAI API", type="password", help="Ou définir dans un fichier .env")
if not api_key:
    st.info("👈 Renseignez votre clé OpenAI API dans la barre latérale pour commencer.")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# ─── Upload PDF ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    uploaded_file = st.file_uploader("📄 Choisir un PDF", type="pdf")
    chunk_size = st.slider("Taille des chunks", 500, 2000, 1000, 100)
    k_results = st.slider("Nb. de passages récupérés", 2, 8, 4)

@st.cache_resource(show_spinner="Indexation du document...")
def build_chain(file_bytes, filename, chunk_size, k):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory,
        return_source_documents=True, verbose=False
    )

    os.unlink(tmp_path)
    return chain, len(docs), len(chunks)

# ─── Interface chat ───────────────────────────────────────────────────────────
if uploaded_file:
    chain, nb_pages, nb_chunks = build_chain(
        uploaded_file.getvalue(), uploaded_file.name, chunk_size, k_results
    )

    with st.sidebar:
        st.success(f"✅ **{uploaded_file.name}**")
        st.info(f"📄 {nb_pages} pages · {nb_chunks} chunks")
        if st.button("🗑️ Effacer la conversation"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Document **{uploaded_file.name}** chargé et indexé ({nb_pages} pages). Posez-moi vos questions !"
        })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📌 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"<div class='source-box'>📄 Page {src['page']+1}<br>{src['text'][:200]}...</div>",
                                    unsafe_allow_html=True)

    if prompt := st.chat_input("Posez votre question sur le document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                result = chain({"question": prompt})
                answer = result["answer"]
                sources = [{"page": d.metadata.get("page", 0),
                            "text": d.page_content}
                           for d in result.get("source_documents", [])]

            st.markdown(answer)
            if sources:
                with st.expander("📌 Sources"):
                    for src in sources:
                        st.markdown(f"<div class='source-box'>📄 Page {src['page']+1}<br>{src['text'][:200]}...</div>",
                                    unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": sources
        })
else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        st.markdown("### 👆 Uploadez un PDF pour commencer")
        st.markdown("""
**Exemples de documents :**
- 📊 Rapport annuel d'entreprise
- 📜 Contrat ou document légal
- 📚 Article scientifique
- 📋 Compte-rendu de réunion
""")

st.sidebar.markdown("---")
st.sidebar.caption("🤖 Harry TEGUE — IBM GenAI Engineer")
