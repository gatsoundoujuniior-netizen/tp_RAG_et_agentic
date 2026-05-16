import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv(override=True)

CV_FILE = "./cv_junior_eria (1).pdf"

prompt_template = """
Réponds à la question suivante en te basant UNIQUEMENT sur le contexte fourni.
Le contexte provient d'un CV professionnel.
Si la réponse n'est pas dans le contexte, réponds : JE NE SAIS PAS

<context>
    {context}
</context>
<question>
    {question}
</question>
"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CSS = """
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Custom header banner ── */
.header-banner {
    background: linear-gradient(90deg, #6c63ff 0%, #3ecfcf 100%);
    border-radius: 16px;
    padding: 20px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 8px 32px rgba(108,99,255,0.35);
}
.header-banner h1 {
    color: #ffffff;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
}
.header-banner p {
    color: rgba(255,255,255,0.8);
    font-size: 0.85rem;
    margin: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3f 0%, #16213e 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.2);
}
[data-testid="stSidebar"] * {
    color: #e0e0f0 !important;
}

/* ── Sidebar cards ── */
.sidebar-card {
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.sidebar-card-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6c63ff !important;
    margin-bottom: 8px;
}
.sidebar-card-body {
    font-size: 0.85rem;
    color: #c0c0e0 !important;
    line-height: 1.6;
}

/* ── Pipeline badges ── */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 6px;
}
.badge {
    background: rgba(62,207,207,0.15);
    border: 1px solid rgba(62,207,207,0.35);
    color: #3ecfcf !important;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-weight: 500;
}

/* ── Suggestion chips ── */
.chip-grid {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 6px;
}
.chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.78rem;
    color: #a0a0c0 !important;
    cursor: default;
    transition: all 0.2s;
}
.chip:hover {
    background: rgba(108,99,255,0.15);
    border-color: rgba(108,99,255,0.4);
    color: #e0e0ff !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 14px;
    padding: 4px 8px;
    margin-bottom: 8px;
}

/* User bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: rgba(108,99,255,0.12);
    border: 1px solid rgba(108,99,255,0.22);
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: rgba(30,30,60,0.7);
    border: 1px solid rgba(62,207,207,0.15);
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(108,99,255,0.35) !important;
    border-radius: 12px !important;
    color: #e0e0f0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(108,99,255,0.7) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
}

/* ── Markdown text in chat ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #d0d0e8;
    line-height: 1.7;
}

/* ── Section title ── */
.section-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #6c63ff;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(108,99,255,0.25);
}

/* ── Welcome card ── */
.welcome-card {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin: 40px auto;
    max-width: 560px;
}
.welcome-card h3 {
    color: #a0a0ff;
    margin-bottom: 8px;
}
.welcome-card p {
    color: #808090;
    font-size: 0.9rem;
}
</style>
"""


@st.cache_resource(show_spinner="Indexation du CV en cours...")
def load_cv():
    loader = PyPDFLoader(CV_FILE)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="o200k_base",
        chunk_size=300,
        chunk_overlap=20,
    )
    chunks = loader.load_and_split(splitter)
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(
        chunks,
        embedding_model,
        collection_name="cv_junior",
        persist_directory="./store",
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )
    return retriever, len(chunks)


def RAG(question, retriever):
    context_docs = retriever.invoke(question)
    context_list = [d.page_content for d in context_docs]
    context_text = ". ".join(context_list)
    prompt = prompt_template.format(context=context_text, question=question)
    resp = llm.invoke(prompt)
    return resp.content


def main():
    st.set_page_config(
        page_title="CV Assistant — RAG",
        page_icon="🤖",
        layout="wide",
    )

    st.markdown(CSS, unsafe_allow_html=True)

    retriever, nb_chunks = load_cv()

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 16px 0 24px 0;">
            <div style="font-size:2.5rem;">🤖</div>
            <div style="font-weight:700; font-size:1.1rem; color:#a0a0ff;">CV Assistant</div>
            <div style="font-size:0.75rem; color:#606080; margin-top:4px;">Powered by RAG</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">📄 Document indexé</div>
            <div class="sidebar-card-body">
                <strong>Gatsoundou Junior Stevy</strong><br>
                Ingénieur d'État IA & Big Data
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-card-title">⚙️ Pipeline RAG</div>
            <div class="badge-row">
                <span class="badge">📦 {nb_chunks} chunks</span>
                <span class="badge">ada-002</span>
                <span class="badge">ChromaDB</span>
                <span class="badge">gpt-4o-mini</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-card">
            <div class="sidebar-card-title">💡 Suggestions</div>
            <div class="chip-grid">
                <div class="chip">📌 Expériences professionnelles ?</div>
                <div class="chip">🎓 Formation académique ?</div>
                <div class="chip">🛠️ Compétences techniques ?</div>
                <div class="chip">🚀 Projets réalisés ?</div>
                <div class="chip">📜 Certifications ?</div>
                <div class="chip">🌍 Langues parlées ?</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Header ──
    st.markdown("""
    <div class="header-banner">
        <div>
            <h1>🤖 CV Assistant — RAG</h1>
            <p>Posez vos questions sur le CV · Retrieval Augmented Generation · LangChain + ChromaDB + OpenAI</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Chat area ──
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <h3>Bienvenue 👋</h3>
            <p>Je suis un assistant RAG capable de répondre à toutes vos questions sur le CV de <strong>Gatsoundou Junior Stevy</strong>.<br><br>
            Commencez par poser une question ci-dessous.</p>
        </div>
        """, unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Posez votre question sur le CV..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans le CV..."):
                response = RAG(question, retriever)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
