import streamlit as st
from pathlib import Path
from io import BytesIO
from docx import Document

# Initialize managers lazily
embed_manager = None
faiss_store = FaissStore(index_path=ARTIFACTS_DIR / "index.faiss", meta_path=ARTIFACTS_DIR / "metadata.npy")
translator = None

# UI
uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

col1, col2 = st.columns([2, 1])

with col1:
    if uploaded_pdf:
        st.info(f"Uploaded: {uploaded_pdf.name}")
        pages = extract_pages_from_pdf(uploaded_pdf)
        st.write(f"Detected {len(pages)} pages")

    if st.button("Build index (page-level)"):
        with st.spinner("Loading embedding model and encoding pages..."):
            embed_manager = EmbeddingManager(model_name=EMBED_MODEL)
            texts = [p['text'] for p in pages]
            embeddings = embed_manager.encode(texts)
            # metadata: list of dicts with filename and page_number
            metadata = [
                {"filename": Path(uploaded_pdf.name).name, "page_number": i+1, "text": pages[i]['text']}
                for i in range(len(pages))
            ]
            faiss_store.build_index(embeddings, metadata)
            st.success("Index built and saved to disk")
            st.session_state['indexed'] = True

    if st.session_state.get('indexed', False):
        topk = st.sidebar.slider("Top K", min_value=1, max_value=10, value=5)
        query = st.text_input("Enter query to retrieve relevant pages:")
        if st.button("Retrieve") and query.strip():
            # lazy load embedder if needed
            embed_manager = embed_manager or EmbeddingManager(model_name=EMBED_MODEL)
            q_emb = embed_manager.encode([query])
            results = faiss_store.search(q_emb, top_k=topk)
            st.write("### Retrieval results")
            for r in results:
                st.write(f"- Page {r['meta']['page_number']} (score: {r['score']:.4f})")
                st.write(r['meta']['text'][:400])

with col2:
    st.write("## Translate a specific page")
    page_num = st.number_input("Page number", min_value=1, value=1)
    if st.button("Translate page"):
        if not st.session_state.get('indexed', False):
            st.warning("Please build the index first")
        else:
            # fetch page text from metadata
            meta = faiss_store.get_metadata_by_page(Path(uploaded_pdf.name).name, page_num)
            if not meta:
                st.warning("Page not found in metadata")
            else:
                st.subheader("Original text")
                st.text_area("orig", meta['text'], height=300)

                translator = translator or GemmaTranslator(model_name=GEN_MODEL, fallback_model=FALLBACK_GEN_MODEL)
                translated = translator.translate(meta['text'])
                st.subheader("Translation")
                st.text_area("translated", translated, height=400)

                if translated:
                    # make docx
                    doc = Document()
                    doc.add_paragraph(translated)
                    buf = BytesIO()
                    doc.save(buf)
                    buf.seek(0)
                    st.download_button(
                        "Download .docx",
                        buf,
                        file_name=f"{Path(uploaded_pdf.name).stem}_page_{page_num}_fa.docx"
                    )
