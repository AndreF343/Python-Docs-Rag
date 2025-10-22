import os, requests, streamlit as st

API = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Python Docs RAG Demo", layout="centered")
st.title("Python Docs Q&A (RAG + re-rank)")

query = st.text_input("Ask a question about Python:", "How do I create a virtual environment?")
k_retrieve = st.slider("Candidates to retrieve", 5, 50, 20, 5)
k_final = st.slider("Final results", 1, 10, 5, 1)

if st.button("Search"):
    with st.spinner("Searching…"):
        try:
            r = requests.post(
                f"{API}/search",
                json={"query": query, "k_retrieve": int(k_retrieve), "k_final": int(k_final)},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                st.info("No results.")
            for i, res in enumerate(results, 1):
                title = res.get("title") or "(no title)"
                url = res.get("url") or ""
                score = res.get("score")
                snippet = (res.get("snippet") or "").strip()
                st.markdown(f"### {i}. {title} {'— {:.3f}'.format(score) if score is not None else ''}")
                if url:
                    # full URL in href as requested
                    st.markdown(f'<a href="{url}" target="_blank">{url}</a>', unsafe_allow_html=True)
                if snippet:
                    st.write(snippet)
                st.divider()
        except Exception as e:
            st.error(f"Request failed: {e}")

if st.button("Answer with citations"):
    with st.spinner("Thinking…"):
        r = requests.post(
            f"{API}/answer",
            json={"query": query, "k_retrieve": int(k_retrieve), "k_final": int(k_final)},
            timeout=120,
        )
        r.raise_for_status()
        data = r.json()
        st.subheader("Answer")
        if not data.get("answerable"):
            st.warning("Low confidence — here are the best sources:")
        st.write(data.get("answer") or "")
        st.subheader("Sources")
        for s in data.get("sources", []):
            url = s.get("url") or ""
            title = s.get("title") or url
            st.markdown(f'<a href="{url}" target="_blank">{title}</a>', unsafe_allow_html=True)
