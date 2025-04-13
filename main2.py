import streamlit as st
from rag import generate_answer, process_urls

# Set page config
st.set_page_config(page_title="🏡 Real Estate Research Tool", page_icon="🏘️", layout="wide")

# Page Title with Emoji
st.title("URL Information Retrieval Tool")
st.markdown("Welcome to your intelligent assistant for researching real estate content online! 🔍💬")

# Sidebar for URLs
st.sidebar.header("🔗 Enter Webpage URLs")
url1 = st.sidebar.text_input('📎 URL 1')
url2 = st.sidebar.text_input('📎 URL 2')
url3 = st.sidebar.text_input('📎 URL 3')

# Placeholder for dynamic updates
placeholder = st.empty()

# Button to Process URLs
process_url_button = st.sidebar.button('🚀 Process URLs')

if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != '']
    if len(urls) == 0:
        placeholder.warning("⚠️ You must provide at least one valid URL.")
    else:
        with st.spinner("🔄 Processing URLs, please wait..."):
            for status in process_urls(urls):
                placeholder.info(f"✅ {status}")
        st.success("✅ All URLs processed successfully!")

# Space divider
st.markdown("---")

# Input for Query
st.subheader("❓ Ask a Question About the Content")
query = st.text_input("Type your question here... 🤔")

if query:
    with st.spinner("💬 Generating an intelligent response..."):
        try:
            answer, sources = generate_answer(query)

            st.success("🎯 Here's what I found:")
            st.markdown(f"**💡 Answer:**\n\n{answer}")

            if sources:
                st.subheader("📚 Sources Used:")
                for source in sources.split("\n"):
                    st.markdown(f"- 🌐 {source}")

        except RuntimeError as e:
            placeholder.error("⚠️ Please process the URLs first using the sidebar.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [Streamlit](https://streamlit.io/) | Powered by RAG 🧠")
