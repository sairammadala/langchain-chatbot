# main.py
import streamlit as st
import requests
import json

def post_data(api_endpoint, data):
    try:
        # Send JSON (sets the proper Content-Type header)
        response = requests.post(api_endpoint, json=data, timeout=30)
        if response.ok:
            return response.json()
        else:
            try:
                err = response.json()
            except Exception:
                err = response.text
            st.error(f"API error {response.status_code}: {err}")
            return None
    except Exception as e:
        st.error(f"Error occurred: {e}")
        return None

def chat_page():
    st.title("Langchain Chatbot Interface")
    api_endpoint = "http://localhost:9091/chat"

    param1 = st.text_input("Query")
    namespace_str = st.text_input("Namespace (Optional)", value="")  # use empty string
    model_selector = st.selectbox(
        "Chat Model",
        ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k",
         "gpt-4-1106-preview", "claude-3-sonnet-20240229", "claude-3-opus-20240229"]
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    source_documents = st.slider("Number of Source Documents", 1, 10, 5)

    if api_endpoint:
        if st.button("Send"):
            payload = {
                "query": str(param1),
                "chat_history": [],
                "model": model_selector,
                "temperature": float(temperature),
                "vector_fetch_k": int(source_documents),
                "namespace": (namespace_str if namespace_str.strip() else None),
            }

            data = post_data(api_endpoint, payload)

            if data and "response" in data:
                st.markdown(data["response"].get("answer", ""))
                st.header("Source Documents")
                for i, msg in enumerate(data["response"].get("source_documents", []), start=1):
                    source = (msg.get("metadata") or {}).get("source", "N/A")
                    page_content = msg.get("page_content", "")
                    st.markdown(
                        f"""### Document **[{i}]**  
__{source}__  

*Page Content:*  

```{page_content}```"""
                    )
            else:
                st.warning("No response payload received from the API.")

def ingest_page():
    st.title("Document Ingestion")
    FASTAPI_URL = "http://localhost:9091/ingest"  

    uploaded_files = st.file_uploader("Upload Document(s)", accept_multiple_files=True)
    namespace = st.text_input("Namespace (Optional)")

    if st.button("Ingest Documents"):
        if uploaded_files:
            try:
                files = [("files", file) for file in uploaded_files]
                payload = {"namespace": namespace}
                response = requests.post(FASTAPI_URL, files=files, data=payload)
                
                if response.status_code == 200:
                    st.success("Documents ingested successfully!")
                else:
                    st.error(f"Failed to ingest documents. Error: {response.text}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload at least one document.")

def main():
    st.sidebar.title("Navigation")
    tabs = ["Ingestion", "Chat"]
    selected_tab = st.sidebar.radio("Go to", tabs)

    if selected_tab == "Chat":
        chat_page()
    elif selected_tab == "Ingestion":
        ingest_page()

if __name__ == "__main__":
    main()
