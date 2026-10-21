# GenAI RAG Application — Cloud-ready edition

A Streamlit Retrieval-Augmented Generation (RAG) app for asking questions about text-based PDFs. This edition is prepared for public deployment using **Streamlit Community Cloud** and the **Groq API** for answer generation. It keeps PyMuPDF extraction, page-aware chunking, `all-MiniLM-L6-v2` embeddings, ChromaDB retrieval, and source-page citations.

## Important deployment behavior

- The app runs on the hosting server, not on each visitor's computer.
- Answer generation uses your configured Groq API key. Never commit the key to GitHub.
- Visitors share the API quota associated with that key, so add limits/monitor usage before promoting the app widely.
- Uploaded PDFs are processed by the app server. Do not upload sensitive documents unless you are comfortable with the hosting and model providers' data-handling terms.
- The ChromaDB index is stored on the app's local filesystem. On free hosted instances, local files may be ephemeral and can be cleared after a restart/redeploy. Each session creates a separate collection.
- PDF upload limit is set to 20 MB. Scanned/image-only PDFs need OCR, which is not implemented.

## 1. Create a Groq API key

1. Sign in to the official [Groq Console](https://console.groq.com/).
2. Create an API key in the API Keys section.
3. Keep it private. Do not paste it into source code or commit it to GitHub.

The default model is `openai/gpt-oss-20b`; check the [current supported models](https://console.groq.com/docs/models) if that model is unavailable for your account.

## 2. Upload the project to GitHub

Create a new repository, for example `genai-rag-application`. Upload the contents of this folder to the repository root, so `app.py`, `rag_pipeline.py`, `requirements.txt`, and `utils/` are at the top level. Do not upload API keys, virtual environments, or the `RAG_Database/` folder.

## 3. Deploy on Streamlit Community Cloud

1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
2. Choose **Create app** and select your repository, branch (`main`), and main file path (`app.py`).
3. Open **Advanced settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "paste-your-real-key-here"
   ```

4. Select Python 3.12 if asked, then click **Deploy**.
5. Wait for the build to finish, open the assigned `*.streamlit.app` URL, upload a small text-based PDF, index it, and test a question.
6. Set the app to public in its sharing settings if you want anyone with the link to use it.

Official guides: [Deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) · [Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management) · [Share your app](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app)

## 4. Test locally first (macOS)

```bash
cd GenAI_RAG_Application_Cloud
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export GROQ_API_KEY="paste-your-real-key-here"
streamlit run app.py
```

## Architecture

```text
PDF upload → PyMuPDF extraction → page-aware chunks → Sentence Transformers embeddings
                                                       ↓
Question → query embedding → ChromaDB similarity search → retrieved passages
                                                       ↓
                                              Groq hosted LLM API
                                                       ↓
                                        answer with source page references
```

## Troubleshooting

- **Missing GROQ_API_KEY:** set the environment variable locally or add it under Streamlit Cloud Secrets.
- **Model not found / unavailable:** verify the model ID in the Groq model catalogue and use a model available to your account.
- **Slow first run:** the sentence-transformer embedding model must be downloaded the first time the app starts.
- **Out of memory / build failure:** hosted free-tier resources are limited; if the embedding library or ChromaDB cannot start, we may need a lighter embedding setup or a different host.
- **429 / rate limit:** the shared API key may have reached its quota. Review your Groq account's current limits and avoid sharing the public link broadly until usage controls are in place.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```
