
import gradio as gr
import os
import torch
from transformers import pipeline
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# =====================================================
# API Key Setup
# NOTE: Ensure the NVIDIA_API_KEY is set in your environment.
# =====================================================
# Global FAISS vectorstore and K-value
vectorstore = None
DEFAULT_K = 3

# Initialize NVIDIA LLM + Embeddings
try:
    # ------------------------------------------------------------------
    # CHANGE: Switched to Llama3 8B and the general nv-embed-v1 
    # (These are generally more stable and less quota-restricted for testing)
    # ------------------------------------------------------------------
    llm = ChatNVIDIA(model="meta/llama3-8b-instruct") 
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
    
    # Quick test to confirm initialization worked
    _ = llm.invoke([HumanMessage(content="Hello")]).content
    
    # If the key is truly valid and has access, this will set the clients.
    LLM_AVAILABLE = True
except Exception as e:
    # This block captures the 403 or 401 error
    print(f"FATAL ERROR: NVIDIA services failed to initialize: {e}")
    print("Please check your NVIDIA_API_KEY, its permissions, and cloud credits.")
    llm = None
    embeddings = None
    LLM_AVAILABLE = False
    
# =====================================================
# Speech-to-Text (STT) Functionality
# =====================================================
try:
    stt_pipe = pipeline(
        "automatic-speech-recognition", 
        model="facebook/wav2vec2-base-960h", 
        tokenizer="facebook/wav2vec2-base-960h", 
        device=0 if torch.cuda.is_available() else -1
    )
    STT_AVAILABLE = True
except Exception as e:
    print(f"Warning: STT model failed to load. Voice input will not work. Error: {e}")
    STT_AVAILABLE = False
    stt_pipe = None

def transcribe_audio(audio_file_tuple):
    """Transcribes the audio file from the Gradio input."""
    if not STT_AVAILABLE or audio_file_tuple is None:
        return "⚠ Speech-to-Text service is unavailable or no audio was recorded.", ""
    
    audio_path = audio_file_tuple[1] if isinstance(audio_file_tuple, tuple) and len(audio_file_tuple) == 2 else audio_file_tuple
        
    try:
        result = stt_pipe(audio_path)
        transcribed_text = result['text']
        return "✅ Audio transcribed successfully!", transcribed_text
    except Exception as e:
        return f"❌ Error during transcription: {str(e)}", ""

# =====================================================
# Build Knowledge Base 
# =====================================================
def build_knowledge_base(pdf_file):
    """Loads PDF and builds the FAISS index."""
    global vectorstore
    
    vectorstore = None 
    
    if not LLM_AVAILABLE:
        return "❌ RAG service is offline due to NVIDIA API error.", DEFAULT_K

    if not pdf_file:
        return "⚠ Please upload a hospital knowledge base PDF to begin.", DEFAULT_K
    
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        if not documents:
            return "⚠ PDF appears empty or couldn't be read.", DEFAULT_K

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # Create FAISS store
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return "✅ Hospital knowledge base built successfully! You can now ask questions.", DEFAULT_K
        
    except Exception as e:
        return f"❌ Error building knowledge base: {str(e)}", DEFAULT_K

# --------------------
# Search Knowledge Base 
# --------------------
def search_docs(query, k_value):
    global vectorstore
    if vectorstore is None:
        return "⚠ Please upload and build the knowledge base first.", ""
    # ... (rest of search_docs remains the same)
    try:
        docs = vectorstore.similarity_search(query, k=k_value)
        
        if not docs:
            return "⚠ No relevant documents found.", ""
            
        formatted_context = []
        for i, d in enumerate(docs):
            source = d.metadata.get('source', 'Unknown Source')
            page = d.metadata.get('page', 'Unknown Page')
            formatted_context.append(f"--- [Source {i+1}: File '{os.path.basename(source)}', Page {page}] ---\n{d.page_content.strip()}")
            
        context_string = "\n\n" + "\n\n".join(formatted_context)
        
        return context_string, context_string
    except Exception as e:
        return f"❌ Error during search: {str(e)}", ""

# --------------------
# Chatbot Response 
# --------------------
def chatbot_response(user_message, history, k_value):
    if not LLM_AVAILABLE:
        yield "❌ RAG service is offline due to NVIDIA API error. Cannot process request."
        return

    query = user_message 
    
    status_msg, context = search_docs(query, k_value)
    
    if status_msg.startswith("⚠") or status_msg.startswith("❌"):
        yield status_msg
        return

    system_prompt = (
        "You are a helpful and concise Hospital Helpdesk Assistant. "
        "Use ONLY the provided context to answer the user's question. "
        "For every piece of factual information, you MUST cite the source information in brackets (e.g., [Source 1] or [Source 2]). "
        "If the context does not contain the answer, state 'I cannot answer this question based on the provided hospital knowledge base.' "
        "Do not invent information."
    )
    
    rag_query = f"Context:\n{context}\n\nUser Question:\n{query}\n\nAnswer in a clear, patient-friendly way, including citations:"
    
    messages = [SystemMessage(content=system_prompt)]
    
    for human_msg, ai_msg in history:
        messages.append(HumanMessage(content=human_msg))
        messages.append(AIMessage(content=ai_msg))

    messages.append(HumanMessage(content=rag_query))

    try:
        full_response = ""
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content') and isinstance(chunk.content, str):
                full_response += chunk.content
                yield full_response
            elif hasattr(chunk, 'content') and isinstance(chunk.content, list) and len(chunk.content) > 0 and isinstance(chunk.content[0], str):
                 full_response += chunk.content[0]
                 yield full_response

    except Exception as e:
        yield f"❌ Error from NVIDIA API during generation: {str(e)}"

# --------------------
# Clear outputs
# --------------------
def clear_outputs():
    return "", "", [], DEFAULT_K 

# --------------------
# Gradio UI
# --------------------
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("## 🏥 Smart Hospital Helpdesk (RAG + Streaming + Citations + Voice 🎙️)")
    gr.Markdown("Drop your hospital policy/FAQ PDF and click 'Build' to activate the RAG chat.")

    k_value_state = gr.State(DEFAULT_K)
    
    with gr.Row():
        pdf_input = gr.File(label="📂 1. Upload Hospital Knowledge Base (PDF)", type="filepath")
        
    with gr.Row():
        # Build button is back to trigger the process explicitly after model initialization
        build_btn = gr.Button("📑 Build Knowledge Base") 
        k_slider = gr.Slider(minimum=1, maximum=10, step=1, value=DEFAULT_K, label="K (No. of Docs to Retrieve for RAG)", interactive=True)

    status = gr.Textbox(label="📢 Status", interactive=False, lines=1)

    with gr.Tabs():
        with gr.TabItem("🤖 Chat Assistant (Streamed & Cited)"):
            chatbot = gr.Chatbot(height=350, label="Hospital Helpdesk Assistant")
            
            with gr.Row():
                text_input = gr.Textbox(
                    label="💬 Ask a Question (or confirm transcription below)", 
                    interactive=True, 
                    scale=4
                )
                chat_btn = gr.Button("Send (Enter)", scale=1)
                
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"], 
                    type="filepath", 
                    label="🎙️ Record Question",
                    scale=4
                )
                transcribe_btn = gr.Button("➡️ Transcribe Audio", scale=1)
                
            clear_btn = gr.Button("🔄 Clear Chat")

        with gr.TabItem("🔍 Raw Retrieval Output"):
            search_output = gr.Textbox(label="📂 Retrieved Knowledge (Context provided to LLM)", lines=12, interactive=False)

    # ----------------
    # Define Logic flow
    # ----------------
    
    # 1. Build Knowledge Base
    build_btn.click(
        build_knowledge_base, 
        inputs=[pdf_input], 
        outputs=[status, k_value_state]
    )
    
    # 2. Transcription Logic
    transcribe_btn.click(
        transcribe_audio,
        inputs=[audio_input],
        outputs=[status, text_input]
    )
    
    # 3. Chat Logic
    def chat_wrapper(message, history, k):
        response_generator = chatbot_response(message, history, k)
        history.append([message, ""])
        
        status_msg, context = search_docs(message, k)
        search_output_value = context if not status_msg.startswith("⚠") and not status_msg.startswith("❌") else status_msg

        for chunk in response_generator:
            history[-1][1] = chunk
            yield history, search_output_value
            
    chat_btn.click(
        chat_wrapper, 
        inputs=[text_input, chatbot, k_value_state], 
        outputs=[chatbot, search_output]
    ).success(lambda: gr.update(value=''), None, [text_input])
    
    text_input.submit(
        chat_wrapper, 
        inputs=[text_input, chatbot, k_value_state], 
        outputs=[chatbot, search_output]
    ).success(lambda: gr.update(value=''), None, [text_input])

    # 4. Clear Outputs
    clear_btn.click(
        clear_outputs, 
        outputs=[status, text_input, chatbot, k_value_state]
    )


# Launch
demo.launch(share=True)
