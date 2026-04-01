import os
import uuid
import json
import time
import PyPDF2
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

uploaded_documents = {}
HISTORY_FILE = "chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading history: {e}")
            return {}
    return {}

def save_history():
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat_sessions, f, indent=4)

chat_sessions = load_history()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filename_lower = file.filename.lower()
    
    if file:
        doc_id = str(uuid.uuid4())
        
        if filename_lower.endswith('.pdf'):
            try:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
                
                uploaded_documents[doc_id] = {
                    "type": "pdf",
                    "filename": file.filename,
                    "content": text
                }
                return jsonify({"document_id": doc_id, "filename": file.filename, "type": "pdf"})
            except Exception as e:
                return jsonify({"error": "Failed to extract text from the PDF file."}), 500
                
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            mime_type = file.mimetype
            image_data = file.read()
            uploaded_documents[doc_id] = {
                "type": "image",
                "filename": file.filename,
                "mime_type": mime_type,
                "data": image_data
            }
            return jsonify({"document_id": doc_id, "filename": file.filename, "type": "image"})
            
        else:
            return jsonify({"error": "Unsupported file format. Please upload a PDF or an Image (PNG, JPG, WEBP)."}), 400

@app.route('/api/history', methods=['GET'])
def get_history():
    sessions_list = []
    for sid, data in chat_sessions.items():
        title = data.get("title", "New Chat")
        updated_at = data.get("updated_at", 0)
        sessions_list.append({"id": sid, "title": title, "updated_at": updated_at})
    
    sessions_list.sort(key=lambda x: x["updated_at"], reverse=True)
    return jsonify(sessions_list)

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    if session_id in chat_sessions:
        return jsonify(chat_sessions[session_id])
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/chat', methods=['POST'])
def chat():
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY is not configured on the server."}), 500

    data = request.json
    prompt = data.get('prompt')
    document_id = data.get('document_id')
    session_id = data.get('session_id')

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    if not session_id or session_id not in chat_sessions:
        session_id = str(uuid.uuid4())
        title = prompt[:30] + ("..." if len(prompt) > 30 else "")
        chat_sessions[session_id] = {
            "title": title,
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time()
        }

    session = chat_sessions[session_id]
    
    session["messages"].append({"role": "user", "text": prompt})
    session["updated_at"] = time.time()

    contents = []
    # Feed previous history
    for msg in session["messages"][:-1]:
        contents.append(
            types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["text"])])
        )

    # Prepare latest message parts with injected multimodality
    final_parts = []
    
    if document_id and document_id in uploaded_documents:
        doc_info = uploaded_documents[document_id]
        if doc_info["type"] == "pdf":
            context = f"--- Background Document ({doc_info['filename']}) ---\n{doc_info['content']}\n\n"
            final_parts.append(types.Part.from_text(text=f"Background Context:\n{context}\nUser Instruction:\n{prompt}"))
        elif doc_info["type"] == "image":
            # Pass image to vision engine
            final_parts.append(types.Part.from_text(text=prompt))
            final_parts.append(types.Part.from_data(data=doc_info["data"], mime_type=doc_info["mime_type"]))
    else:
        final_parts.append(types.Part.from_text(text=prompt))

    contents.append(
        types.Content(role="user", parts=final_parts)
    )

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=contents,
        )
        
        reply_text = response.text
        
        session["messages"].append({"role": "model", "text": reply_text})
        session["updated_at"] = time.time()
        
        save_history()

        return jsonify({"response": reply_text, "session_id": session_id})
    except Exception as e:
        session["messages"].pop()
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": f"Failed to generate content: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
