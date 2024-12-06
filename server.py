from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from main import MultiSourceChatbot

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize global variables
chatbot = None
uploaded_files = []
added_urls = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    
    files = request.files.getlist('files')
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    return jsonify({
        'status': 'success',
        'message': f'{len(files)} files uploaded successfully',
        'files': [os.path.basename(f) for f in uploaded_files]
    })

@app.route('/add_url', methods=['POST'])
def add_url():
    url = request.json.get('url')
    if not url:
        return jsonify({'status': 'error', 'message': 'No URL provided'}), 400
    
    added_urls.append(url)
    return jsonify({
        'status': 'success',
        'message': 'URL added successfully',
        'urls': added_urls
    })

@app.route('/init', methods=['POST'])
def init_chatbot():
    try:
        global chatbot
        chatbot = MultiSourceChatbot()
        
        # Prepare sources dictionary
        sources = {}
        
        # Add uploaded files to appropriate source types
        text_files = [f for f in uploaded_files if f.endswith('.txt')]
        pdf_files = [f for f in uploaded_files if f.endswith('.pdf')]
        
        if text_files:
            sources['text'] = text_files
        if pdf_files:
            sources['pdf'] = pdf_files
        if added_urls:
            sources['web'] = added_urls
            
        if not sources:
            return jsonify({
                'status': 'error',
                'message': 'No sources provided. Please upload files or add URLs first.'
            }), 400
            
        # Initialize chatbot with provided sources
        chatbot.load_documents(sources)
        chatbot.setup_chain()
        
        return jsonify({
            'status': 'success',
            'message': 'Chatbot initialized successfully',
            'sources': {
                'files': [os.path.basename(f) for f in uploaded_files],
                'urls': added_urls
            }
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    if not chatbot:
        return jsonify({'status': 'error', 'message': 'Chatbot not initialized'}), 400
    
    try:
        question = request.json.get('message')
        if not question:
            return jsonify({'status': 'error', 'message': 'No message provided'}), 400
        
        response = chatbot.chat(question)
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)