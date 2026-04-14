# api.py
from flask import Flask, request, jsonify
from transfer.classify_audio import FrogCallClassifier
import tempfile
import os

app = Flask(__name__)
classifier = FrogCallClassifier()

@app.route('/classify', methods=['POST'])
def classify():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        audio_file.save(tmp.name)
        
        # Classify
        pred_class, confidence, probs = classifier.classify(tmp.name, return_probs=True)
        
        # Clean up
        os.unlink(tmp.name)
    
    return jsonify({
        'species': pred_class,
        'confidence': confidence,
        'probabilities': {classifier.processor.CLASSES[i]: float(probs[i]) for i in range(len(probs))}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)