from flask import Flask, request
import whisper
import whisperx
from flask_restful import Resource, Api
import torch
import threading

from whisperx.transcribe import transcribe
import os.path

app = Flask(__name__)
api = Api(app)
working = False

@app.route('/transcribe/')


class TranscribeHandler(Resource):
    
    def post(self):
        global model
        transcriptionRequest = request.json
        print("request is:", transcriptionRequest, working)
        pathToFile = transcriptionRequest.get('pathToFile')
        if pathToFile is None:
            return {
                'message': "'pathToFile' key is missing"
            }, 400        

        if str(pathToFile) == False:
            return {
                'message': "'pathToFile' value is not a string"
            }, 400

        if os.path.isfile(pathToFile) == False:
            return {
                'message': f"the file at path '{pathToFile}' was not found"
            }, 404

        outputPath = transcriptionRequest.get('outputPath')
        if outputPath is None:
            outputPath = os.path.dirname(os.path.abspath(pathToFile))

    
        modelSize = transcriptionRequest.get('modelSize')
        if modelSize is None:
            modelSize = 'large'
        elif (modelSize not in ['small', 'medium', 'large', 'large-v2']):
            return {
                'message': f"'modelSize' must be one of ['small', 'medium', 'large', 'large-v2']"
            }, 400

        if working:
            return {
                'message': f"a video is currently being processed, try again later"
            }, 529

        print('creating task for request')

        transcribe_thread = threading.Thread(target=workOnce, name="Transcriber Function", args=[pathToFile, outputPath, modelSize])
        transcribe_thread.start()

        return {}, 200

    pass

def workOnce(audio_file, outputPath, modelSize):
    global working
    working = True
    print('working')
    work(audio_file, outputPath, modelSize)
    print('done working')
    working = False

def work1(audio_file, outputPath, modelSize):
    transcribe(model, audio_file)

def work(audio_file, outputPath, modelSize):
    device = "cuda" 
    # transcribe with original whisper
    model = whisperx.load_model(modelSize, device)
    result = model.transcribe(audio_file)

    print(result["segments"]) # before alignment

    # load alignment model and metadata
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

    # align whisper output
    result_aligned = whisperx.align(result["segments"], model_a, metadata, audio_file, device)

    print("SEGMENTS", result_aligned["segments"], "\n\n") # after alignment
    print("WORD_SEGMENTS", result_aligned["word_segments"],"\n\n") # after alignment



print('\n\n\nloading Whisper model...')
model = whisper.load_model('medium').cuda()
print("model loaded. CUDA available:", torch.cuda.is_available())
api.add_resource(TranscribeHandler, '/transcribe')

