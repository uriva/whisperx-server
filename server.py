from flask import Flask, request
from flask_restful import Resource, Api

from whisperx.worker import Worker

import logging
import threading
import os.path

app = Flask(__name__)
api = Api(app)

@app.route('/transcribe/')


class TranscribeHandler(Resource):
    
    def post(self):
        global model
        transcriptionRequest = request.json

        audio_path = transcriptionRequest.get('audioPath')
        output_dir = transcriptionRequest.get('outputDir')
        task = transcriptionRequest.get('task')
        if audio_path is None:
            return {
                'message': "'audioPath' key is missing"
            }, 400        

        if str(audio_path) == False:
            return {
                'message': "'pathToFile' value is not a string"
            }, 400

        if os.path.isfile(audio_path) == False:
            return {
                'message': f"the file at path '{audio_path}' was not found"
            }, 404
        
        if task is not None and str(task) not in ['translate', 'transcribe']:
            return {
                'message': f"the value of task is not valid: '{audio_path}'. Must be one of [translate, transcribe]"
            }, 404

        if task is None:
            logging.info('"task" query parameter not provided - using the default \'transcribe\'')
            task = 'transcribe'

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(audio_path))

        logging.info(f"request received: {task} [{audio_path} -> {output_dir}]")

        if worker.isBusy():
            return {
                'message': f"a video is currently being processed, try again later"
            }, 529

        transcribe_thread = threading.Thread(target=worker.work, name="Transcriber Function", args=[audio_path, output_dir, task])
        transcribe_thread.start()

        return {}, 200

    pass

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

device = os.environ.get('WHISPER_DEVICE')
if device == None:
    device = 'cpu'

torch_threads = int(os.getenv('TORCH_THREADS', 1))
if torch_threads == None:
    torch_threads = 1

modelSize = os.environ.get('WHISPER_MODEL')
if modelSize == None:
    modelSize = 'large-v2'
if modelSize not in ['small', 'medium', 'large', 'large-v2']:
    logging.error("invalid WHISPER_MODEL value. Must be one of ['small', 'medium', 'large', 'large-v2']")
worker = Worker(modelSize, device, torch_threads)



api.add_resource(TranscribeHandler, '/transcribe')


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
