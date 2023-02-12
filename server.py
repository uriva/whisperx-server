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

        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(audio_path))

        logging.info(f"request received [{audio_path} -> {output_dir}]")

        if worker.isBusy():
            return {
                'message': f"a video is currently being processed, try again later"
            }, 529

        transcribe_thread = threading.Thread(target=worker.work, name="Transcriber Function", args=[audio_path, output_dir])
        transcribe_thread.start()

        return {}, 200

    pass

format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")

modelSize = os.environ.get('WHISPER_MODEL')
if modelSize == None:
    modelSize = 'large-v2'
if modelSize not in ['small', 'medium', 'large', 'large-v2']:
    logging.error("invalid WHISPER_MODEL value. Must be one of ['small', 'medium', 'large', 'large-v2']")
worker = Worker(modelSize)

api.add_resource(TranscribeHandler, '/transcribe')

