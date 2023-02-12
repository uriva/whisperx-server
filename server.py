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

        pathToFile = transcriptionRequest.get('pathToFile')
        outputPath = transcriptionRequest.get('outputPath')
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

        if outputPath is None:
            outputPath = os.path.dirname(os.path.abspath(pathToFile))

        logging.info(f"request received [{pathToFile} -> {outputPath}]")

        if worker.isBusy():
            return {
                'message': f"a video is currently being processed, try again later"
            }, 529

        transcribe_thread = threading.Thread(target=worker.work, name="Transcriber Function", args=[pathToFile, outputPath])
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

