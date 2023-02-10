from flask import Flask, request
from flask_restful import Resource, Api
import asyncio

from whisperx.transcribe import transcribe
import os.path

app = Flask(__name__)
api = Api(app)

@app.route('/transcribe/')
def hello():
    return 'Hello, World!'

class TranscribeHandler(Resource):
    def post(self):

        transcriptionRequest = request.json
        print("request is:", transcriptionRequest)
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
                'message': f"the file at path {pathToFile} was not found"
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


        task1 = asyncio.create_task(
            transcribe(modelSize, pathToFile, True))
        return {}, 200

    pass

api.add_resource(TranscribeHandler, '/transcribe')

