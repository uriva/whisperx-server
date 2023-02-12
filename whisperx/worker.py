import logging
from . import load_model
from .transcribe import work_on_file
import time
import torch


class Worker():
    _model = None
    _working_status = False

    def __init__(self, modelSize):
        self._model_size = modelSize

    def isBusy(self):
        return self._working_status

    def work(self, audio_path, output_dir):
        if self.working_status:
            logging.info("Already working! sorry..")
            return

        self.working_status = True
        model = self.model

        ######## REAL WORK IS BEING DONE HERE ###########
        start = time.time()
        work_on_file(model, audio_path, output_dir)
        end = time.time()
        logging.info(f"Finished processing {audio_path} within {round(end - start)} seconds (model: {self._model_size})")
        #################################################
        self.working_status = False
        return

    def _get_working_status(self):
        return self._working_status

    def _set_working_status(self, value):
        self._working_status = value

    working_status = property(
        fget=_get_working_status,
        fset=_set_working_status,
        doc="Whether the worker is currently working or not."
    )

    @property
    def model(self):
        if not self._model:
            logging.info(f'lazy loading Whisper model... (size: {self._model_size})')
            self._model = load_model(self._model_size, device='cuda')
            logging.info("Whisper model loaded successfully!")
        return self._model


format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO,
                    datefmt="%H:%M:%S")
