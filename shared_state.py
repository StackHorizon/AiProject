# shared_state.py
import multiprocessing as mp


class SharedState:
    def __init__(self):
        self.manager = mp.Manager()
        self.current_model = self.manager.Value('s', "face_paint_512_v2")
        self.processing_resolution = self.manager.Value('i', 768)
        self.running = self.manager.Value('b', False)
        self.request_model_change = self.manager.Event()
        self.model_changed = self.manager.Event()

        # Utilizziamo oggetti primitivi condivisi che sono sicuri per il pickling
        self.available_models = ["face_paint_512_v1", "face_paint_512_v2", "celeba_distill", "paprika"]

        # Aggiungiamo code per la comunicazione tra processi
        self.command_queue = self.manager.Queue()
        self.status_queue = self.manager.Queue()

    def set_model(self, model_name):
        if model_name in self.available_models:
            self.current_model.value = model_name
            self.request_model_change.set()

    def set_resolution(self, resolution):
        self.processing_resolution.value = min(max(512, resolution), 1024)


def create_shared_state():
    return SharedState()
