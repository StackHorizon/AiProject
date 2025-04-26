# main.py
import cv2
import numpy as np
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import torch
import torchvision.transforms as transforms
from PIL import Image
import multiprocessing as mp
import sys
import os
import platform
from pathlib import Path
import subprocess


# Verifica dipendenze all'avvio
def check_dependencies():
    required_packages = ["onnxruntime", "numpy", "opencv-python", "torch", "customtkinter", "pillow"]
    missing = []

    for package in required_packages:
        try:
            __import__(package.split('-')[0])
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Installazione pacchetti mancanti: {', '.join(missing)}...")
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

        # Per macOS, aggiungi supporto CoreML
        if platform.system() == 'Darwin':
            try:
                import onnxruntime
                if 'CoreMLExecutionProvider' not in onnxruntime.get_available_providers():
                    print("Installazione supporto CoreML per ONNX...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime-coreml"])
            except:
                print("⚠️ Impossibile installare CoreML, si utilizzerà CPU")

        print("✅ Dipendenze installate")


# Avvia l'interfaccia in un processo separato
def start_gui_process(shared_state):
    from gui import start_gui
    gui_process = mp.Process(
        target=start_gui,
        args=(
            shared_state.available_models,
            shared_state.command_queue,
            shared_state.status_queue,
            shared_state.current_model.value,
            shared_state.processing_resolution.value
        )
    )
    gui_process.daemon = True
    gui_process.start()
    return gui_process


# Funzione ottimizzata per caricare modello ONNX
def load_optimized_model(model_name, device):
    try:
        import onnxruntime as ort
        print(f"Preparazione modello ottimizzato: {model_name}")

        # Crea cartella per i modelli ONNX
        os.makedirs("models", exist_ok=True)
        onnx_path = os.path.join("models", f"{model_name}.onnx")

        # Converti il modello PyTorch in ONNX se non esiste
        if not os.path.exists(onnx_path):
            print(f"Conversione modello {model_name} in ONNX...")

            # Carica modello PyTorch
            torch_model = torch.hub.load("bryandlee/animegan2-pytorch:main",
                                         "generator", pretrained=model_name)
            torch_model.eval()

            # Esporta in ONNX con supporto dimensioni dinamiche
            dummy_input = torch.randn(1, 3, 512, 512)
            torch.onnx.export(torch_model, dummy_input, onnx_path,
                              input_names=["input"], output_names=["output"],
                              dynamic_axes={"input": {2: "height", 3: "width"},
                                            "output": {2: "height", 3: "width"}})
            print(f"✅ Modello {model_name} convertito in ONNX")

            # Libera memoria
            del torch_model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Ottimizza il runtime in base alla piattaforma
        providers = []
        if device.type == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif platform.system() == 'Darwin':  # macOS
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Crea sessione ONNX ottimizzata
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        model = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        available_provider = model.get_providers()[0]
        print(f"✅ Modello ONNX caricato con provider: {available_provider}")
        return model

    except Exception as e:
        print(f"❌ Errore nell'ottimizzazione del modello: {e}")
        print("⚠️ Fallback al modello PyTorch")
        return load_torch_model(model_name, device)


# Fallback al caricamento PyTorch originale
def load_torch_model(model_name, device):
    try:
        print(f"Caricamento modello PyTorch: {model_name}")
        model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",
                               pretrained=model_name).to(device).eval()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            model = model.half()  # FP16 per accelerare su GPU
        print(f"✅ Modello PyTorch {model_name} caricato")

        # Ottimizzazione memoria
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        return model
    except Exception as e:
        print(f"❌ Errore durante il caricamento del modello: {e}")
        return None


def main(shared_state):
    # Inizializzazione delle variabili
    frame_queue = Queue(maxsize=4)  # Ridotto da 16 per diminuire latenza
    result_queue = Queue(maxsize=2)  # Ridotto da 8 per diminuire latenza
    running = True
    target_fps = 60
    model_lock = threading.Lock()
    process_frame_lock = threading.Lock()

    # Flag per il rendering anteprima
    import __main__
    setattr(__main__, 'show_preview', True)

    # Monitoraggio performance
    last_fps_check = time.time()
    frame_count = 0
    fps = 0
    processing_times = []
    dynamic_processing = True

    # Verifica GPU e configurazioni
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzando: {device}")

    # Ottimizzazioni per PyTorch
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch.set_num_threads(1)  # Riduce interferenza tra CPU e GPU
    else:
        torch.set_num_threads(4)  # Usa più thread per CPU

    # Setup webcam con risoluzione nativa
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Risoluzione Webcam: {native_width} x {native_height}")

    if not cap.isOpened():
        print("❌ Webcam non trovata.")
        return

    # Calcola aspect ratio
    aspect_ratio = native_width / native_height

    # Inizializzazione risoluzione di elaborazione
    processing_resolution = shared_state.processing_resolution.value
    if aspect_ratio > 1:  # Landscape
        proc_width = processing_resolution
        proc_height = int(processing_resolution / aspect_ratio)
    else:  # Portrait
        proc_height = processing_resolution
        proc_width = int(processing_resolution * aspect_ratio)

    # Caricamento ottimizzato del modello
    model = load_optimized_model(shared_state.current_model.value, device)

    # Funzione per elaborare un singolo frame
    def process_frame(frame):
        if model is None:
            return frame

        with model_lock:
            start_time = time.time()
            try:
                # Conversione diretta a RGB senza passare per PIL
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Ridimensionamento ottimizzato
                input_img = cv2.resize(rgb, (proc_width, proc_height),
                                       interpolation=cv2.INTER_LINEAR)

                # Normalizzazione e preparazione per modello
                input_data = input_img.astype(np.float32) / 255.0
                input_data = (input_data - 0.5) / 0.5
                input_data = np.transpose(input_data, (2, 0, 1))
                input_data = np.expand_dims(input_data, axis=0)

                # Esegui inferenza in base al tipo di modello
                import onnxruntime as ort
                if isinstance(model, ort.InferenceSession):
                    # ONNX Runtime (più veloce)
                    ort_inputs = {'input': input_data}
                    output_data = model.run(None, ort_inputs)[0][0]
                else:
                    # Fallback a PyTorch
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(input_data).to(device)
                        if device.type == 'cuda':
                            input_tensor = input_tensor.half()

                        output_tensor = model(input_tensor)
                        if device.type == 'cuda':
                            torch.cuda.synchronize()

                        output_data = output_tensor.cpu().numpy()[0]

                # Post-processing ottimizzato
                output_data = np.transpose(output_data, (1, 2, 0))
                output_data = ((output_data * 0.5) + 0.5) * 255.0
                output_data = output_data.clip(0, 255).astype(np.uint8)

                # Ridimensiona alla risoluzione originale
                anime_frame = cv2.resize(output_data, (native_width, native_height),
                                         interpolation=cv2.INTER_LINEAR)

                # Riconverti a BGR per OpenCV
                anime_frame = cv2.cvtColor(anime_frame, cv2.COLOR_RGB2BGR)

                # Traccia tempo elaborazione
                proc_time = time.time() - start_time
                processing_times.append(proc_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)

                return anime_frame

            except Exception as e:
                print(f"❌ Errore nell'elaborazione: {e}")
                return frame

    # Thread di monitoraggio per il cambio modello
    def model_monitor_thread():
        nonlocal model, proc_height, proc_width, processing_resolution, running

        while running:
            # Leggi i comandi dalla coda
            try:
                if not shared_state.command_queue.empty():
                    command, data = shared_state.command_queue.get_nowait()

                    if command == "exit":
                        print("Richiesta di chiusura ricevuta dalla GUI")
                        running = False

                    elif command == "change_model" and data in shared_state.available_models:
                        with model_lock:
                            print(f"⚙️ Cambio modello richiesto: {data}")
                            shared_state.current_model.value = data

                            new_model = load_optimized_model(data, device)
                            if new_model is not None:
                                model = new_model

                                # Notifica la GUI
                                shared_state.status_queue.put(("model_changed", data))

                    elif command == "change_resolution":
                        with model_lock:
                            resolution = max(512, min(1024, int(data)))
                            print(f"⚙️ Aggiornamento risoluzione a: {resolution}")

                            # Aggiorna valore nello stato condiviso
                            shared_state.processing_resolution.value = resolution
                            processing_resolution = resolution

                            # Ricalcola dimensioni
                            if aspect_ratio > 1:  # Landscape
                                proc_width = processing_resolution
                                proc_height = int(processing_resolution / aspect_ratio)
                            else:  # Portrait
                                proc_height = processing_resolution
                                proc_width = int(processing_resolution * aspect_ratio)
            except Empty:
                pass
            except Exception as e:
                print(f"Errore nella gestione dei comandi: {e}")

            # Invia aggiornamenti FPS alla GUI ogni secondo
            if fps > 0:
                try:
                    shared_state.status_queue.put(("fps_update", fps))
                except:
                    pass

            time.sleep(0.1)  # Controlla ogni 100ms

    # Thread worker per elaborazione
    def processing_worker():
        nonlocal running

        while running:
            if frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = frame_queue.get()
            processed = process_frame(frame)

            # Aggiorna il frame elaborato evitando blocchi
            if not result_queue.full():
                result_queue.put(processed)
            frame_queue.task_done()

    # Thread per cattura frame
    def capture_frames():
        nonlocal running, frame_count
        frame_skip = 0  # Per saltare frame se necessario

        while running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1

            # Skip frame se la coda è piena (controllo dinamico)
            if frame_skip > 0:
                frame_skip -= 1
                continue

            # Gestione ottimizzata della coda
            if not frame_queue.full():
                frame_queue.put(frame.copy())
            else:
                frame_skip = 1  # Salta il prossimo frame
                try:
                    _ = frame_queue.get_nowait()
                    frame_queue.put(frame.copy())
                except:
                    pass

    # Avvio il processo GUI
    gui_process = start_gui_process(shared_state)

    # Imposta stato condiviso
    shared_state.running.value = True
    shared_state.status_queue.put(("running_status", True))

    # Avvia thread di monitoraggio modello
    monitor = threading.Thread(target=model_monitor_thread)
    monitor.daemon = True
    monitor.start()

    # Avvia thread di elaborazione
    num_workers = 1  # Ridotto per evitare contesa risorse
    executor = ThreadPoolExecutor(max_workers=num_workers)
    for _ in range(num_workers):
        executor.submit(processing_worker)

    # Avvia thread di cattura
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()

    print("🎥 Premi 'q' per uscire.")

    # Regolazione risoluzione
    current_res_scale = 1.0

    # Loop di visualizzazione ottimizzato
    processed_frame = None  # Variabile per memorizzare l'ultimo frame elaborato

    while running:
        # Aggiorna FPS
        current_time = time.time()
        if current_time - last_fps_check >= 1.0:
            fps = frame_count / (current_time - last_fps_check)
            frame_count = 0
            last_fps_check = current_time

            # Regolazione dinamica della risoluzione in base all'FPS
            if dynamic_processing and processing_times and len(processing_times) > 5:
                avg_proc = sum(processing_times) / len(processing_times)
                if fps < 15 and current_res_scale > 0.6:  # FPS troppo basso
                    current_res_scale -= 0.05
                    new_res = int(768 * current_res_scale)
                    if new_res >= 512:  # Non scendere sotto 512
                        shared_state.command_queue.put(("change_resolution", new_res))
                        print(f"⚙️ Risoluzione ridotta a {new_res} per migliorare FPS")

                elif fps > 25 and current_res_scale < 1.0:  # FPS abbastanza alto
                    current_res_scale += 0.05
                    new_res = int(768 * current_res_scale)
                    if new_res <= 1024:  # Non salire oltre 1024
                        shared_state.command_queue.put(("change_resolution", new_res))
                        print(f"⚙️ Risoluzione aumentata a {new_res} per migliorare qualità")

        # Visualizza frame elaborato
        if not result_queue.empty():
            processed_frame = result_queue.get()

        # Solo se show_preview è attivo
        if hasattr(__main__, 'show_preview') and __main__.show_preview and processed_frame is not None:
            cv2.imshow("Capitan Acciaio Cartoonizer - Live", processed_frame)
        else:
            time.sleep(0.001)  # Minima attesa

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('r'):  # Riduce manualmente la risoluzione
            new_res = max(512, processing_resolution - 64)
            shared_state.command_queue.put(("change_resolution", new_res))
            current_res_scale = new_res / 768
            print(f"⬇️ Risoluzione ridotta a {new_res}")
        elif key == ord('t'):  # Aumenta manualmente la risoluzione
            new_res = min(1024, processing_resolution + 64)
            shared_state.command_queue.put(("change_resolution", new_res))
            current_res_scale = new_res / 768
            print(f"⬆️ Risoluzione aumentata a {new_res}")

    # Pulizia
    shared_state.running.value = False
    shared_state.status_queue.put(("running_status", False))
    running = False
    time.sleep(0.2)
    cap.release()
    cv2.destroyAllWindows()
    executor.shutdown(wait=False)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    # Attendi chiusura processo GUI
    gui_process.join(timeout=1)
    sys.exit(0)


if __name__ == "__main__":
    # Prima di tutto, esegui freeze_support
    mp.freeze_support()

    # Verifica dipendenze
    check_dependencies()

    # Crea lo stato condiviso DOPO freeze_support
    from shared_state import create_shared_state

    shared_state = create_shared_state()

    # Avvio dell'applicazione
    main(shared_state)
