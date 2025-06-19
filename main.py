# main.py - Ottimizzato per prestazioni su Windows 11 con GPU NVIDIA 4070
import cv2
import numpy as np
import threading
import time
import gc
from queue import Queue, Empty
import torch
import torchvision.transforms as transforms
import multiprocessing as mp
import sys
import os
import importlib
import platform
import subprocess
from PIL import Image
import ctypes


# Verifica e installa dipendenze
def check_dependencies():
    print("Installazione pacchetti mancanti: opencv-python, pillow...")

    required_packages = ["numpy", "opencv-python", "torch", "customtkinter", "pillow", "psutil"]

    # Controlla e installa pacchetti necessari
    missing = [pkg for pkg in required_packages if importlib_check(pkg) is False]
    if missing:
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
            except Exception as e:
                print(f"Errore installando {pkg}: {e}")

    print("✅ Dipendenze installate")


def importlib_check(name):
    try:
        __import__(name.split('-')[0])
        return True
    except ImportError:
        return False


# Ottimizzazioni specifiche per Windows 11 con GPU NVIDIA 4070
def setup_high_priority():
    system = platform.system()

    if system == "Windows":
        try:
            import psutil
            process = psutil.Process(os.getpid())
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            # Imposta priorità thread principale
            try:
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
            except Exception:
                pass
            print("✅ Priorità elevata impostata su Windows")
        except Exception as e:
            print(f"⚠️ Impossibile impostare priorità elevata: {e}")
        # Usa tutti i core disponibili per OpenCV
        cv2.setNumThreads(mp.cpu_count())
    elif system == "Darwin":
        # Configurazione specifica per macOS (rimossa per ottimizzazione Windows)
        pass
    else:
        # Configurazione standard per altri sistemi
        try:
            import psutil
            process = psutil.Process(os.getpid())
            process.nice(psutil.HIGH_PRIORITY_CLASS if system == "Windows" else -10)
        except Exception:
            pass
        cv2.setNumThreads(mp.cpu_count())

    cv2.setUseOptimized(True)


# Caricamento modello ottimizzato per Windows 11 con GPU NVIDIA 4070
def load_optimized_model(model_name, device):
    print(f"Caricamento modello ottimizzato per {platform.system()}: {model_name}")

    os.makedirs("models", exist_ok=True)

    if platform.system() == "Windows" and device.type == "cuda":
        jit_path = os.path.join("models", f"{model_name}_jit_cuda.pt")
    elif platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Ottimizzazione specifica per Metal su macOS
        jit_path = os.path.join("models", f"{model_name}_mps.pt")
    else:
        jit_path = os.path.join("models", f"{model_name}_jit_quantized.pt")

    try:
        if os.path.exists(jit_path):
            model = torch.jit.load(jit_path, map_location=device)
            print("✅ Modello JIT caricato dalla cache")
            return model

        model = load_torch_model(model_name, device)

        if device.type == 'cuda':
            dummy_input = torch.randn(1, 3, 384, 384, device=device).half()
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(jit_path)
            print("✅ Modello JIT ottimizzato e salvato (CUDA)")
            return traced_model
        elif device.type == 'mps':
            # Ottimizzazione specifica per MPS (macOS)
            return model
        else:
            # Conversione a JIT con quantizzazione per altri dispositivi
            try:
                dummy_input = torch.randn(1, 3, 384, 384, device=device)
                if device.type == 'cuda':
                    dummy_input = dummy_input.half()

                # Traccia il modello per futura esecuzione
                with torch.no_grad():
                    traced_model = torch.jit.trace(model, dummy_input)

                # Quantizza il modello se non è su GPU
                if device.type not in ['cuda', 'mps']:
                    traced_model = torch.quantization.quantize_dynamic(
                        traced_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                    )

                traced_model.save(jit_path)
                print("✅ Modello JIT ottimizzato e salvato")
                return traced_model

            except Exception as e:
                print(f"⚠️ Conversione JIT fallita: {e}")
                return model

    except Exception as e:
        print(f"⚠️ Caricamento modello ottimizzato fallito: {e}")
        return load_torch_model(model_name, device)


# Caricamento modello PyTorch con ottimizzazioni per Windows 11
def load_torch_model(model_name, device):
    print(f"Caricamento modello PyTorch: {model_name}")

    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained=model_name
    )
    model = model.to(device)
    model.eval()

    if device.type == 'cuda':
        model = model.half()
        for param in model.parameters():
            param.requires_grad = False
        # Ottimizzazioni CUDA avanzate
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        torch.set_num_threads(mp.cpu_count())
    elif device.type == 'mps':
        # Ottimizzazioni specifiche per Metal (macOS)
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Ottimizzazioni specifiche per CPU su macOS
        if platform.system() == "Darwin":
            # Su macOS, impostare i thread in maniera conservativa
            torch.set_num_threads(min(4, mp.cpu_count()))
        else:
            torch.set_num_threads(mp.cpu_count())

        # Abilita optimized memory layout
        torch.set_flush_denormal(True)

    print(f"✅ Modello PyTorch caricato su {device}")

    # Pulizia memoria
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    return model


# Buffer pre-allocati ottimizzati per macOS
input_tensor_cache = {}
transform_cache = {}


def get_input_tensor(size, device):
    key = (size[0], size[1], str(device))
    if key not in input_tensor_cache:
        if device.type == 'cuda':
            input_tensor_cache[key] = torch.zeros((1, 3, size[0], size[1]),
                                                  device=device,
                                                  dtype=torch.half)
        elif device.type == 'cpu':
            # Solo su CPU puoi usare pin_memory=True
            input_tensor_cache[key] = torch.zeros((1, 3, size[0], size[1]),
                                                  device=device,
                                                  pin_memory=True)
        else:
            input_tensor_cache[key] = torch.zeros((1, 3, size[0], size[1]),
                                                  device=device)
    return input_tensor_cache[key]


# Pipeline di elaborazione ottimizzata per Windows 11
def process_frame(frame, model, device, resolution, last_frame_hash=None):
    try:
        # RIMOSSO: controllo hash per evitare skip frame inutili

        # Downscale frame con INTER_AREA (qualità/velocità su Windows)
        h, w = frame.shape[:2]
        target_size = min(resolution, min(h, w))
        ratio = target_size / min(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        align_multiple = 32
        tensor_h = ((new_h + align_multiple - 1) // align_multiple) * align_multiple
        tensor_w = ((new_w + align_multiple - 1) // align_multiple) * align_multiple

        input_tensor = get_input_tensor((tensor_h, tensor_w), device)

        img = Image.fromarray(rgb_frame)
        img_key = (new_h, new_w)
        if img_key not in transform_cache:
            transform_cache[img_key] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        img_tensor = transform_cache[img_key](img).unsqueeze(0)

        with torch.no_grad():
            input_tensor.zero_()
            if device.type == 'cuda':
                input_tensor[:, :, :new_h, :new_w].copy_(img_tensor.half())
            else:
                input_tensor[:, :, :new_h, :new_w].copy_(img_tensor)

            output = model(input_tensor[:, :, :new_h, :new_w])

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps':
                torch.mps.synchronize()

            result = (output[0, :, :new_h, :new_w].permute(1, 2, 0) * 0.5 + 0.5)
            result = result.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if new_h != h or new_w != w:
            result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        return result_bgr, None  # Non serve più last_frame_hash

    except Exception as e:
        print(f"❌ Errore elaborazione frame: {e}")
        return frame, None


def main(shared_state):
    setup_high_priority()

    # Coda frame grezzi (lettura webcam) e coda risultati (frame cartoonizzati)
    raw_frame_queue = Queue(maxsize=1)    # SOLO l'ultimo frame letto
    result_queue = Queue(maxsize=1)       # SOLO l'ultimo frame processato
    running = True

    # Variabili per monitoraggio performance
    last_update = time.time()
    frame_count = 0
    fps = 0
    last_frame_hash = None

    # Imposta dispositivo - Supporto speciale per Metal su macOS
    if platform.system() == "Windows" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Utilizzando dispositivo: CUDA (NVIDIA)")
    elif platform.system() == "Darwin" and hasattr(torch, "backends") and hasattr(torch.backends,
                                                                                "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Utilizzando dispositivo: Metal (MPS) su MacOS")
    else:
        device = torch.device("cpu")
        print(f"Utilizzando dispositivo: {device}")

    # Ottimizzazioni performance per Windows 11
    if platform.system() == "Windows" and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()
        torch.set_num_threads(mp.cpu_count())
        cv2.setNumThreads(mp.cpu_count())
    elif platform.system() == "Darwin":
        # Configurazione specifica per macOS (rimossa per ottimizzazione Windows)
        pass
    else:
        # Configurazione standard per altri sistemi
        cv2.setNumThreads(mp.cpu_count())

    # Configura webcam con ottimizzazioni Windows 11
    cap = cv2.VideoCapture(0)
    if platform.system() == "Windows":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_FPS, 60)  # Prova a forzare 60fps su webcam moderne
    elif platform.system() == "Darwin":
        # Su macOS, ridurre la risoluzione iniziale per migliorare performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Riduci latenze su Windows
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # Usa MJPEG per bassa latenza
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Riduci buffer

    # Verifica apertura webcam
    if not cap.isOpened():
        print("❌ Errore apertura webcam")
        return

    # Info risoluzione
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Risoluzione Webcam: {width} x {height}")

    # Carica modello ottimizzato
    print(f"Preparazione modello ottimizzato: {shared_state.current_model.value}")
    model = load_optimized_model(shared_state.current_model.value, device)
    model_lock = threading.Lock()

    # Thread di lettura webcam: mantiene solo l'ultimo frame disponibile
    def webcam_reader():
        while running:
            ret, frame = cap.read()
            if not ret:
                continue
            # Svuota la coda per tenere solo l'ultimo frame
            while not raw_frame_queue.empty():
                try:
                    raw_frame_queue.get_nowait()
                except:
                    break
            try:
                raw_frame_queue.put_nowait(frame)
            except:
                pass

    # Thread di elaborazione: prende sempre l'ultimo frame disponibile
    def processing_worker():
        nonlocal running
        while running:
            try:
                if not raw_frame_queue.empty():
                    # Prendi sempre l'ultimo frame disponibile
                    while raw_frame_queue.qsize() > 1:
                        try:
                            raw_frame_queue.get_nowait()
                        except:
                            break
                    frame = raw_frame_queue.get()
                    with model_lock:
                        result, _ = process_frame(
                            frame,
                            model,
                            device,
                            shared_state.processing_resolution.value,
                            None
                        )
                    if result is not None:
                        # Sostituisci sempre il frame processato (drop frame vecchi)
                        while not result_queue.empty():
                            try:
                                result_queue.get_nowait()
                            except:
                                break
                        result_queue.put(result)
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"❌ Errore worker: {e}")
                time.sleep(0.005)

    # Thread monitoraggio comandi
    def monitor_thread():
        nonlocal running, model

        while running:
            try:
                if not shared_state.command_queue.empty():
                    command, value = shared_state.command_queue.get(block=False)

                    if command == "change_model":
                        print(f"🔄 Cambio modello a: {value}")

                        with model_lock:
                            # Libera memoria prima di caricare nuovo modello
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                            elif device.type == 'mps':
                                # Forza garbage collection su macOS
                                gc.collect()
                            else:
                                gc.collect()

                            # Carica nuovo modello
                            shared_state.current_model.value = value
                            model = load_optimized_model(value, device)

                            # Aggiorna stato
                            shared_state.status_queue.put(("model_changed", value))

                    elif command == "change_resolution":
                        print(f"🖼️ Cambio risoluzione: {value}px")
                        shared_state.processing_resolution.value = value
                        shared_state.status_queue.put(("resolution_changed", value))

                    elif command == "exit":
                        print("👋 Chiusura applicazione...")
                        running = False

                # Dormi meno su macOS per essere più reattivi
                time.sleep(0.05 if platform.system() == "Darwin" else 0.1)

            except Empty:
                time.sleep(0.05 if platform.system() == "Darwin" else 0.1)
            except Exception as e:
                print(f"❌ Errore monitoraggio: {e}")
                time.sleep(0.05 if platform.system() == "Darwin" else 0.1)

    # Avvia interfaccia GUI
    from gui import start_gui
    gui_process = mp.Process(
        target=start_gui,
        args=(shared_state.available_models,
              shared_state.command_queue,
              shared_state.status_queue,
              shared_state.current_model.value,
              shared_state.processing_resolution.value)
    )
    gui_process.daemon = True
    gui_process.start()

    # Avvia thread monitoraggio
    monitor = threading.Thread(target=monitor_thread)
    monitor.daemon = True
    monitor.start()

    # Avvia thread elaborazione
    worker = threading.Thread(target=processing_worker)
    worker.daemon = True
    worker.start()

    # Avvia thread lettura webcam
    webcam_thread = threading.Thread(target=webcam_reader)
    webcam_thread.daemon = True
    webcam_thread.start()

    # Aggiorna stato
    shared_state.running.value = True
    print("🎥 Premi 'q' per uscire")

    # Parametri per rendering ottimizzato - Skip frame adattivo (più aggressivo su macOS)
    skip_frames = 0
    dynamic_skip = 0 if platform.system() != "Darwin" else 2

    # Crea finestra con flag speciali
    if platform.system() == "Darwin":
        cv2.namedWindow("Cartoonizer", cv2.WINDOW_AUTOSIZE)  # AUTOSIZE è più veloce su macOS
    else:
        cv2.namedWindow("Cartoonizer", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Target FPS per adattamento dinamico
    target_fps = 20 if platform.system() == "Darwin" else 30

    # Performance counters
    perf_history = []
    last_update = time.time()
    frame_count = 0

    # Loop principale: mostra sempre l'ultimo frame processato disponibile
    while running:
        start_time = time.time()
        frame_count += 1

        # Mostra sempre l'ultimo frame processato disponibile
        if not result_queue.empty():
            processed = result_queue.get()
            cv2.imshow("Cartoonizer", processed)

        current_time = time.time()
        if current_time - last_update >= 0.5:
            fps = frame_count / (current_time - last_update)
            frame_count = 0
            last_update = current_time

            perf_history.append(fps)
            if len(perf_history) > 5:
                perf_history.pop(0)

            shared_state.status_queue.put(("fps_update", fps))
            print(f"📊 FPS: {fps:.1f}")

            if platform.system() == "Windows":
                if fps < 30:
                    dynamic_skip = 2
                elif fps < 45:
                    dynamic_skip = 1
                else:
                    dynamic_skip = 0
            elif platform.system() == "Darwin":
                pass
            else:
                if fps < 30:
                    dynamic_skip = 3
                elif fps < 45:
                    dynamic_skip = 2
                elif fps < 60:
                    dynamic_skip = 1
                else:
                    dynamic_skip = 0

        # Input utente
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('-'):
            new_res = max(256, shared_state.processing_resolution.value - 64)
            shared_state.command_queue.put(("change_resolution", new_res))
        elif key == ord('+') or key == ord('='):
            new_res = min(1024, shared_state.processing_resolution.value + 64)
            shared_state.command_queue.put(("change_resolution", new_res))

        if platform.system() == "Darwin":
            elapsed = time.time() - start_time
            if elapsed < 1.0 / 30:
                time.sleep(max(0, 1.0 / 30 - elapsed))

    # Pulizia risorse
    running = False
    shared_state.running.value = False

    # Attendi chiusura thread
    time.sleep(0.2)

    # Su macOS, termina il processo caffeinate se attivo
    if platform.system() == "Darwin":
        try:
            os.system("pkill caffeinate 2>/dev/null")
        except:
            pass

    # Rilascia webcam e finestra
    cap.release()
    cv2.destroyAllWindows()

    # Pulizia memoria
    input_tensor_cache.clear()
    transform_cache.clear()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        # Garbage collection aggressivo su MPS
        gc.collect()
    gc.collect()

    print("👋 Applicazione terminata")


if __name__ == "__main__":
    mp.freeze_support()
    check_dependencies()

    from shared_state import create_shared_state

    shared_state = create_shared_state()

    main(shared_state)
