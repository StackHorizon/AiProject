# main.py - Ottimizzato per prestazioni su macOS
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


# Ottimizzazioni specifiche per macOS
def setup_high_priority():
    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            # Imposta sia il processo principale che i thread in modalità real-time
            os.system(
                f"sudo -n chrt -r -p 99 {os.getpid()} 2>/dev/null || sudo -n renice -20 -p {os.getpid()} 2>/dev/null || renice -10 -p {os.getpid()}")

            # Disabilita sleep per migliorare responsività
            os.system("caffeinate -d -i -m -s &")

            print("✅ Priorità elevata impostata su macOS")
        except Exception as e:
            print(f"⚠️ Impossibile impostare priorità elevata: {e}")

        # Ottimizzazione thread OpenCV per macOS (numero ideale per M1/M2)
        cv2.setNumThreads(min(4, mp.cpu_count()))
    else:
        # Configurazione standard per altri sistemi
        try:
            import psutil
            process = psutil.Process(os.getpid())
            process.nice(psutil.HIGH_PRIORITY_CLASS if system == "Windows" else -10)
        except Exception:
            pass
        cv2.setNumThreads(mp.cpu_count())

    # Impostiamo flag di ottimizzazione per OpenCV
    cv2.setUseOptimized(True)


# Caricamento modello ottimizzato per macOS
def load_optimized_model(model_name, device):
    print(f"Caricamento modello ottimizzato per {platform.system()}: {model_name}")

    # Crea directory se non esiste
    os.makedirs("models", exist_ok=True)

    # Su macOS, preferisci MPS se disponibile (Metal Performance Shaders)
    if platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✅ Utilizzo accelerazione Metal (MPS)")
        jit_path = os.path.join("models", f"{model_name}_mps.pt")
    else:
        jit_path = os.path.join("models", f"{model_name}_jit_quantized.pt")

    try:
        # Caricamento diretto JIT se già esistente
        if os.path.exists(jit_path):
            model = torch.jit.load(jit_path, map_location=device)
            print("✅ Modello JIT caricato dalla cache")
            return model

        # Altrimenti carica e ottimizza il modello PyTorch
        model = load_torch_model(model_name, device)

        # Per macOS, usiamo un'ottimizzazione specifica per MPS
        if device.type == 'mps':
            # Non facciamo jit trace su MPS (può causare problemi)
            return model

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


# Caricamento modello PyTorch con ottimizzazioni macOS
def load_torch_model(model_name, device):
    print(f"Caricamento modello PyTorch: {model_name}")

    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator",
        pretrained=model_name
    )
    model = model.to(device)
    model.eval()

    # Ottimizzazioni per GPU
    if device.type == 'cuda':
        model = model.half()

        for param in model.parameters():
            param.requires_grad = False
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
        else:
            input_tensor_cache[key] = torch.zeros((1, 3, size[0], size[1]),
                                                  device=device)
    return input_tensor_cache[key]


# Pipeline di elaborazione ottimizzata per macOS
def process_frame(frame, model, device, resolution, last_frame_hash=None):
    try:
        # Skip se il frame è identico al precedente (ultra-ottimizzato)
        if last_frame_hash is not None:
            # Su macOS, usiamo un campionamento ancora più aggressivo per ridurre il carico
            stride = 100 if platform.system() == "Darwin" else 50
            curr_hash = hash(frame[::stride, ::stride, 0].tobytes())
            if curr_hash == last_frame_hash:
                return None, curr_hash
            last_frame_hash = curr_hash

        # Ottimizzazione macOS: riduci subito il frame per elaborazioni successive
        h, w = frame.shape[:2]
        target_size = min(resolution, min(h, w))
        ratio = target_size / min(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # Su macOS, usiamo INTER_NEAREST per la velocità massima durante il downscaling
        if platform.system() == "Darwin":
            small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Converti BGR a RGB (ottimizzato per macOS, che ha problemi con le conversioni colore)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Allineamento a multipli di 16 per macOS (invece di 32 per CUDA)
        align_multiple = 16 if platform.system() == "Darwin" else 32
        tensor_h = ((new_h + align_multiple - 1) // align_multiple) * align_multiple
        tensor_w = ((new_w + align_multiple - 1) // align_multiple) * align_multiple

        input_tensor = get_input_tensor((tensor_h, tensor_w), device)

        # Converti immagine in tensor ottimizzato
        img = Image.fromarray(rgb_frame)

        # Preprocessing ottimizzato con cache dei trasformatori
        img_key = (new_h, new_w)
        if img_key not in transform_cache:
            transform_cache[img_key] = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        img_tensor = transform_cache[img_key](img).unsqueeze(0)

        # Copia nel buffer pre-allocato
        with torch.no_grad():
            input_tensor.zero_()
            if device.type == 'cuda':
                input_tensor[:, :, :new_h, :new_w].copy_(img_tensor.half())
            else:
                input_tensor[:, :, :new_h, :new_w].copy_(img_tensor)

            # Inferenza
            output = model(input_tensor[:, :, :new_h, :new_w])

            # Sincronizza per Metal/CUDA
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()

            # Denormalizza e converti risultato
            result = (output[0, :, :new_h, :new_w].permute(1, 2, 0) * 0.5 + 0.5)
            result = result.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Converti RGB a BGR per OpenCV
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Ridimensiona al formato originale con algoritmo più veloce su macOS
        if new_h != h or new_w != w:
            if platform.system() == "Darwin":
                result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        return result_bgr, last_frame_hash

    except Exception as e:
        print(f"❌ Errore elaborazione frame: {e}")
        return frame, last_frame_hash


def main(shared_state):
    # Imposta priorità elevata
    setup_high_priority()

    # Configurazioni
    frame_queue = Queue(maxsize=1)  # Ridotto a 1 per macOS per ridurre latenza
    result_queue = Queue(maxsize=1)
    running = True

    # Variabili per monitoraggio performance
    last_update = time.time()
    frame_count = 0
    fps = 0
    last_frame_hash = None

    # Imposta dispositivo - Supporto speciale per Metal su macOS
    if platform.system() == "Darwin" and hasattr(torch, "backends") and hasattr(torch.backends,
                                                                                "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Utilizzando dispositivo: Metal (MPS) su MacOS")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilizzando dispositivo: {device}")

    # Ottimizzazioni performance per macOS
    if platform.system() == "Darwin":
        # Riduci il buffer di OpenCV per macOS
        cv2.setNumThreads(min(4, mp.cpu_count()))
    elif device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True
        torch.cuda.empty_cache()

    # Configura webcam con ottimizzazioni macOS
    cap = cv2.VideoCapture(0)

    # Su macOS, ridurre la risoluzione iniziale per migliorare performance
    if platform.system() == "Darwin":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 30fps più stabile su macOS

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

    # Thread di elaborazione ottimizzato per macOS
    def processing_worker():
        nonlocal running, last_frame_hash

        # Su macOS, imposta thread priority
        if platform.system() == "Darwin":
            try:
                os.system(f"renice -10 {os.getpid()} 2>/dev/null")
            except:
                pass

        while running:
            try:
                if not frame_queue.empty():
                    frame = frame_queue.get()

                    # Elaborazione frame
                    with model_lock:
                        result, last_frame_hash = process_frame(
                            frame,
                            model,
                            device,
                            shared_state.processing_resolution.value,
                            last_frame_hash
                        )

                    # Metti risultato in coda se valido
                    if result is not None and not result_queue.full():
                        # Su macOS, sostituisci invece di svuotare per evitare overhead
                        while not result_queue.empty():
                            try:
                                result_queue.get_nowait()
                            except:
                                break
                        result_queue.put(result)

                else:
                    # Su macOS dormi meno per essere più reattivi
                    time.sleep(0.0005)

            except Exception as e:
                print(f"❌ Errore worker: {e}")
                time.sleep(0.01)

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

    # Aggiorna stato
    shared_state.running.value = True
    print("🎥 Premi 'q' per uscire")

    # Parametri per rendering ottimizzato - Skip frame adattivo (più aggressivo su macOS)
    skip_frames = 0
    dynamic_skip = 0 if platform.system() != "Darwin" else 2  # Start with higher skip on macOS

    # Crea finestra con flag speciali
    if platform.system() == "Darwin":
        cv2.namedWindow("Cartoonizer", cv2.WINDOW_AUTOSIZE)  # AUTOSIZE è più veloce su macOS
    else:
        cv2.namedWindow("Cartoonizer", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Target FPS per adattamento dinamico
    target_fps = 20 if platform.system() == "Darwin" else 30

    # Performance counters
    perf_history = []

    # Loop principale
    while running:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            if running:
                print("⚠️ Errore lettura webcam, riprovo...")
                time.sleep(0.1)
                continue
            break

        frame_count += 1

        # Gestione skip frame per mantenere fluidità (più aggressiva su macOS)
        if skip_frames > 0:
            skip_frames -= 1
            continue

        # Dynamic skip per frame rate
        if dynamic_skip > 0:
            dynamic_skip -= 1
            continue

        # Su macOS, ridimensiona subito il frame se è troppo grande
        if platform.system() == "Darwin" and (frame.shape[0] > 480 or frame.shape[1] > 640):
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)

        # Metti frame in coda se non piena
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # Salta frame se coda elaborazione è piena
            # Su macOS usiamo uno skip più aggressivo
            dynamic_skip = 3 if platform.system() == "Darwin" else 2

        # Mostra risultato se disponibile
        if not result_queue.empty():
            processed = result_queue.get()
            cv2.imshow("Cartoonizer", processed)

        # Calcola FPS ogni 0.5 secondi
        current_time = time.time()
        if current_time - last_update >= 0.5:
            fps = frame_count / (current_time - last_update)
            frame_count = 0
            last_update = current_time

            # Tieni traccia della storia performance
            perf_history.append(fps)
            if len(perf_history) > 5:
                perf_history.pop(0)

            # Invia FPS alla GUI
            shared_state.status_queue.put(("fps_update", fps))
            print(f"📊 FPS: {fps:.1f}")

            # Adatta skip frame dinamico in base a FPS
            # Su macOS usiamo soglie diverse
            if platform.system() == "Darwin":
                # Ottimizzazione più aggressiva per macOS
                if fps < 15:
                    dynamic_skip = 4
                    # Riduci anche la risoluzione se persistente
                    if len(perf_history) > 2 and all(f < 15 for f in perf_history[-2:]):
                        new_res = max(512, shared_state.processing_resolution.value - 128)
                        shared_state.command_queue.put(("change_resolution", new_res))
                elif fps < 20:
                    dynamic_skip = 3
                elif fps < 25:
                    dynamic_skip = 2
                else:
                    dynamic_skip = 1  # Mantieni almeno skip=1 per sicurezza su macOS
            else:
                # Ottimizzazioni standard
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

        # Se su macOS, limita il frame rate per evitare sovraccarico
        if platform.system() == "Darwin":
            elapsed = time.time() - start_time
            if elapsed < 1.0 / 30:  # Limita a 30 Hz
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
