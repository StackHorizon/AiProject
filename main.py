# main.py - Ottimizzato per prestazioni massime con risoluzione manuale
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


# Ottimizzazioni specifiche per sistema operativo
def setup_high_priority():
    system = platform.system()

    if system == "Windows":
        try:
            import psutil
            process = psutil.Process(os.getpid())
            process.nice(psutil.HIGH_PRIORITY_CLASS)
            print("✅ Priorità elevata impostata su Windows")
        except Exception as e:
            print(f"⚠️ Impossibile impostare priorità elevata: {e}")

    elif system == "Darwin":  # macOS
        try:
            os.system(f"sudo -n renice -n -10 -p {os.getpid()} 2>/dev/null || renice -n -10 -p {os.getpid()}")
            print("✅ Priorità elevata impostata su macOS")
        except Exception as e:
            print(f"⚠️ Impossibile impostare priorità elevata: {e}")

    # Ottimizzazioni OpenCV
    cv2.setNumThreads(mp.cpu_count())

    # Impostiamo flag di ottimizzazione per OpenCV
    cv2.setUseOptimized(True)


# Caricamento modello ultra-ottimizzato con TorchScript e quantizzazione
def load_optimized_model(model_name, device):
    print(f"Caricamento modello ottimizzato: {model_name}")

    # Crea directory se non esiste
    os.makedirs("models", exist_ok=True)
    jit_path = os.path.join("models", f"{model_name}_jit_quantized.pt")

    try:
        # Caricamento diretto JIT se già esistente
        if os.path.exists(jit_path):
            model = torch.jit.load(jit_path, map_location=device)
            print("✅ Modello JIT quantizzato caricato dalla cache")
            return model

        # Altrimenti carica e ottimizza il modello PyTorch
        model = load_torch_model(model_name, device)

        # Conversione a JIT con quantizzazione
        try:
            dummy_input = torch.randn(1, 3, 384, 384, device=device)
            if device.type == 'cuda':
                dummy_input = dummy_input.half()

            # Traccia il modello per futura esecuzione
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)

            # Quantizza il modello se non è su GPU
            if device.type != 'cuda':
                traced_model = torch.quantization.quantize_dynamic(
                    traced_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )

            traced_model.save(jit_path)
            print("✅ Modello JIT ottimizzato, quantizzato e salvato")
            return traced_model

        except Exception as e:
            print(f"⚠️ Conversione JIT fallita: {e}")
            return model

    except Exception as e:
        print(f"⚠️ Caricamento modello ottimizzato fallito: {e}")
        return load_torch_model(model_name, device)


# Caricamento modello PyTorch standard
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
        model = model.half()  # Usa half precision (FP16) per velocità

        # Disabilita gradient tracking per inferenza
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Ottimizzazioni specifiche per CPU
        # Setta thread interni di PyTorch
        torch.set_num_threads(mp.cpu_count())
        # Abilita optimized memory layout
        torch.set_flush_denormal(True)

    print("✅ Modello PyTorch caricato")
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()

    return model


# Buffer pre-allocati per evitare allocazioni continue
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


# Pipeline di elaborazione ultra-ottimizzata per alte prestazioni
def process_frame(frame, model, device, resolution, last_frame_hash=None):
    try:
        # Skip se il frame è identico al precedente (ultra-ottimizzato)
        if last_frame_hash is not None:
            # Campionamento più aggressivo per confronto hash
            curr_hash = hash(frame[::50, ::50, 0].tobytes())
            if curr_hash == last_frame_hash:
                return None, curr_hash
            last_frame_hash = curr_hash

        # Mantiene la risoluzione definita dall'utente
        h, w = frame.shape[:2]
        target_size = min(resolution, min(h, w))
        ratio = target_size / min(h, w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # Ridimensionamento fast-path usando INTER_AREA (miglior compromesso velocità/qualità)
        small_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Converti BGR a RGB (solo i canali che servono)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Allineamento a multipli di 32 per ottimizzazioni CUDA/simd
        tensor_h, tensor_w = ((new_h + 31) // 32) * 32, ((new_w + 31) // 32) * 32
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
            input_tensor[:, :, :new_h, :new_w].copy_(img_tensor if device.type != 'cuda'
                                                     else img_tensor.half())

            # Inferenza con profiling disabilitato per velocità
            output = model(input_tensor[:, :, :new_h, :new_w])
            torch.cuda.synchronize() if device.type == 'cuda' else None

            # Denormalizza e converti risultato
            result = (output[0, :, :new_h, :new_w].permute(1, 2, 0) * 0.5 + 0.5)
            result = result.mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()

        # Converti RGB a BGR per OpenCV
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Ridimensiona al formato originale con algoritmo veloce
        if new_h != h or new_w != w:
            result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LINEAR)

        return result_bgr, last_frame_hash

    except Exception as e:
        print(f"❌ Errore elaborazione frame: {e}")
        return frame, last_frame_hash


def main(shared_state):
    # Imposta priorità elevata
    setup_high_priority()

    # Configurazioni
    frame_queue = Queue(maxsize=2)  # Aumentato a 2 per buffer migliore
    result_queue = Queue(maxsize=2)
    running = True

    # Variabili per monitoraggio performance
    last_update = time.time()
    frame_count = 0
    fps = 0
    last_frame_hash = None

    # Imposta dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzando dispositivo: {device}")

    # Ottimizzazioni performance
    if device.type == 'cuda':
        # Ottimizzazioni CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

        # Svuota cache GPU
        torch.cuda.empty_cache()

    # Configura webcam con buffer minimo per bassa latenza
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer minimo per bassa latenza
    cap.set(cv2.CAP_PROP_FPS, 60)  # Richiedi massimo frame rate

    # Ottimizzazioni aggiuntive per webcam
    if platform.system() == "Windows":
        # Accelerazione hardware e formato compresso
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            print("⚠️ Accelerazione hardware webcam non supportata")

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

    # Thread di elaborazione
    def processing_worker():
        nonlocal running, last_frame_hash

        try:
            # Precarica thread e ottimizza affinity
            if platform.system() == "Windows":
                try:
                    import psutil
                    p = psutil.Process()
                    # Utilizza solo core fisici, non logici
                    physical_cores = [i for i in range(0, mp.cpu_count(), 2)]
                    if physical_cores:
                        p.cpu_affinity(physical_cores)
                        print("✅ Worker su core fisici")
                except Exception:
                    pass
        except Exception:
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
                        # Svuota coda per evitare ritardi
                        while not result_queue.empty():
                            result_queue.get()
                        result_queue.put(result)

                else:
                    time.sleep(0.001)

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

                time.sleep(0.1)

            except Empty:
                time.sleep(0.1)
            except Exception as e:
                print(f"❌ Errore monitoraggio: {e}")
                time.sleep(0.1)

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

    # Parametri per rendering ottimizzato - Skip frame adattivo
    skip_frames = 0
    dynamic_skip = 0

    # Crea finestra con flag speciali per priorità
    cv2.namedWindow("Cartoonizer", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Loop principale
    while running:
        ret, frame = cap.read()
        if not ret:
            if running:
                print("⚠️ Errore lettura webcam, riprovo...")
                time.sleep(0.1)
                continue
            break

        frame_count += 1

        # Gestione skip frame per mantenere fluidità
        if skip_frames > 0:
            skip_frames -= 1
            continue

        # Dynamic skip per frame rate
        if dynamic_skip > 0:
            dynamic_skip -= 1
            continue

        # Metti frame in coda se non piena
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # Salta frame se coda elaborazione è piena
            dynamic_skip = 2

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

            # Invia FPS alla GUI
            shared_state.status_queue.put(("fps_update", fps))
            print(f"📊 FPS: {fps:.1f}")

            # Adatta skip frame dinamico in base a FPS (senza toccare risoluzione)
            if fps < 30:
                dynamic_skip = 3  # Skip più aggressivo se fps bassi
            elif fps < 45:
                dynamic_skip = 2
            elif fps < 60:
                dynamic_skip = 1
            else:
                dynamic_skip = 0  # Non saltare frame se FPS alti

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

    # Pulizia risorse
    running = False
    shared_state.running.value = False

    # Attendi chiusura thread
    time.sleep(0.2)

    # Rilascia webcam e finestra
    cap.release()
    cv2.destroyAllWindows()

    # Pulizia memoria
    input_tensor_cache.clear()
    transform_cache.clear()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    print("👋 Applicazione terminata")


if __name__ == "__main__":
    mp.freeze_support()
    check_dependencies()

    from shared_state import create_shared_state

    shared_state = create_shared_state()

    main(shared_state)
