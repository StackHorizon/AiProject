# main.py
import cv2
import numpy as np
import threading
import time
import gc
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
import torch
import multiprocessing as mp
import sys
import os
import platform
import subprocess
from PIL import Image


# Verifica dipendenze all'avvio con migliore gestione errori
def check_dependencies():
    required_packages = ["numpy", "opencv-python", "torch", "customtkinter", "pillow"]
    optional_packages = ["onnxruntime"]
    missing = []

    # Verifica pacchetti richiesti
    for package in required_packages:
        try:
            __import__(package.split('-')[0])
        except ImportError:
            missing.append(package)

    # Installa i pacchetti mancanti
    if missing:
        print(f"Installazione pacchetti mancanti: {', '.join(missing)}...")
        for package in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", package])
            except Exception as e:
                print(f"❌ Errore nell'installazione di {package}: {e}")
                print("⚠️ L'applicazione potrebbe funzionare in modo limitato")

    # Tenta installazione pacchetti opzionali per accelerazione
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            try:
                print(f"Installazione pacchetto opzionale: {package}...")
                if platform.system() == 'Darwin':  # macOS
                    # Su macOS installiamo onnxruntime-silicon per supporto M1/M2
                    if platform.processor() in ['arm', 'arm64']:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install",
                            "--no-cache-dir", "onnxruntime-silicon"
                        ])
                    else:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install",
                            "--no-cache-dir", "onnxruntime"
                        ])
                else:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--no-cache-dir", package
                    ])
            except Exception as e:
                print(f"⚠️ {package} non disponibile: {e}")
                print(f"⚠️ Le prestazioni saranno ridotte senza {package}")

    print("✅ Verifica dipendenze completata")


# Funzione migliorata per caricare il modello
def load_model(model_name, device):
    try:
        # Prima tenta di usare ONNX per prestazioni migliori
        try:
            import onnxruntime as ort
            return load_onnx_model(model_name, device)
        except ImportError:
            print("⚠️ ONNX non disponibile, uso PyTorch (prestazioni ridotte)")
        except Exception as e:
            print(f"⚠️ Errore caricamento modello ONNX: {e}")

        # Fallback a PyTorch
        return load_torch_model(model_name, device)
    except Exception as e:
        print(f"❌ Errore critico caricamento modello: {e}")
        return None


# Caricamento modello ONNX ottimizzato
def load_onnx_model(model_name, device):
    import onnxruntime as ort
    print(f"Preparazione modello ONNX: {model_name}")

    # Crea directory se non esiste
    os.makedirs("models", exist_ok=True)
    onnx_path = os.path.join("models", f"{model_name}.onnx")

    # Converti modello se necessario
    if not os.path.exists(onnx_path):
        try:
            print("Conversione modello PyTorch -> ONNX...")

            # Carica modello PyTorch temporaneamente
            torch_model = torch.hub.load(
                "bryandlee/animegan2-pytorch:main",
                "generator", pretrained=model_name
            )
            torch_model.eval()

            # Esporta in formato ONNX
            dummy_input = torch.randn(1, 3, 512, 512)
            torch.onnx.export(
                torch_model, dummy_input, onnx_path,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},
                              'output': {0: 'batch', 2: 'height', 3: 'width'}},
                opset_version=11
            )

            # Libera memoria
            del torch_model
            torch.cuda.empty_cache()
            gc.collect()

            print("✅ Modello convertito in ONNX")
        except Exception as e:
            print(f"❌ Errore nella conversione ONNX: {e}")
            raise e

    # Configura provider ottimale per ogni piattaforma
    providers = []
    if device.type == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif platform.system() == 'Darwin':
        # Supporto Apple Silicon
        if 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # Ottimizzazioni specifiche per sessione ONNX
    try:
        print(f"Provider disponibili: {ort.get_available_providers()}")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Ottimizzazione CPU in base al numero di core
        import multiprocessing
        num_threads = max(1, multiprocessing.cpu_count() - 1)
        sess_options.intra_op_num_threads = num_threads

        model = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        print(f"✅ Modello ONNX caricato con provider: {model.get_providers()[0]}")
        return model
    except Exception as e:
        print(f"❌ Errore inizializzazione ONNX: {e}")
        raise e


def load_torch_model(model_name, device):
    print(f"Caricamento modello PyTorch: {model_name}")
    model = torch.hub.load(
        "bryandlee/animegan2-pytorch:main",
        "generator", pretrained=model_name
    ).to(device).eval()

    # Ottimizzazione per GPU
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        model = model.half()

    print(f"✅ Modello PyTorch caricato")
    gc.collect()
    return model


def process_frame(frame, model, device, model_lock, processing_resolution):
    if model is None:
        return frame

    with model_lock:
        try:
            # Ridimensiona immagine conservando aspect ratio
            h, w = frame.shape[:2]
            ratio = min(processing_resolution / h, processing_resolution / w)
            new_w = int(w * ratio)
            new_h = int(h * ratio)

            # Ridimensiona frame per elaborazione più veloce
            small_frame = cv2.resize(frame, (new_w, new_h))

            # Converti per il modello
            image = Image.fromarray(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))

            # Verifica se usiamo ONNX o PyTorch
            import onnxruntime
            if isinstance(model, onnxruntime.InferenceSession):
                # Pipeline ONNX
                input_tensor = np.transpose(np.array(image, dtype=np.float32) / 127.5 - 1.0, (2, 0, 1))[np.newaxis, ...]

                # Inferenza ONNX
                result = model.run(['output'], {'input': input_tensor})[0]

                # Post-processing
                result = (np.transpose(result[0], (1, 2, 0)) * 127.5 + 127.5).astype(np.uint8)
                output_image = Image.fromarray(result)

            else:
                # Pipeline PyTorch (più lenta)
                from torchvision import transforms

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ])

                input_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    if device.type == 'cuda':
                        input_tensor = input_tensor.half()

                    output = model(input_tensor)
                    output = output.squeeze(0).cpu().detach()

                    # Post-processing
                    output = output * 0.5 + 0.5
                    output_image = transforms.ToPILImage()(output)

            # Converti di nuovo in formato OpenCV
            result_frame = cv2.cvtColor(np.array(output_image), cv2.COLOR_RGB2BGR)

            # Ridimensiona al formato originale
            result_frame = cv2.resize(result_frame, (w, h))

            # Aggiungi overlay informazioni
            if hasattr(output_image, "_provider"):
                provider = output_image._provider
                cv2.putText(result_frame, f"Provider: {provider}",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return result_frame

        except Exception as e:
            print(f"❌ Errore elaborazione frame: {e}")
            return frame


def main(shared_state):
    # Inizializzazione variabili
    frame_queue = Queue(maxsize=2)  # Buffer minimo per ridurre latenza
    result_queue = Queue(maxsize=1)  # Buffer minimo per output immediato
    running = True
    model_lock = threading.Lock()

    # Impostazioni monitoraggio performance
    last_fps_check = time.time()
    frame_count = 0
    fps = 0

    # Rileva dispositivo di calcolo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # Ottimizzazioni per macOS
    is_macos = platform.system() == 'Darwin'
    if is_macos:
        print("Ottimizzazioni specifiche per macOS attivate")
        # Limitiamo la risoluzione iniziale su macOS per migliorare prestazioni
        if shared_state.processing_resolution.value > 640:
            shared_state.processing_resolution.value = 640
            print(f"Risoluzione iniziale ridotta a {shared_state.processing_resolution.value} per macOS")

    # Ottimizzazioni PyTorch
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)
    else:
        import multiprocessing
        torch.set_num_threads(max(1, multiprocessing.cpu_count() - 1))

    # Apri webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore apertura webcam")
        return

    # Imposta risoluzione webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam: {native_width}x{native_height}")

    # Carica modello
    model = load_model(shared_state.current_model.value, device)

    # Funzione worker per elaborazione frame
    def processing_worker():
        nonlocal running
        while running:
            try:
                if frame_queue.empty():
                    time.sleep(0.001)
                    continue

                frame = frame_queue.get()
                processed = process_frame(
                    frame,
                    model,
                    device,
                    model_lock,
                    shared_state.processing_resolution.value
                )

                if not result_queue.full():
                    result_queue.put(processed)

                frame_queue.task_done()
            except Exception as e:
                print(f"❌ Errore worker: {e}")

    # Monitor thread per comandi GUI
    def monitor_thread():
        nonlocal model, running, fps

        while running:
            try:
                if not shared_state.command_queue.empty():
                    cmd, value = shared_state.command_queue.get()

                    if cmd == "change_model":
                        with model_lock:
                            print(f"Cambio modello a: {value}")
                            model = load_model(value, device)
                            shared_state.status_queue.put(("model_changed", True))

                    elif cmd == "change_resolution":
                        print(f"⚙️ Risoluzione: {value}")
                        shared_state.processing_resolution.value = value

                    elif cmd == "exit":
                        running = False

                # Invia FPS alla GUI per monitoraggio
                if fps > 0:
                    shared_state.status_queue.put(("fps_update", fps))

            except Exception as e:
                print(f"❌ Errore monitor: {e}")

            time.sleep(0.1)

    # Avvia interfaccia grafica
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

    # Avvia thread di monitoraggio
    monitor = threading.Thread(target=monitor_thread)
    monitor.daemon = True
    monitor.start()

    # Avvia thread di elaborazione
    num_workers = 1  # Thread singolo per evitare contesa
    workers = []
    for _ in range(num_workers):
        worker = threading.Thread(target=processing_worker)
        worker.daemon = True
        worker.start()
        workers.append(worker)

    shared_state.running.value = True
    print("🎥 Premi 'q' per uscire")

    # Loop principale applicazione
    processed_frame = None
    skip_frames = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            if running:
                print("⚠️ Errore lettura webcam")
            break

        frame_count += 1

        # Gestione frame skip per mantenere reattività
        if skip_frames > 0:
            skip_frames -= 1
            continue

        # Metti frame in coda solo se non piena
        if not frame_queue.full():
            frame_queue.put(frame)
        else:
            # Se coda piena, skippa prossimi 2 frame per recuperare
            skip_frames = 2

        # Mostra frame elaborato
        if not result_queue.empty():
            processed_frame = result_queue.get()

        if processed_frame is not None:
            cv2.imshow("Cartoonizer", processed_frame)

        # Calcola FPS
        current_time = time.time()
        if current_time - last_fps_check >= 1.0:
            fps = frame_count / (current_time - last_fps_check)
            frame_count = 0
            last_fps_check = current_time

            # Debug FPS
            print(f"📊 FPS: {fps:.1f}")

            # Aggiustamento automatico risoluzione per macOS
            if is_macos and fps < 8 and shared_state.processing_resolution.value > 512:
                new_res = max(512, shared_state.processing_resolution.value - 64)
                shared_state.processing_resolution.value = new_res
                print(f"⬇️ Risoluzione ridotta a {new_res} per migliorare prestazioni")

        # Input utente
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord('-'):
            new_res = max(512, shared_state.processing_resolution.value - 64)
            shared_state.command_queue.put(("change_resolution", new_res))
        elif key == ord('+') or key == ord('='):
            new_res = min(1024, shared_state.processing_resolution.value + 64)
            shared_state.command_queue.put(("change_resolution", new_res))

    # Pulizia
    running = False
    shared_state.running.value = False
    cap.release()
    cv2.destroyAllWindows()

    # Libera memoria
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    print("👋 Applicazione terminata")
    sys.exit(0)


if __name__ == "__main__":
    mp.freeze_support()
    check_dependencies()

    from shared_state import create_shared_state

    shared_state = create_shared_state()

    main(shared_state)
