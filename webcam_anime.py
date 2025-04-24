import cv2
import torch
from PIL import Image
import numpy as np
import os
import platform
import time
import threading
from queue import Queue
import gc
import customtkinter as ctk
import tkinter as tk
import sys

# All'inizio del file, dopo gli import esistenti
if platform.system() == "Darwin":  # macOS
    try:
        import pygame
        import pygame.camera
        from pygame.locals import DOUBLEBUF, QUIT

        pygame.init()
    except ImportError:
        print("Pygame non trovato su macOS. Installa con: pip install pygame")

# Controlla se è disponibile la GPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
else:
    device = "cpu"
print(f"Utilizzando: {device}")

# Importa in base al sistema operativo
IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

if IS_MACOS:
    try:
        import pygame
        import pygame.camera
        from pygame.locals import *

        pygame.init()

        # Prova ad importare PySyphon per macOS
        try:
            from pysyphon import server

            syphon_available = True
        except ImportError:
            syphon_available = False
            print("PySyphon non trovato. Installa con: pip install PySyphon")
    except ImportError:
        print("Pygame non trovato. Installa con: pip install pygame")
else:
    # Su Windows, usa pyvirtualcam standard
    try:
        import pyvirtualcam

        virtual_cam_available = True
    except ImportError:
        virtual_cam_available = False
        print("pyvirtualcam non trovato. Installa con: pip install pyvirtualcam")

# Lista dei modelli disponibili
modelli_disponibili = [
    "face_paint_512_v1",
    "face_paint_512_v2",
    "celeba_distill",
    "paprika"
]

# Variabili globali
running = False
model = None
current_model_name = ""
frame_queue = Queue(maxsize=2)
processed_frame = None
process_frame_lock = threading.Lock()
skip_frames = 1
processing_resolution = 384
show_preview = True
syphon_server = None


# Funzione per individuare le sorgenti video
def trova_sorgenti_video():
    sorgenti = {}

    if IS_MACOS:
        # Su macOS, usa AVFoundation attraverso OpenCV
        for i in range(10):  # Controlla fino a 10 possibili dispositivi
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # Ottieni il nome del dispositivo (quando possibile)
                    name = f"Camera {i}"
                    vendor_info = cap.getBackendName()
                    if vendor_info and len(vendor_info) > 0:
                        name = f"{vendor_info} {i}"
                    sorgenti[name] = i
                cap.release()

        # Controlla anche i device specifici di AVFoundation (per schede di acquisizione)
        avf_indexes = [900 + i for i in range(5)]  # Indici AVFoundation specifici
        for i in avf_indexes:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    sorgenti[f"Dispositivo AVF {i - 900}"] = i
                cap.release()
    else:
        # Su Windows/Linux, usa l'approccio standard
        for indice in range(10):  # Controlla fino a 10 dispositivi
            cap = cv2.VideoCapture(indice)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    nome = f"Sorgente {indice}"
                    if indice == 0:
                        nome = "Webcam principale"
                    sorgenti[nome] = indice
                cap.release()

    if len(sorgenti) == 0:
        print("Nessuna sorgente video trovata!")
    else:
        print(f"Trovate {len(sorgenti)} sorgenti video")

    return sorgenti


# Funzione per caricare il modello
def carica_modello(nome_modello):
    global model

    # Libera memoria prima di caricare un nuovo modello
    if model is not None:
        del model
        torch.cuda.empty_cache() if device == "cuda" else gc.collect()

    print(f"Caricamento del modello: {nome_modello}")

    try:
        model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",
                               pretrained=nome_modello).to(device).eval()
        return model
    except Exception as e:
        print(f"ERRORE nel caricamento del modello: {e}")
        return carica_modello("face_paint_512_v2") if nome_modello != "face_paint_512_v2" else None


# Ridimensiona l'output alla dimensione originale
def resize_without_padding(image, target_width, target_height):
    return cv2.resize(image, (target_width, target_height))


# Thread che acquisisce i frame dalla webcam
def acquisizione_thread(sorgente_id):
    global running, frame_queue

    cap = cv2.VideoCapture(sorgente_id)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire la sorgente video {sorgente_id}")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Errore nella lettura del frame")
            time.sleep(0.1)
            continue

        # Aggiungi un frame alla coda solo se non è piena
        if not frame_queue.full():
            frame_queue.put(frame)

        # Limita la frequenza di acquisizione
        time.sleep(0.01)

    cap.release()


# Thread che elabora i frame
def elaborazione_thread(width, height):
    global running, processed_frame, model, frame_queue, current_model_name

    face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)

    frame_count = 0
    last_process_time = 0

    while running:
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        # Preleva un frame dalla coda
        frame = frame_queue.get()

        # Decidi se elaborare questo frame
        current_time = time.time()
        should_process = (frame_count % (skip_frames + 1) == 0) or (current_time - last_process_time > 0.5)

        if should_process:
            # Ridimensiona per l'elaborazione
            h, w = frame.shape[:2]
            aspect = w / h
            if w > h:
                new_w, new_h = int(processing_resolution * aspect), processing_resolution
            else:
                new_w, new_h = processing_resolution, int(processing_resolution / aspect)

            small_frame = cv2.resize(frame, (new_w, new_h))

            # Converti il frame da BGR a RGB
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            try:
                # Applica la trasformazione in stile anime
                with torch.no_grad():
                    anime_pil = face2paint(model, img_pil, size=processing_resolution)

                # Converti l'immagine risultante da PIL a formato OpenCV
                anime_frame = cv2.cvtColor(np.array(anime_pil), cv2.COLOR_RGB2BGR)

                # Ridimensiona l'output alla dimensione originale
                anime_frame = resize_without_padding(anime_frame, width, height)

                # Aggiorna il frame elaborato con lock per thread safety
                with process_frame_lock:
                    processed_frame = anime_frame.copy()

                last_process_time = current_time

            except Exception as e:
                print(f"Errore nell'elaborazione del frame: {e}")

        frame_count += 1
        frame_queue.task_done()


# Funzione che gestisce la fotocamera virtuale
def avvia_fotocamera_virtuale(sorgente_id, modello_iniziale):
    global running, current_model_name, processed_frame, model, syphon_server

    try:
        model = carica_modello(modello_iniziale)
        current_model_name = modello_iniziale

        cap = cv2.VideoCapture(sorgente_id)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire la sorgente video {sorgente_id}")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        cap.release()

        running = True
        acq_thread = threading.Thread(target=acquisizione_thread, args=(sorgente_id,))
        elab_thread = threading.Thread(target=elaborazione_thread, args=(width, height))
        acq_thread.daemon = True
        elab_thread.daemon = True
        acq_thread.start()
        elab_thread.start()

        # Attendi che il primo frame sia elaborato
        while processed_frame is None and running:
            time.sleep(0.1)

        if IS_MACOS:
            # Su macOS, usa Syphon + finestra di anteprima Pygame
            if syphon_available:
                pygame.display.set_mode((width, height), DOUBLEBUF)
                syphon_server = server.SyphonServer("AnimeGAN2", width, height)

                print("Server Syphon avviato: 'AnimeGAN2'")
                print("Per utilizzare la fotocamera virtuale:")
                print("1. Apri OBS Studio")
                print("2. Aggiungi una sorgente 'Syphon Client' e seleziona 'AnimeGAN2'")
                print("3. Avvia OBS Virtual Camera")

                # Inizializza lo schermo pygame
                screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption('AnimeGAN Preview')

                clock = pygame.time.Clock()

                while running:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            running = False

                    with process_frame_lock:
                        if processed_frame is not None:
                            # Converti il frame OpenCV in una surface pygame
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            pygame_frame = pygame.image.frombuffer(frame_rgb.tobytes(),
                                                                   (width, height),
                                                                   'RGB')

                            # Mostra l'anteprima se richiesto
                            if show_preview:
                                screen.blit(pygame_frame, (0, 0))
                                pygame.display.flip()

                            # Invia al server Syphon
                            syphon_server.publish_frame(pygame_frame)

                    clock.tick(fps)
            else:
                # Fallback a una semplice finestra di anteprima se Syphon non è disponibile
                print("PySyphon non disponibile. Mostro solo l'anteprima.")
                while running:
                    with process_frame_lock:
                        if processed_frame is not None:
                            cv2.imshow('AnimeGAN Preview (Cattura questa finestra)', processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        running = False
                    time.sleep(1.0 / fps)

        else:
            # Su Windows/Linux usa pyvirtualcam
            try:
                with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
                    print(f"Fotocamera virtuale avviata: {cam.device}")
                    while running:
                        with process_frame_lock:
                            if processed_frame is not None:
                                current_frame = processed_frame.copy()
                            else:
                                current_frame = np.zeros((height, width, 3), dtype=np.uint8)

                        cam.send(current_frame)
                        if show_preview:
                            cv2.imshow('AnimeGAN Preview', current_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        cam.sleep_until_next_frame()
            except Exception as e:
                print(f"Errore con la fotocamera virtuale: {e}")
                if show_preview:
                    while running:
                        with process_frame_lock:
                            if processed_frame is not None:
                                cv2.imshow('AnimeGAN Preview (Cattura questa finestra)', processed_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            running = False
                        time.sleep(1.0 / fps)

        return True

    except Exception as e:
        print(f"Errore nell'avvio della fotocamera virtuale: {e}")
        return False

    finally:
        if show_preview:
            cv2.destroyAllWindows()
        if IS_MACOS and syphon_server:
            syphon_server.stop()


# Funzione per cambiare modello durante l'esecuzione
def cambia_modello(nome_modello):
    global current_model_name, model
    model = carica_modello(nome_modello)
    current_model_name = nome_modello


# Interfaccia grafica principale
def crea_interfaccia_grafica():
    global sorgenti_disponibili, running, skip_frames, processing_resolution, show_preview

    # Configura l'aspetto di CustomTkinter
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")

    # Trova le sorgenti video disponibili
    sorgenti_disponibili = trova_sorgenti_video()
    if not sorgenti_disponibili:
        sorgenti_disponibili = {}

    # Crea la finestra principale
    root = ctk.CTk()
    root.title("Code22 Ai Camera")
    root.geometry("600x720")
    root.minsize(500, 600)
    root.configure(fg_color="#000000")

    # Crea un frame scrollable
    container = ctk.CTkFrame(root, fg_color="#000000")
    container.pack(fill="both", expand=True)

    canvas = ctk.CTkCanvas(container, bg="#000000", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = ctk.CTkScrollbar(container, orientation="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    scroll_frame = ctk.CTkFrame(canvas, fg_color="#000000")
    scroll_frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(scroll_frame_id, width=event.width)

    scroll_frame.bind("<Configure>", configure_scroll_region)
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(scroll_frame_id, width=e.width))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    # Card per sorgente video
    source_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020",
                               border_width=1, border_color="#303030")
    source_card.pack(padx=20, pady=10, fill="x")

    source_title = ctk.CTkLabel(source_card, text="Sorgente Video",
                                font=ctk.CTkFont(size=18, weight="bold"))
    source_title.pack(anchor="w", padx=20, pady=(15, 10))

    sorgente_var = ctk.StringVar()
    if sorgenti_disponibili:
        sorgente_var.set(list(sorgenti_disponibili.keys())[0])

    source_inner_frame = ctk.CTkFrame(source_card, fg_color="transparent")
    source_inner_frame.pack(fill="x", padx=20, pady=(0, 15))

    for nome_sorgente in sorgenti_disponibili:
        source_option = ctk.CTkRadioButton(source_inner_frame, text=nome_sorgente,
                                           value=nome_sorgente, variable=sorgente_var,
                                           border_width_checked=2,
                                           fg_color="#0066ff",
                                           hover_color="#0052cc")
        source_option.pack(anchor="w", padx=15, pady=8)

    # Aggiungi pulsante di scansione sorgenti
    scan_btn = ctk.CTkButton(source_card, text="Scansiona sorgenti video",
                             height=30, corner_radius=8,
                             font=ctk.CTkFont(size=12),
                             fg_color="#333333",
                             hover_color="#444444")
    scan_btn.pack(padx=20, pady=(0, 15), fill="x")

    # Card per modelli
    model_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020",
                              border_width=1, border_color="#303030")
    model_card.pack(padx=20, pady=10, fill="x")

    model_title = ctk.CTkLabel(model_card, text="Modello AnimeGAN",
                               font=ctk.CTkFont(size=18, weight="bold"))
    model_title.pack(anchor="w", padx=20, pady=(15, 10))

    modello_var = ctk.StringVar()
    modello_var.set(modelli_disponibili[1])  # Default a face_paint_512_v2

    model_inner_frame = ctk.CTkFrame(model_card, fg_color="transparent")
    model_inner_frame.pack(fill="x", padx=20, pady=(0, 15))

    for modello in modelli_disponibili:
        model_option = ctk.CTkRadioButton(model_inner_frame, text=modello,
                                          value=modello, variable=modello_var,
                                          border_width_checked=2,
                                          fg_color="#0066ff",
                                          hover_color="#0052cc")
        model_option.pack(anchor="w", padx=15, pady=8)

    # Card per impostazioni prestazioni
    perf_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020",
                             border_width=1, border_color="#303030")
    perf_card.pack(padx=20, pady=10, fill="x")

    perf_title = ctk.CTkLabel(perf_card, text="Impostazioni Prestazioni",
                              font=ctk.CTkFont(size=18, weight="bold"))
    perf_title.pack(anchor="w", padx=20, pady=(15, 10))

    # Frame skip slider
    skip_frame = ctk.CTkFrame(perf_card, fg_color="transparent")
    skip_frame.pack(fill="x", padx=20, pady=5)

    skip_label = ctk.CTkLabel(skip_frame, text="Frame da saltare:")
    skip_label.pack(side="left", padx=(10, 0))

    skip_value = ctk.CTkLabel(skip_frame, text=str(skip_frames), width=30)
    skip_value.pack(side="right", padx=(0, 10))

    frame_skip_slider = ctk.CTkSlider(perf_card, from_=0, to=5, number_of_steps=5,
                                      progress_color="#0066ff",
                                      button_color="#0066ff",
                                      button_hover_color="#0052cc")
    frame_skip_slider.set(skip_frames)
    frame_skip_slider.pack(fill="x", padx=30, pady=(0, 10))

    def update_skip_label(val):
        global skip_frames
        skip_frames = int(val)
        skip_value.configure(text=str(skip_frames))

    frame_skip_slider.configure(command=update_skip_label)

    # Resolution slider
    res_frame = ctk.CTkFrame(perf_card, fg_color="transparent")
    res_frame.pack(fill="x", padx=20, pady=5)

    res_label = ctk.CTkLabel(res_frame, text="Risoluzione di elaborazione:")
    res_label.pack(side="left", padx=(10, 0))

    res_value = ctk.CTkLabel(res_frame, text=str(processing_resolution), width=30)
    res_value.pack(side="right", padx=(0, 10))

    resolution_slider = ctk.CTkSlider(perf_card, from_=128, to=512, number_of_steps=6,
                                      progress_color="#0066ff",
                                      button_color="#0066ff",
                                      button_hover_color="#0052cc")
    resolution_slider.set(processing_resolution)
    resolution_slider.pack(fill="x", padx=30, pady=(0, 10))

    def update_res_label(val):
        global processing_resolution
        val_rounded = round(val / 64) * 64
        processing_resolution = int(val_rounded)
        res_value.configure(text=str(processing_resolution))

    resolution_slider.configure(command=update_res_label)

    # Preview switch
    preview_frame = ctk.CTkFrame(perf_card, fg_color="transparent")
    preview_frame.pack(fill="x", padx=20, pady=(5, 15))

    preview_var = ctk.BooleanVar(value=show_preview)
    preview_switch = ctk.CTkSwitch(preview_frame, text="Mostra anteprima",
                                   variable=preview_var,
                                   progress_color="#0066ff",
                                   button_color="#0066ff",
                                   button_hover_color="#0052cc")
    preview_switch.pack(anchor="w", padx=10, pady=5)

    def update_preview():
        global show_preview
        show_preview = preview_var.get()

    preview_switch.configure(command=update_preview)

    # Card per cambio modello
    switch_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020",
                               border_width=1, border_color="#303030")
    switch_card.pack(padx=20, pady=10, fill="x")

    switch_title = ctk.CTkLabel(switch_card, text="Cambio Modello in Tempo Reale",
                                font=ctk.CTkFont(size=18, weight="bold"))
    switch_title.pack(anchor="w", padx=20, pady=(15, 10))

    bottoni_cambio_modello = []
    buttons_grid = ctk.CTkFrame(switch_card, fg_color="transparent")
    buttons_grid.pack(padx=20, pady=(5, 15), fill="x")

    for i, modello in enumerate(modelli_disponibili):
        btn = ctk.CTkButton(buttons_grid, text=modello,
                            command=lambda m=modello: cambia_modello(m),
                            state="disabled", height=36,
                            corner_radius=8,
                            fg_color="#2b2b2b",
                            hover_color="#353535")
        btn.grid(row=i // 2, column=i % 2, padx=8, pady=8, sticky="ew")
        bottoni_cambio_modello.append(btn)

    buttons_grid.columnconfigure(0, weight=1)
    buttons_grid.columnconfigure(1, weight=1)

    # Card per info sistema
    info_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020",
                             border_width=1, border_color="#303030")
    info_card.pack(padx=20, pady=10, fill="x")

    info_title = ctk.CTkLabel(info_card, text="Informazioni Sistema",
                              font=ctk.CTkFont(size=18, weight="bold"))
    info_title.pack(anchor="w", padx=20, pady=(15, 10))

    os_name = platform.system()
    gpu_info = "CUDA" if device == "cuda" else "MPS (Apple Silicon)" if device == "mps" else "CPU"

    info_text = (f"Sistema: {os_name}\n"
                 f"Dispositivo: {gpu_info}\n")

    if IS_MACOS:
        info_text += "\nModalità macOS: Utilizzo Syphon + OBS\n"
        if not syphon_available:
            info_text += "ATTENZIONE: PySyphon non installato! Solo anteprima disponibile.\n"
            info_text += "Per installare: pip install PySyphon\n"
    elif not virtual_cam_available:
        info_text += "\nATTENZIONE: pyvirtualcam non installato!\n"
        info_text += "Per installare: pip install pyvirtualcam\n"

    info_label = ctk.CTkLabel(info_card, text=info_text, justify="left")
    info_label.pack(anchor="w", padx=20, pady=(5, 15))

    # Pulsanti principali
    footer_frame = ctk.CTkFrame(root, fg_color="transparent", height=80)
    footer_frame.pack(fill="x", side="bottom")

    btn_frame = ctk.CTkFrame(footer_frame, fg_color="transparent")
    btn_frame.pack(pady=(2, 5), padx=20, fill="x")

    avvia_btn = ctk.CTkButton(btn_frame, text="Avvia Fotocamera Virtuale",
                              height=34, corner_radius=8,
                              font=ctk.CTkFont(size=14, weight="bold"),
                              fg_color="#0066ff",
                              hover_color="#0052cc")
    avvia_btn.pack(side="left", padx=5, expand=True, fill="x")

    stop_btn = ctk.CTkButton(btn_frame, text="Ferma",
                             height=34, corner_radius=8,
                             font=ctk.CTkFont(size=14, weight="bold"),
                             fg_color="#ff0033",
                             hover_color="#cc0029",
                             state="disabled")
    stop_btn.pack(side="right", padx=5, expand=True, fill="x")

    # Testo "powered by code22" in basso
    footer_label = ctk.CTkLabel(footer_frame, text="powered by code22",
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color="#888888")
    footer_label.pack(side="bottom", pady=(0, 2))

    def rescansiona_sorgenti():
        global sorgenti_disponibili

        # Rimuovi i vecchi radiobutton
        for widget in source_inner_frame.winfo_children():
            widget.destroy()

        # Trova nuove sorgenti
        sorgenti_disponibili = trova_sorgenti_video()

        # Ricostruisci i radiobutton
        if sorgenti_disponibili:
            sorgente_var.set(list(sorgenti_disponibili.keys())[0])
            for nome_sorgente in sorgenti_disponibili:
                source_option = ctk.CTkRadioButton(source_inner_frame, text=nome_sorgente,
                                                   value=nome_sorgente, variable=sorgente_var,
                                                   border_width_checked=2,
                                                   fg_color="#0066ff",
                                                   hover_color="#0052cc")
                source_option.pack(anchor="w", padx=15, pady=8)
        else:
            # Mostra messaggio se non ci sono sorgenti disponibili
            no_source_label = ctk.CTkLabel(source_inner_frame, text="Nessuna sorgente video trovata")
            no_source_label.pack(anchor="w", padx=15, pady=8)

    # Collega il pulsante di scansione alla funzione
    scan_btn.configure(command=rescansiona_sorgenti)

    def avvia():
        global running

        if not sorgenti_disponibili:
            messagebox = ctk.CTkMessageBox(title="Errore",
                                           message="Nessuna sorgente video disponibile.",
                                           icon="cancel")
            return

        sorgente = sorgenti_disponibili[sorgente_var.get()]
        modello = modello_var.get()

        # Disabilita il pulsante di avvio e abilita quello di stop
        avvia_btn.configure(state="disabled")
        stop_btn.configure(state="normal")

        # Abilita i pulsanti di cambio modello
        for btn in bottoni_cambio_modello:
            btn.configure(state="normal")

        # Avvia in un thread separato per non bloccare l'interfaccia
        threading.Thread(target=lambda: avvia_fotocamera_virtuale(sorgente, modello),
                         daemon=True).start()

    def ferma():
        global running, processed_frame, syphon_server

        running = False
        processed_frame = None

        # Ripristina l'interfaccia
        stop_btn.configure(state="disabled")
        avvia_btn.configure(state="normal")

        # Disabilita i pulsanti di cambio modello
        for btn in bottoni_cambio_modello:
            btn.configure(state="disabled")

        # Assicuriamoci che le risorse siano rilasciate
        if IS_MACOS and syphon_server:
            syphon_server = None

        # Pulizia memoria
        if 'model' in globals() and model is not None:
            torch.cuda.empty_cache() if device == "cuda" else gc.collect()

    # Collega i pulsanti alle rispettive funzioni
    avvia_btn.configure(command=avvia)
    stop_btn.configure(command=ferma)

    # Funzione da eseguire alla chiusura dell'app
    def on_closing():
        global running
        if running:
            ferma()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Avvia il loop principale
    root.mainloop()


if __name__ == "__main__":
    crea_interfaccia_grafica()
