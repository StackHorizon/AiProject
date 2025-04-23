import cv2
import torch
from PIL import Image
import numpy as np
import os
import platform
import time
import pyvirtualcam
from model import Generator
import tkinter as tk
from tkinter import ttk, Scale
import threading
from queue import Queue
import gc

# Controlla se è disponibile la GPU
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
else:
    device = "cpu"
print(f"Utilizzando: {device}")

# Lista dei modelli disponibili
modelli_disponibili = [
    "face_paint_512_v1",
    "face_paint_512_v2",
    "celeba_distill",
    "paprika"
]

# Lista delle possibili sorgenti video
sorgenti_disponibili = {}

# Variabili globali
running = False
model = None
current_model_name = ""
frame_queue = Queue(maxsize=2)  # Buffer limitato per evitare memory leak
processed_frame = None
process_frame_lock = threading.Lock()
skip_frames = 1  # Numero di frame da saltare (0 = processa tutti i frame)
processing_resolution = 384  # Risoluzione di processing ridotta
show_preview = True  # Mostra anteprima (disattivare per prestazioni migliori)


# Funzione per individuare le sorgenti video disponibili
def trova_sorgenti_video():
    def trova_sorgenti_video():
        sorgenti = {}

        # Prima prova le sorgenti standard (0, 1, 2)
        for indice in range(3):  # Limita a 3 sorgenti massime
            cap = cv2.VideoCapture(indice)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    nome = f"Sorgente {indice}"
                    if indice == 0:
                        nome = "Webcam principale"
                    sorgenti[nome] = indice
                cap.release()

        # Se su macOS e nessuna sorgente trovata, prova specificamente l'indice 0
        if len(sorgenti) == 0 and platform.system() == "Darwin":
            print("Tentativo di accesso specifico alla webcam principale...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                sorgenti["Webcam macOS"] = 0
            cap.release()

        # Se non ci sono ancora sorgenti, aggiungi un messaggio
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
        # Rimuovo il blocco per il modello hayao_ghibli
        model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator",
                               pretrained=nome_modello).to(device).eval()
        return model
    except Exception as e:
        print(f"ERRORE nel caricamento del modello: {e}")
        return carica_modello("face_paint_512_v2") if nome_modello != "face_paint_512_v2" else None


# Ridimensiona l'output alla dimensione originale preservando le proporzioni
def resize_without_padding(image, target_width, target_height):
    # Ridimensiona l'immagine per riempire completamente lo spazio target
    # senza mantenere le proporzioni originali
    return cv2.resize(image, (target_width, target_height))  # Thread che acquisisce i frame dalla webcam


def acquisizione_thread(sorgente_id):
    global running, frame_queue

    cap = cv2.VideoCapture(sorgente_id)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire la sorgente video {sorgente_id}")
        return

    frame_count = 0

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Errore nella lettura del frame")
            time.sleep(0.1)
            continue

        # Aggiungi un frame alla coda solo se non è piena
        if not frame_queue.full():
            frame_queue.put(frame)

        frame_count += 1
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

        # Decidi se elaborare questo frame in base al contatore
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
                with torch.no_grad():  # Disabilita il calcolo del gradiente per risparmiare memoria
                    anime_pil = face2paint(model, img_pil, size=processing_resolution)

                # Converti l'immagine risultante da PIL a formato OpenCV
                anime_frame = cv2.cvtColor(np.array(anime_pil), cv2.COLOR_RGB2BGR)

                # Ridimensiona l'output alla dimensione originale
                anime_frame = resize_without_padding(anime_frame, width, height)
                # Aggiungi info sul modello
                # cv2.putText(anime_frame, f"Modello: {current_model_name}", (10, 30),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Aggiorna il frame elaborato con lock per thread safety
                with process_frame_lock:
                    processed_frame = anime_frame.copy()

                last_process_time = current_time

            except Exception as e:
                print(f"Errore nell'elaborazione del frame: {e}")

        frame_count += 1
        frame_queue.task_done()


# Funzione principale che gestisce la fotocamera virtuale
def avvia_fotocamera_virtuale(sorgente_id, modello_iniziale):
    global running, current_model_name, processed_frame, model, skip_frames, processing_resolution, show_preview

    # Inizializza il modello
    model = carica_modello(modello_iniziale)
    current_model_name = modello_iniziale

    # Ottieni la dimensione del video
    cap = cv2.VideoCapture(sorgente_id)
    if not cap.isOpened():
        print(f"Errore: Impossibile aprire la sorgente video {sorgente_id}")
        messagebox.showerror("Errore",
                             f"Impossibile aprire la sorgente video {sorgente_id}.\nAssicurati che la webcam sia collegata e non in uso da altre applicazioni.")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    cap.release()

    # Inizia i thread per acquisizione ed elaborazione
    running = True
    acq_thread = threading.Thread(target=acquisizione_thread, args=(sorgente_id,))
    elab_thread = threading.Thread(target=elaborazione_thread, args=(width, height))

    acq_thread.daemon = True
    elab_thread.daemon = True

    acq_thread.start()
    elab_thread.start()

    # Aspetta che il primo frame sia pronto
    while processed_frame is None and running:
        time.sleep(0.1)

    # Crea la fotocamera virtuale
    try:
        with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
            print(f"Fotocamera virtuale avviata: {cam.device}")

            while running:
                # Invia il frame elaborato alla fotocamera virtuale
                with process_frame_lock:
                    if processed_frame is not None:
                        current_frame = processed_frame.copy()
                    else:
                        # Se non c'è un frame elaborato, usa un frame nero
                        current_frame = np.zeros((height, width, 3), dtype=np.uint8)

                # Invia il frame alla fotocamera virtuale
                cam.send(current_frame)

                # Visualizza anteprima se attivata
                if show_preview:
                    cv2.imshow('AnimeGAN Preview', current_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Sincronizza con la frequenza della fotocamera virtuale
                cam.sleep_until_next_frame()

    except Exception as e:
        print(f"Errore con la fotocamera virtuale: {e}")

    # Pulisci risorse
    if show_preview:
        cv2.destroyAllWindows()

    # Reset il frame elaborato
    with process_frame_lock:
        processed_frame = None


# Funzione per cambiare modello durante l'esecuzione
def cambia_modello(nome_modello):
    global current_model_name, model
    model = carica_modello(nome_modello)
    current_model_name = nome_modello


# Interfaccia grafica per la selezione della sorgente e del modello
def crea_interfaccia_grafica():
    global sorgenti_disponibili, running, skip_frames, processing_resolution, show_preview

    # Importa CustomTkinter
    import customtkinter as ctk
    import tkinter as tk

    # Configura l'aspetto di CustomTkinter
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")

    # Trova le sorgenti video disponibili
    sorgenti_disponibili = trova_sorgenti_video()

    # Crea la finestra principale
    root = ctk.CTk()
    root.title("Code22 Ai Camera")
    root.geometry("600x700")
    root.minsize(500, 600)

    # Imposta un colore di sfondo nero OLED
    root.configure(fg_color="#000000")

    # Crea un frame scrollable
    container = ctk.CTkFrame(root, fg_color="#000000")
    container.pack(fill="both", expand=True)

    # Crea un canvas per lo scrolling
    canvas = ctk.CTkCanvas(container, bg="#000000", highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    # Aggiungi scrollbar
    scrollbar = ctk.CTkScrollbar(container, orientation="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Frame interno scrollabile
    scroll_frame = ctk.CTkFrame(canvas, fg_color="#000000")
    scroll_frame_id = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

    # Funzione per adattare il canvas al contenuto
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(scroll_frame_id, width=event.width)

    scroll_frame.bind("<Configure>", configure_scroll_region)

    # Aggiorna la dimensione del canvas in base alla dimensione della finestra
    def on_canvas_configure(event):
        canvas.itemconfig(scroll_frame_id, width=event.width)

    canvas.bind("<Configure>", on_canvas_configure)

    # Abilita lo scrolling con rotellina del mouse
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # Card per sorgente video
    source_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020", border_width=1,
                               border_color="#303030")
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

    # Card per modelli
    model_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020", border_width=1,
                              border_color="#303030")
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
    perf_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020", border_width=1,
                             border_color="#303030")
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

    # Update skip value label when slider changes
    def update_skip_label(val):
        skip_value.configure(text=str(int(val)))
        setattr(__main__, 'skip_frames', int(val))

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

    # Update resolution label when slider changes
    def update_res_label(val):
        val_rounded = round(val / 64) * 64
        res_value.configure(text=str(int(val_rounded)))
        setattr(__main__, 'processing_resolution', int(val_rounded))

    resolution_slider.configure(command=update_res_label)

    # Preview switch
    preview_frame = ctk.CTkFrame(perf_card, fg_color="transparent")
    preview_frame.pack(fill="x", padx=20, pady=(5, 15))

    preview_var = ctk.BooleanVar(value=show_preview)
    preview_switch = ctk.CTkSwitch(preview_frame, text="Mostra anteprima",
                                   variable=preview_var,
                                   progress_color="#0066ff",
                                   button_color="#0066ff",
                                   button_hover_color="#0052cc",
                                   command=lambda: setattr(__main__, 'show_preview', preview_var.get()))
    preview_switch.pack(anchor="w", padx=10, pady=5)

    # Card per cambio modello
    switch_card = ctk.CTkFrame(scroll_frame, corner_radius=15, fg_color="#202020", border_width=1,
                               border_color="#303030")
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

    # Configura le colonne in modo che abbiano lo stesso peso
    buttons_grid.columnconfigure(0, weight=1)
    buttons_grid.columnconfigure(1, weight=1)

    # Aggiungi spaziatura dopo le card
    ctk.CTkFrame(scroll_frame, fg_color="#000000", height=20).pack(fill="x")

    # Footer collassabile
    footer_visible = tk.BooleanVar(value=True)

    def toggle_footer():
        if footer_visible.get():
            footer_content.pack_forget()
            toggle_btn.configure(text="▲")
            footer_frame.configure(height=25)
        else:
            footer_content.pack(fill="x", expand=True)
            toggle_btn.configure(text="▼")
            footer_frame.configure(height=80)
        footer_visible.set(not footer_visible.get())

    # Footer trasparente
    footer_frame = ctk.CTkFrame(root, corner_radius=0, fg_color="transparent", height=80)
    footer_frame.pack(fill="x", side="bottom")
    footer_frame.pack_propagate(False)

    # Toggle button per nascondere/mostrare
    toggle_btn = ctk.CTkButton(footer_frame, text="▼", width=30, height=20,
                               fg_color="#333333", hover_color="#444444",
                               corner_radius=0)
    toggle_btn.configure(command=toggle_footer)
    toggle_btn.pack(anchor="center", pady=(0, 1))

    # Contenuto del footer
    footer_content = ctk.CTkFrame(footer_frame, fg_color="transparent")
    footer_content.pack(fill="x", expand=True)

    # Pulsanti principali con migliore contrasto
    btn_frame = ctk.CTkFrame(footer_content, fg_color="transparent")
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
    footer_label = ctk.CTkLabel(footer_content, text="powered by code22",
                                font=ctk.CTkFont(size=12, weight="bold"),
                                text_color="#888888")
    footer_label.pack(side="bottom", pady=(0, 2))

    # Funzione per avviare la fotocamera virtuale
    def avvia():
        sorgente_selezionata = sorgente_var.get()
        modello_selezionato = modello_var.get()

        if sorgente_selezionata in sorgenti_disponibili:
            sorgente_id = sorgenti_disponibili[sorgente_selezionata]
            # Avvia la fotocamera virtuale in un thread separato
            thread = threading.Thread(target=avvia_fotocamera_virtuale, args=(sorgente_id, modello_selezionato))
            thread.daemon = True
            thread.start()

            # Abilita il cambio modello durante l'esecuzione
            for btn in bottoni_cambio_modello:
                btn.configure(state="normal", fg_color="#0066ff", hover_color="#0052cc")
            avvia_btn.configure(state="disabled")
            stop_btn.configure(state="normal")

            # Disabilita gli slider durante l'esecuzione
            frame_skip_slider.configure(state="disabled")
            resolution_slider.configure(state="disabled")
            preview_switch.configure(state="disabled")

    # Funzione per fermare la fotocamera virtuale
    def ferma():
        global running, processed_frame
        running = False

        # Aspetta un po' per assicurarsi che i thread siano terminati
        time.sleep(0.3)

        # Reimposta frame_queue e processed_frame
        with process_frame_lock:
            processed_frame = None

        while not frame_queue.empty():
            frame_queue.get()
            frame_queue.task_done()

        # Libera memoria
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        avvia_btn.configure(state="normal")
        stop_btn.configure(state="disabled")
        for btn in bottoni_cambio_modello:
            btn.configure(state="disabled", fg_color="#2b2b2b", hover_color="#353535")

        # Riabilita gli slider dopo aver fermato
        frame_skip_slider.configure(state="normal")
        resolution_slider.configure(state="normal")
        preview_switch.configure(state="normal")

    # Collega le funzioni ai pulsanti
    avvia_btn.configure(command=avvia)
    stop_btn.configure(command=ferma)

    # Avvia il loop principale della GUI
    root.mainloop()


if __name__ == "__main__":
    # Prima di avviare, verifica i requisiti
    try:
        import __main__
        import pyvirtualcam
    except ImportError:
        print("Pacchetto 'pyvirtualcam' non installato.")
        print("Installa con: pip install pyvirtualcam")
        print("Potrebbe essere necessario anche installare OBS Virtual Camera")
        exit(1)

    crea_interfaccia_grafica()
