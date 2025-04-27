# gui.py
import threading
import time

import customtkinter as ctk


def setup_font():
    # Importazioni necessarie all'inizio della funzione
    import platform
    import os
    import sys

    system = platform.system()

    if system == "Darwin":  # macOS
        try:
            # Approccio specifico per macOS che non richiede tkextrafont
            os.makedirs("fonts", exist_ok=True)

            # Controlla se il font è già scaricato
            if not os.path.exists("fonts/Lexend-Regular.ttf"):
                print("Scaricamento font Lexend per macOS...")
                import requests
                import zipfile
                from io import BytesIO

                url = "https://github.com/googlefonts/lexend/archive/refs/heads/main.zip"
                response = requests.get(url)

                with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                    for file in zip_file.namelist():
                        if file.endswith('.ttf') and "/fonts/ttf/" in file and "Lexend-" in file:
                            font_data = zip_file.read(file)
                            font_name = os.path.basename(file)
                            with open(f"fonts/{font_name}", "wb") as f:
                                f.write(font_data)
                print("✅ Font Lexend scaricato")

            # Su macOS, i font vengono caricati diversamente senza tkextrafont
            print("✅ Font configurati per macOS")
            return

        except Exception as e:
            print(f"Nota: Font Lexend non disponibile su macOS, uso font predefinito: {e}")
            return
    else:
        # Codice originale per Windows e Linux
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "tkextrafont"])
            from tkextrafont import Font

            # Crea directory fonts se non esiste
            os.makedirs("fonts", exist_ok=True)

            # Installa Lexend se non presente
            if not os.path.exists("fonts/Lexend-Regular.ttf"):
                import requests
                import zipfile
                from io import BytesIO

                print("Scaricamento font Lexend...")
                url = "https://github.com/googlefonts/lexend/archive/refs/heads/main.zip"
                response = requests.get(url)

                with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
                    for file in zip_file.namelist():
                        if file.endswith('.ttf') and "/fonts/ttf/" in file and "Lexend-" in file:
                            font_data = zip_file.read(file)
                            font_name = os.path.basename(file)
                            with open(f"fonts/{font_name}", "wb") as f:
                                f.write(font_data)
                print("✅ Font Lexend scaricato")

            # Registra i font
            for font_file in os.listdir("fonts"):
                if font_file.endswith(".ttf") and "Lexend-" in font_file:
                    Font(file=f"fonts/{font_file}", family="Lexend")

        except Exception as e:
            print(f"Nota: Font Lexend non disponibile, uso font predefinito: {e}")


def start_gui(model_list, command_queue, status_queue, initial_model, initial_resolution):
    # Setup font Lexend (opzionale)
    setup_font()

    # Avvia l'interfaccia
    app = AnimeGANControlPanel(model_list, command_queue, status_queue, initial_model, initial_resolution)
    app.run()


class ModernFrame(ctk.CTkFrame):
    """Frame con effetto acrilico simulato"""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        # Colori che simulano l'effetto acrilico
        self.configure(fg_color=("#EBEDF0", "#2B2D32"))  # Simula trasparenza
        self.configure(corner_radius=18)  # Bordi più smussati

        # Linea decorativa superiore
        self.accent_line = ctk.CTkFrame(self, height=4, corner_radius=2,
                                        fg_color=("#6E85B7", "#8A6FDF"))
        self.accent_line.place(relx=0.5, y=0, relwidth=0.25, anchor="n")


class AnimeGANControlPanel:
    def __init__(self, model_list, command_queue, status_queue, current_model, processing_resolution):
        # Configurazione CustomTkinter
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Memorizziamo i parametri
        self.model_list = model_list
        self.command_queue = command_queue
        self.status_queue = status_queue
        self.current_model = current_model
        self.processing_resolution = processing_resolution

        # Finestra principale
        self.root = ctk.CTk()
        self.root.title("Capitan Acciaio Cartoonizer")
        self.root.geometry("480x620")
        self.root.resizable(False, False)

        # Definiamo i font in base al sistema operativo
        import platform
        if platform.system() == "Darwin":  # macOS
            # Font di sistema per macOS
            self.title_font = ctk.CTkFont(family="Arial", size=24, weight="bold")
            self.header_font = ctk.CTkFont(family="Arial", size=18, weight="bold")
            self.text_font = ctk.CTkFont(family="Arial", size=14)
            self.small_font = ctk.CTkFont(family="Arial", size=12)
        else:
            # Font personalizzati per altri sistemi
            self.title_font = ctk.CTkFont(family="Lexend", size=24, weight="bold")
            self.header_font = ctk.CTkFont(family="Lexend", size=18, weight="bold")
            self.text_font = ctk.CTkFont(family="Lexend", size=14)
            self.small_font = ctk.CTkFont(family="Lexend", size=12)

        self.setup_ui()

        # Monitoraggio con thread separato
        self.monitor_thread = threading.Thread(target=self.monitor_status_queue, daemon=True)
        self.monitor_thread.start()

    def setup_ui(self):
        # Titolo con stile moderno
        title_frame = ModernFrame(self.root)
        title_frame.pack(fill="x", padx=20, pady=(20, 0))

        title = ctk.CTkLabel(title_frame, text="Capitan Acciaio Cartoonizer",
                             font=self.title_font)
        title.pack(pady=18)

        subtitle = ctk.CTkLabel(title_frame, text="Powered by Code22",
                                font=self.small_font, text_color="gray")
        subtitle.pack(pady=(0, 15))

        # Container principale
        main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=15)

        # Sezione modelli
        model_frame = ModernFrame(main_frame)
        model_frame.pack(fill="x", padx=10, pady=10)

        # Intestazione con icona
        model_header = ctk.CTkFrame(model_frame, fg_color="transparent")
        model_header.pack(fill="x", padx=15, pady=(15, 5))

        model_icon = ctk.CTkLabel(model_header, text="🎨", font=ctk.CTkFont(size=20))
        model_icon.pack(side="left", padx=(0, 10))

        model_label = ctk.CTkLabel(model_header, text="Seleziona Modello",
                                   font=self.header_font)
        model_label.pack(side="left")

        # Menu a tendina per i modelli
        self.model_var = ctk.StringVar(value=self.current_model)
        self.model_dropdown = ctk.CTkComboBox(
            model_frame, values=self.model_list,
            command=self.change_model,
            variable=self.model_var,
            font=self.text_font, button_hover_color="#0066ff",
            dropdown_font=self.text_font,
            height=36
        )
        self.model_dropdown.pack(padx=15, pady=(5, 20), fill="x")

        # Sezione risoluzione
        res_frame = ModernFrame(main_frame)
        res_frame.pack(fill="x", padx=10, pady=10)

        # Intestazione risoluzione
        res_header = ctk.CTkFrame(res_frame, fg_color="transparent")
        res_header.pack(fill="x", padx=15, pady=(15, 5))

        res_icon = ctk.CTkLabel(res_header, text="🔍", font=ctk.CTkFont(size=20))
        res_icon.pack(side="left", padx=(0, 10))

        res_label = ctk.CTkLabel(res_header, text="Risoluzione di Elaborazione",
                                 font=self.header_font)
        res_label.pack(side="left")

        # Display valore risoluzione
        res_value_container = ctk.CTkFrame(res_frame, fg_color=("#EAEAEA", "#333333"), corner_radius=10, height=40)
        res_value_container.pack(fill="x", padx=20, pady=(10, 5))
        res_value_container.pack_propagate(False)

        self.res_value_label = ctk.CTkLabel(res_value_container,
                                            text=f"{self.processing_resolution} px",
                                            font=self.header_font)
        self.res_value_label.pack(fill="both", expand=True)

        # Slider per la risoluzione
        self.res_slider = ctk.CTkSlider(
            res_frame, from_=512, to=1024,
            command=self.update_resolution,
            number_of_steps=8
        )
        self.res_slider.set(self.processing_resolution)
        self.res_slider.pack(padx=20, pady=(15, 15), fill="x")

        # Etichette min/max
        slider_labels = ctk.CTkFrame(res_frame, fg_color="transparent")
        slider_labels.pack(fill="x", padx=20, pady=(0, 5))

        min_label = ctk.CTkLabel(slider_labels, text="512px", font=self.small_font, text_color="gray")
        min_label.pack(side="left")

        max_label = ctk.CTkLabel(slider_labels, text="1024px", font=self.small_font, text_color="gray")
        max_label.pack(side="right")

        # Pulsanti regolazione
        btn_frame = ctk.CTkFrame(res_frame, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(10, 20))

        decrease_btn = ctk.CTkButton(btn_frame, text="- Riduci", width=160,
                                     command=self.decrease_resolution,
                                     font=self.text_font)
        decrease_btn.pack(side="left", padx=5)

        increase_btn = ctk.CTkButton(btn_frame, text="+ Aumenta", width=160,
                                     command=self.increase_resolution,
                                     font=self.text_font)
        increase_btn.pack(side="right", padx=5)

        # Aggiunta FPS sotto lo slider della risoluzione
        fps_container = ctk.CTkFrame(res_frame, fg_color="transparent")
        fps_container.pack(fill="x", padx=20, pady=(10, 0))

        fps_info = ctk.CTkFrame(fps_container, fg_color=("#EAEAEA", "#333333"), corner_radius=10, height=36)
        fps_info.pack(side="left", fill="x", expand=True)
        fps_info.pack_propagate(False)

        fps_label_title = ctk.CTkLabel(fps_info, text="FPS:", font=self.text_font, text_color=("#555555", "#BBBBBB"))
        fps_label_title.pack(side="left", padx=(10, 5), pady=5)

        self.fps_label = ctk.CTkLabel(fps_info, text="--", font=self.text_font)
        self.fps_label.pack(side="left", pady=5)

        # Etichetta stato modello
        model_info = ctk.CTkFrame(fps_container, fg_color=("#EAEAEA", "#333333"), corner_radius=10, height=36)
        model_info.pack(side="right", fill="x", expand=True, padx=(10, 0))
        model_info.pack_propagate(False)

        status_label_title = ctk.CTkLabel(model_info, text="Stato:", font=self.text_font,
                                          text_color=("#555555", "#BBBBBB"))
        status_label_title.pack(side="left", padx=(10, 5), pady=5)

        self.model_status = ctk.CTkLabel(model_info, text="In attesa", font=self.text_font)
        self.model_status.pack(side="left", pady=5)

        # Footer
        footer_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        footer_frame.pack(fill="x", padx=20, pady=(0, 15))

        footer = ctk.CTkLabel(footer_frame, text="Capitan Acciaio Cartoonizer • Realizzato con ❤️",
                              font=self.small_font, text_color="gray")
        footer.pack(pady=5)

    def change_model(self, choice):
        self.command_queue.put(("change_model", choice))
        self.model_status.configure(text="Cambiando modello...")

    def update_resolution(self, value):
        resolution = int(value)
        self.processing_resolution = resolution
        self.command_queue.put(("change_resolution", resolution))
        self.res_value_label.configure(text=f"{resolution} px")

    def decrease_resolution(self):
        current = int(self.res_slider.get())
        new_value = max(512, current - 64)
        self.res_slider.set(new_value)
        self.update_resolution(new_value)

    def increase_resolution(self):
        current = int(self.res_slider.get())
        new_value = min(1024, current + 64)
        self.res_slider.set(new_value)
        self.update_resolution(new_value)

    def monitor_status_queue(self):
        while True:
            try:
                status_type, value = self.status_queue.get(timeout=0.5)

                if status_type == "fps_update":
                    self.fps_label.configure(text=f"{value:.1f}")
                elif status_type == "model_changed":
                    self.model_status.configure(text="Attivo")
                    self.model_var.set(value)
                elif status_type == "resolution_changed":
                    self.res_slider.set(value)
                    self.res_value_label.configure(text=f"{value} px")
            except:
                # Nessun aggiornamento disponibile
                pass
            time.sleep(0.1)

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.command_queue.put(("exit", None))
        self.root.destroy()


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

    command_queue = mp.Queue()
    status_queue = mp.Queue()
    models = ["face_paint_512_v1", "face_paint_512_v2", "celeba_distill", "paprika"]

    start_gui(models, command_queue, status_queue, "face_paint_512_v2", 768)
