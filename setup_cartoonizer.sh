#!/bin/bash

# Definizione colori per output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Installer Cartoonizer ===${NC}"

# Rileva sistema operativo
OS="$(uname)"
echo -e "${BLUE}Sistema operativo rilevato: ${OS}${NC}"

# Gestione specifiche per macOS
if [ "$OS" == "Darwin" ]; then
    echo -e "${BLUE}Configurazione ambiente macOS...${NC}"

    # Installa Homebrew se non presente
    if ! command -v brew &>/dev/null; then
        echo -e "${YELLOW}Homebrew non trovato. Installazione...${NC}"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Aggiungi Homebrew al PATH in base all'architettura
        if [[ $(uname -m) == 'arm64' ]]; then
            # Apple Silicon
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.bash_profile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        else
            # Intel
            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zshrc
            echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.bash_profile
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi

    # Installa XCode Command Line Tools se necessario
    xcode-select --print-path &>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Installazione XCode Command Line Tools...${NC}"
        xcode-select --install
        echo -e "${YELLOW}Attendi che l'installazione di XCode CLI Tools termini, poi premi un tasto...${NC}"
        read -n 1
    fi

    # Installa Python 3.12 con supporto tkinter
    if ! command -v python3.12 &>/dev/null; then
        echo -e "${BLUE}Installazione Python 3.12 con supporto tkinter...${NC}"
        brew install python@3.12
        brew install python-tk@3.12
    else
        echo -e "${GREEN}✓ Python 3.12 già installato${NC}"
    fi

    # Configura PATH per Python
    if [[ $(uname -m) == 'arm64' ]]; then
        # Apple Silicon
        export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"
        if ! grep -q "python@3.12" ~/.zshrc; then
            echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
        fi
        if [ -f ~/.bash_profile ] && ! grep -q "python@3.12" ~/.bash_profile; then
            echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.bash_profile
        fi
    else
        # Intel
        export PATH="/usr/local/opt/python@3.12/bin:$PATH"
        if ! grep -q "python@3.12" ~/.zshrc; then
            echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
        fi
        if [ -f ~/.bash_profile ] && ! grep -q "python@3.12" ~/.bash_profile; then
            echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.bash_profile
        fi
    fi

    # Installa OBS per webcam virtuale (opzionale)
    if ! command -v obs &>/dev/null; then
        echo -e "${YELLOW}OBS non rilevato. Vuoi installare OBS per la webcam virtuale? [s/n]${NC}"
        read -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            brew install --cask obs
        fi
    else
        echo -e "${GREEN}✓ OBS già installato${NC}"
    fi

    PYTHON_CMD="python3.12"

# Gestione per Linux
elif [ "$OS" == "Linux" ]; then
    echo -e "${BLUE}Configurazione ambiente Linux...${NC}"

    # Rileva distribuzione
    if command -v apt-get &>/dev/null; then
        # Debian/Ubuntu
        echo -e "${BLUE}Distribuzione basata su Debian rilevata...${NC}"
        sudo apt-get update

        # Controlla se Python 3.12 è disponibile nel repository standard
        if apt-cache show python3.12 &>/dev/null; then
            sudo apt-get install -y python3.12 python3.12-venv python3.12-tk python3-pip
        else
            # Aggiungi il repository deadsnakes per versioni recenti di Python
            echo -e "${YELLOW}Python 3.12 non disponibile nei repository standard. Aggiunta PPA...${NC}"
            sudo apt-get install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.12 python3.12-venv python3.12-tk python3-pip
        fi

        # Installa dipendenze necessarie per opencv
        sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

    elif command -v dnf &>/dev/null; then
        # Fedora/RHEL
        echo -e "${BLUE}Distribuzione basata su Red Hat rilevata...${NC}"
        sudo dnf install -y python3.12 python3.12-devel python3.12-tkinter
    else
        echo -e "${RED}Gestore pacchetti non supportato. Procedi con l'installazione manuale di Python 3.12${NC}"
        exit 1
    fi

    PYTHON_CMD="python3.12"

# Gestione per Windows
elif [[ "$OS" == "MINGW"* || "$OS" == "MSYS"* || "$OS" == "CYGWIN"* ]]; then
    echo -e "${BLUE}Configurazione ambiente Windows...${NC}"

    # Controlla se Python 3.12 è installato
    if python -c "import sys; assert sys.version_info >= (3, 12)" &>/dev/null; then
        echo -e "${GREEN}✓ Python 3.12 o superiore trovato${NC}"
        PYTHON_CMD="python"
    elif py -3.12 --version &>/dev/null; then
        echo -e "${GREEN}✓ Python 3.12 trovato tramite py launcher${NC}"
        PYTHON_CMD="py -3.12"
    else
        echo -e "${RED}Python 3.12 non trovato. Vai su https://www.python.org/downloads/ per installarlo.${NC}"
        echo -e "${YELLOW}Assicurati di selezionare 'Add Python to PATH' e 'Install pip' durante l'installazione.${NC}"
        exit 1
    fi

    # Verifica che tkinter sia installato
    if ! $PYTHON_CMD -c "import tkinter" &>/dev/null; then
        echo -e "${RED}Tkinter non trovato. Reinstalla Python 3.12 selezionando l'opzione tcl/tk.${NC}"
        exit 1
    fi

else
    echo -e "${RED}Sistema operativo non supportato: $OS${NC}"
    exit 1
fi

# Verifica Python e versione
echo -e "${BLUE}Verifica versione Python...${NC}"
if ! command -v $PYTHON_CMD &>/dev/null; then
    echo -e "${RED}Python non trovato. L'installazione potrebbe essere fallita.${NC}"
    exit 1
fi

$PYTHON_CMD --version
$PYTHON_CMD -c "import sys; print('Python path:', sys.executable)"

# Verifica prerequisiti
echo -e "${BLUE}Verifica prerequisiti...${NC}"
$PYTHON_CMD -c "
import sys
try:
    import tkinter
    print('✓ Tkinter disponibile')
except ImportError:
    print('ERROR: Tkinter non disponibile!')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Tkinter non disponibile. Installalo prima di continuare.${NC}"
    exit 1
fi

# Crea e attiva ambiente virtuale
echo -e "${BLUE}Configurazione ambiente virtuale...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${BLUE}Creazione nuovo ambiente virtuale...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo -e "${RED}Errore nella creazione dell'ambiente virtuale. Verifica i permessi.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Ambiente virtuale creato in: $VENV_DIR${NC}"
else
    echo -e "${GREEN}✓ Usando ambiente virtuale esistente${NC}"
fi

# Attiva ambiente virtuale
if [ "$OS" == "Darwin" ] || [ "$OS" == "Linux" ]; then
    source "$VENV_DIR/bin/activate"
else
    source "$VENV_DIR/Scripts/activate"
fi

if [ $? -ne 0 ]; then
    echo -e "${RED}Errore nell'attivazione dell'ambiente virtuale.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Ambiente virtuale attivato${NC}"

# Aggiorna pip
echo -e "${BLUE}Aggiornamento pip...${NC}"
python -m pip install --upgrade pip

# Installa dipendenze
echo -e "${BLUE}Installazione dipendenze Python...${NC}"

# Controlla se esiste requirements.txt
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo -e "${BLUE}Installazione da requirements.txt...${NC}"
    python -m pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo -e "${BLUE}Installazione dipendenze specificate...${NC}"

    # Installa prima le librerie di base
    python -m pip install numpy pillow certifi charset-normalizer requests packaging

    # Installa custom tkinter
    python -m pip install customtkinter

    # Installa PyTorch in modo specifico per ogni OS
    if [ "$OS" == "Darwin" ]; then
        if [[ $(uname -m) == 'arm64' ]]; then
            # Apple Silicon ottimizzato (MPS)
            echo -e "${BLUE}Installazione PyTorch ottimizzata per Apple Silicon...${NC}"
            python -m pip install torch torchvision
        else
            # Intel Mac
            python -m pip install torch torchvision
        fi
    else
        # Installazione standard per altri OS
        python -m pip install torch torchvision
    fi

    # OpenCV - particolare attenzione per macOS
    if [ "$OS" == "Darwin" ]; then
        echo -e "${BLUE}Installazione OpenCV ottimizzata per macOS...${NC}"
        python -m pip install opencv-python-headless
        # Controlla se l'installazione è riuscita
        if ! python -c "import cv2" &>/dev/null; then
            echo -e "${YELLOW}Riprovo con opencv-python standard...${NC}"
            python -m pip install opencv-python
        fi
    else
        python -m pip install opencv-python
    fi

    # Altre dipendenze
    python -m pip install psutil pyvirtualcam
fi

# Ottimizzazione specifica per Apple Silicon
if [ "$OS" == "Darwin" ] && [[ $(uname -m) == 'arm64' ]]; then
    echo -e "${BLUE}Ottimizzazione per Apple Silicon...${NC}"
    python -c "
import torch
if torch.backends.mps.is_available():
    print('${GREEN}✓ MPS disponibile per accelerazione Metal${NC}')
else:
    print('${YELLOW}⚠️ MPS non disponibile, usando CPU${NC}')
"
fi

# Crea directory necessarie
mkdir -p "$SCRIPT_DIR/models"
mkdir -p "$SCRIPT_DIR/fonts"

# Verifica file main.py
if [ ! -f "$SCRIPT_DIR/main.py" ]; then
    echo -e "${RED}File main.py non trovato. Controlla che ti trovi nella directory corretta.${NC}"
    exit 1
fi

# Verifica finale dipendenze
echo -e "${BLUE}Verifica finale dipendenze...${NC}"
python -c "
import sys
required = ['numpy', 'cv2', 'torch', 'torchvision', 'customtkinter', 'PIL', 'psutil', 'pyvirtualcam']
missing = []

for pkg in required:
    try:
        if pkg == 'PIL':
            __import__('PIL')
        else:
            __import__(pkg.split('-')[0])
    except ImportError:
        missing.append(pkg)

if missing:
    print('${YELLOW}⚠️ Dipendenze mancanti: ' + ', '.join(missing) + '${NC}')
    sys.exit(1)
else:
    print('${GREEN}✓ Tutte le dipendenze necessarie sono installate${NC}')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Attenzione: alcune dipendenze potrebbero non essere installate correttamente.${NC}"
    echo -e "${YELLOW}Vuoi procedere comunque? [s/n]${NC}"
    read -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        exit 1
    fi
fi

# Avvio applicazione
echo -e "${GREEN}✅ Installazione completata con successo!${NC}"
echo -e "${BLUE}Avvio applicazione Cartoonizer...${NC}"
python "$SCRIPT_DIR/main.py"