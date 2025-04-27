#!/bin/bash

# Colori per output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Cartoonizer - Installazione automatica per macOS ===${NC}"

# Controlla se Homebrew è installato
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew non trovato. Installazione consigliata.${NC}"
    echo -e "Per installare Homebrew, esegui questo comando nel terminale:"
    echo -e "${GREEN}/bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
    echo -e "Dopo l'installazione, riavvia questo script."
    read -p "Vuoi provare a installare Homebrew automaticamente? [s/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    else
        exit 1
    fi
fi

# Installa Python 3.12 (versione richiesta)
echo -e "${BLUE}Verifica/installazione Python 3.12...${NC}"
if brew list python@3.12 &>/dev/null; then
    echo -e "${GREEN}✓ Python 3.12 è già installato${NC}"
else
    echo -e "${YELLOW}Installazione Python 3.12 in corso...${NC}"
    brew install python@3.12

    # Aggiungi Python 3.12 al PATH se non è già disponibile
    if ! command -v python3.12 &> /dev/null; then
        echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
        echo 'export PATH="/usr/local/opt/python@3.12/bin:$PATH"' >> ~/.bash_profile
        export PATH="/usr/local/opt/python@3.12/bin:$PATH"
    fi
fi

# Verifica Python installato correttamente
if command -v python3.12 &> /dev/null; then
    echo -e "${GREEN}✓ Python 3.12 installato e disponibile${NC}"
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Usando Python 3 generico (non versione specifica 3.12)${NC}"
    PYTHON_CMD="python3"
else
    echo -e "${RED}Impossibile trovare Python. Uscita.${NC}"
    exit 1
fi

# Mostra versione Python
echo -e "${BLUE}Versione Python in uso:${NC}"
$PYTHON_CMD --version

# Crea cartella del progetto se non esiste
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo -e "${BLUE}Cartella del progetto: ${SCRIPT_DIR}${NC}"

# Crea ambiente virtuale
echo -e "${BLUE}Creazione ambiente virtuale...${NC}"
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    $PYTHON_CMD -m venv "${SCRIPT_DIR}/venv"
    echo -e "${GREEN}✓ Ambiente virtuale creato${NC}"
else
    echo -e "${GREEN}✓ Ambiente virtuale già esistente${NC}"
fi

# Attiva ambiente virtuale
echo -e "${BLUE}Attivazione ambiente virtuale...${NC}"
source "${SCRIPT_DIR}/venv/bin/activate"

# Aggiorna pip
echo -e "${BLUE}Aggiornamento pip...${NC}"
python -m pip install --upgrade pip

# Installa dipendenze
echo -e "${BLUE}Installazione dipendenze...${NC}"
python -m pip install numpy opencv-python torch torchvision customtkinter pillow psutil pyvirtualcam requests

# Verifica se è necessario installare pacchetti specifici per macOS M1/M2
if [[ $(uname -m) == 'arm64' ]]; then
    echo -e "${BLUE}Rilevato Mac con Apple Silicon, installazione pacchetti ottimizzati...${NC}"
    # Controlla se PyTorch è già installato con supporto MPS
    if python -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        echo -e "${GREEN}✓ PyTorch con supporto MPS già installato${NC}"
    else
        echo -e "${YELLOW}Reinstallazione PyTorch con supporto MPS...${NC}"
        python -m pip uninstall -y torch torchvision
        python -m pip install torch torchvision
    fi
fi

# Verifica se installare OBS Virtual Camera
echo -e "${YELLOW}Per la webcam virtuale è necessario OBS Virtual Camera${NC}"
read -p "Vuoi installare OBS? [s/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Ss]$ ]]; then
    brew install --cask obs
    echo -e "${GREEN}✓ OBS installato. Ora devi attivare OBS Virtual Camera:${NC}"
    echo -e "1. Apri OBS"
    echo -e "2. Vai su Strumenti > Start Virtual Camera"
else
    echo -e "${YELLOW}Nota: Senza OBS Virtual Camera alcune funzionalità potrebbero non funzionare${NC}"
fi

# Crea cartelle necessarie
mkdir -p "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/fonts"

# Scarica il progetto se i file principali non esistono
if [ ! -f "${SCRIPT_DIR}/main.py" ]; then
    echo -e "${YELLOW}File main.py non trovato. Vuoi scaricare i file del progetto? [s/n] ${NC}"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        # Qui dovresti inserire il comando per scaricare i file del progetto
        # Ad esempio con git clone o curl
        echo -e "${YELLOW}Per favore, copia manualmente i file del progetto nella cartella: ${SCRIPT_DIR}${NC}"
        exit 1
    else
        echo -e "${RED}File main.py non trovato. Impossibile continuare.${NC}"
        exit 1
    fi
fi

# Verifica dipendenze Python
echo -e "${BLUE}Verifica dipendenze Python installate...${NC}"
python -c "
import sys
required = ['numpy', 'cv2', 'torch', 'torchvision', 'customtkinter', 'PIL', 'psutil', 'requests']
missing = [pkg for pkg in required if pkg not in sys.modules and not __import__(pkg, fromlist=[''])]
if missing:
    print('${YELLOW}⚠️ Dipendenze mancanti:', ', '.join(missing))
else:
    print('${GREEN}✓ Tutte le dipendenze necessarie sono installate')
"

# Esegui l'applicazione
echo -e "${GREEN}✅ Installazione completata!${NC}"
echo -e "${BLUE}Avvio dell'applicazione Cartoonizer...${NC}"
python "${SCRIPT_DIR}/main.py"