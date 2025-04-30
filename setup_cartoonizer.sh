#!/bin/bash

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Funzione per stampare messaggi di stato
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESSO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[AVVISO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERRORE]${NC} $1"
}

# Controllo se Homebrew è installato
print_status "Verifico se Homebrew è installato..."
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew non trovato. Installazione in corso..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Configurazione di Homebrew per Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        print_status "Configurazione di Homebrew per Apple Silicon..."
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    print_success "Homebrew già installato"
fi

# Installazione delle dipendenze richieste
print_status "Installazione di Tcl/Tk (richiesto per tkinter)..."
brew install tcl-tk
brew install python-tk
python -m tkinter

# Disinstallazione di qualsiasi versione precedente di Python 3.12
print_status "Disinstallazione di eventuali versioni precedenti di Python 3.12..."
brew uninstall --ignore-dependencies python@3.12 2>/dev/null

# Installazione di Python 3.12 (senza flag non valido)
print_status "Installazione di Python 3.12..."
brew install python@3.12

# Verifica che Python sia installato correttamente
if ! command -v python3.12 &> /dev/null; then
    print_error "Impossibile trovare Python 3.12. L'installazione potrebbe essere fallita."
    exit 1
fi

# Aggiungi Python al PATH
print_status "Configurazione delle variabili d'ambiente..."
echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zprofile
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"

# Configurazione delle variabili d'ambiente per Tkinter
print_status "Configurazione delle variabili d'ambiente per Tkinter..."
echo 'export PYTHON_CONFIGURE_OPTS="--with-tcltk-includes=-I/opt/homebrew/opt/tcl-tk/include --with-tcltk-libs=\"-L/opt/homebrew/opt/tcl-tk/lib -ltcl8.6 -ltk8.6\""' >> ~/.zprofile
export PYTHON_CONFIGURE_OPTS="--with-tcltk-includes=-I/opt/homebrew/opt/tcl-tk/include --with-tcltk-libs=\"-L/opt/homebrew/opt/tcl-tk/lib -ltcl8.6 -ltk8.6\""

# Collegamento simbolico per tcl-tk
print_status "Creazione di collegamenti simbolici per Tcl/Tk..."
mkdir -p ~/Library/Frameworks
ln -sf /opt/homebrew/opt/tcl-tk/lib/libtcl* ~/Library/Frameworks/
ln -sf /opt/homebrew/opt/tcl-tk/lib/libtk* ~/Library/Frameworks/

# Directory del progetto
PROJECT_DIR="$(pwd)"
print_status "Progetto in: $PROJECT_DIR"

# Creazione e attivazione dell'ambiente virtuale
print_status "Creazione dell'ambiente virtuale..."
python3.12 -m venv venv
source venv/bin/activate

# Aggiornamento di pip
print_status "Aggiornamento di pip..."
pip install --upgrade pip

# Installazione delle dipendenze
print_status "Installazione delle dipendenze..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy customtkinter requests
pip install pyvirtualcam tkextrafont

# Verifica che tkinter funzioni correttamente
print_status "Verifica dell'installazione di tkinter..."
python -c "import tkinter; print('Tkinter funziona correttamente!')" || (print_error "Tkinter non funziona ancora! Consulta l'output per dettagli." && exit 1)

print_success "Installazione completata con successo!"
print_status "Avvio dell'applicazione..."

# Esecuzione dell'applicazione
python main.py