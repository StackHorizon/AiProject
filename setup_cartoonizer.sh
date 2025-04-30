#!/bin/bash

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Funzioni per messaggi
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESSO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[AVVISO]${NC} $1"; }
print_error() { echo -e "${RED}[ERRORE]${NC} $1"; }

# Verifica e installazione Homebrew
print_status "Verifico se Homebrew è installato..."
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew non trovato. Installazione in corso..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    if [[ $(uname -m) == 'arm64' ]]; then
        print_status "Configurazione Homebrew per Apple Silicon..."
        # Aggiungi al profilo solo se la riga non esiste già
        grep -q 'eval "$(/opt/homebrew/bin/brew shellenv)"' ~/.zprofile ||
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    print_success "Homebrew già installato"
fi

# Installazione e configurazione Tcl/Tk
print_status "Installazione Tcl/Tk per tkinter..."
brew install tcl-tk

# Configura variabili d'ambiente per Tkinter
print_status "Configurazione variabili d'ambiente..."
TCL_CONFIG=$(cat <<EOF
export LDFLAGS="-L/opt/homebrew/opt/tcl-tk/lib"
export CPPFLAGS="-I/opt/homebrew/opt/tcl-tk/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/tcl-tk/lib/pkgconfig"
export PYTHON_CONFIGURE_OPTS="--with-tcltk-includes='-I/opt/homebrew/opt/tcl-tk/include' --with-tcltk-libs='-L/opt/homebrew/opt/tcl-tk/lib -ltcl8.6 -ltk8.6'"
EOF
)

# Aggiungi al profilo solo se non esistono già
for line in $TCL_CONFIG; do
    grep -q "$line" ~/.zprofile || echo "$line" >> ~/.zprofile
done

# Applica le variabili d'ambiente alla sessione corrente
eval "$TCL_CONFIG"

# Crea collegamenti simbolici se necessario
print_status "Configurazione collegamenti simbolici..."
mkdir -p ~/Library/Frameworks
if [ -d "/opt/homebrew/opt/tcl-tk/lib" ]; then
    ln -sf /opt/homebrew/opt/tcl-tk/lib/libtcl* ~/Library/Frameworks/
    ln -sf /opt/homebrew/opt/tcl-tk/lib/libtk* ~/Library/Frameworks/
else
    print_error "Directory Tcl/Tk non trovata!"
    exit 1
fi

# Gestione Python
print_status "Configurazione Python 3.12..."
brew uninstall --ignore-dependencies python@3.12 2>/dev/null
brew install --build-from-source python@3.12

# Verifica Python
if ! command -v python3.12 &> /dev/null; then
    print_error "Python 3.12 non trovato. Installazione fallita."
    exit 1
fi

# Configura PATH per Python
grep -q 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' ~/.zprofile ||
    echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zprofile
export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"

# Setup progetto
PROJECT_DIR="$(pwd)"
print_status "Progetto in: $PROJECT_DIR"

# Ambiente virtuale
print_status "Configurazione ambiente virtuale..."
python3.12 -m venv venv
source venv/bin/activate

# Verifica tkinter
print_status "Verifica tkinter..."
python -c "import tkinter; print('Tkinter funziona!')" ||
    print_warning "Tkinter non configurato correttamente, ma continuo..."

# Installazione dipendenze
print_status "Installazione dipendenze..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow numpy customtkinter requests pyvirtualcam tkextrafont

# Verifica finale
print_status "Verifica finale..."
if python -c "import tkinter, customtkinter; print('Tkinter e CustomTkinter OK!')"; then
    print_success "Installazione completata con successo!"
    print_status "Avvio dell'applicazione..."
    python main.py
else
    print_error "Tkinter o CustomTkinter non funzionano! Verifica l'installazione."
    exit 1
fi