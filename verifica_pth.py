import torch
from model import Generator
import sys

def verifica_compatibilita_pth(percorso_file):
    print(f"Verifica del file: {percorso_file}")

    # Carica il modello vuoto per confronto
    modello_vuoto = Generator()
    struttura_riferimento = {k: v.shape for k, v in modello_vuoto.state_dict().items()}

    try:
        # Carica il file .pth
        checkpoint = torch.load(percorso_file, map_location="cpu")

        # Controlla se è un dizionario con "model_state_dict" oppure direttamente lo state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print("Formato: Dizionario con chiave 'model_state_dict'")
        else:
            state_dict = checkpoint
            print("Formato: State dict diretto")

        # Verifica della compatibilità
        if not isinstance(state_dict, dict):
            print("❌ ERRORE: Il file non contiene un dizionario valido di parametri")
            return False

        # Verifica le chiavi e le forme
        chiavi_modello = set(struttura_riferimento.keys())
        chiavi_file = set(state_dict.keys())

        if chiavi_modello != chiavi_file:
            print("❌ ERRORE: Le chiavi del modello non corrispondono")
            print(f"Mancanti: {chiavi_modello - chiavi_file}")
            print(f"Extra: {chiavi_file - chiavi_modello}")
            return False

        # Verifica le dimensioni dei tensori
        for chiave in chiavi_modello:
            if state_dict[chiave].shape != struttura_riferimento[chiave]:
                print(f"❌ ERRORE: Forma incompatibile per {chiave}")
                print(f"Attesa: {struttura_riferimento[chiave]}, Trovata: {state_dict[chiave].shape}")
                return False

        print("✅ COMPATIBILE: Il file è compatibile con il modello Generator")
        return True

    except Exception as e:
        print(f"❌ ERRORE durante la verifica: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verifica_compatibilita_pth(sys.argv[1])
    else:
        print("Utilizzo: python verifica_pth.py percorso/al/file.pth")