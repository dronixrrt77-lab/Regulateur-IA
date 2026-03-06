import streamlit as st
import pickle
import neat
import matplotlib.pyplot as plt

# 1. Interface pour ton téléphone
st.set_page_config(page_title="Régulateur IA", layout="centered")
st.title("⚡ Régulateur de Tension IA")
st.write("Cible de stabilisation : 220V")

# 2. Fonction pour charger le cerveau (IA) et la config
@st.cache_resource
def load_ia():
    try:
        # On charge le fichier .pkl présent sur ton GitHub
        with open('IA_stabilisatrice.pkl', 'rb') as f:
            gagnant = pickle.load(f)
        
        # On charge la config en utilisant le nom exact 'config_ia'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             'config_ia')
        
        return neat.nn.FeedForwardNetwork.create(gagnant, config)
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return None

# 3. Application du régulateur
net = load_ia()

if net:
    st.success("✅ IA connectée")
    
    # Curseur pour simuler une tension au Togo (100V - 400V)
    tension_entree = st.slider("Tension d'entrée (V)", 100.0, 400.0, 300.0)
    
    if st.button("Stabiliser à 220V"):
        historique = []
        tension = tension_entree
        
        # L'IA corrige la tension sur 50 cycles
        for _ in range(50):
            sortie = net.activate((tension, 220.0))
            tension += sortie[0]
            historique.append(tension)
        
        st.metric("Tension stabilisée", f"{tension:.2f} V")
        
        # Affichage du graphique de stabilisation
        fig, ax = plt.subplots()
        ax.plot(historique, label="Tension", color="blue", linewidth=2)
        ax.axhline(y=220, color='red', linestyle='--', label="Cible 220V")
        ax.set_ylim(0, 450)
        ax.set_ylabel("Voltage (V)")
        ax.legend()
        st.pyplot(fig)
else:
    st.warning("Vérifie que 'IA_stabilisatrice.pkl' et 'config_ia' sont bien sur GitHub.")
    
