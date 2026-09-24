import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Oculta logs desnecessários do TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from scipy.ndimage import center_of_mass, shift

print("=== PASSO 1: A carregar o MNIST via TensorFlow ===")
(X_train_full, y_train_full), (_, _) = tf.keras.datasets.mnist.load_data()

# Limpa o ecrã do terminal no VS Code antes de solicitar os parâmetros
os.system('cls' if os.name == 'nt' else 'clear')

print("=== PASSO 2: Configuração Interativa do Treino ===")
NUM_AMOSTRAS_TREINO = int(input("Quantidade de amostras para o treino (ex: 10000): "))
NUM_AMOSTRAS_TREINO = min(max(NUM_AMOSTRAS_TREINO, 10), len(X_train_full))

EPOCAS = int(input("Quantidade de épocas para o treino (ex: 5): "))
EPOCAS = min(max(EPOCAS, 1), 100)

# Preparar os dados para o formato tensorial exigido pelo TensorFlow
X_treino = X_train_full[:NUM_AMOSTRAS_TREINO].astype(float) / 255.0
X_treino = X_treino.reshape((NUM_AMOSTRAS_TREINO, 28, 28, 1))
y_treino = y_train_full[:NUM_AMOSTRAS_TREINO]


# --- PASSO 3: CONSTRUÇÃO DA ARQUITETURA DA CNN ---
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),  # Etapa 1: Convolução
    layers.MaxPooling2D((2, 2)),                  # Etapa 2: Pooling
    layers.Flatten(),                             # Etapa 3: Flattening
    layers.Dense(64, activation='relu'),          # Camada Densa Oculta
    layers.Dense(10, activation='softmax')        # Etapa 4: Rede Neural Densa (Saída)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Callback para apresentar o progresso das épocas em português
class RelogioEmPortugues(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"--- Fim da Época {epoch + 1}/{EPOCAS} ---")
        print(f"  • Exatidão (Treino): {logs.get('accuracy') * 100:.2f}%")
        print(f"  • Erro (Treino):     {logs.get('loss'):.4f}\n")


print(f"\n=== PASSO 4: A treinar o modelo via TensorFlow ({NUM_AMOSTRAS_TREINO} amostras | {EPOCAS} épocas) ===")
model.fit(X_treino, y_treino, epochs=EPOCAS, verbose=0, callbacks=[RelogioEmPortugues()])
print("-> Treino concluído com sucesso! A abrir a tela de desenho inteligente...")


# --- PASSO 5: JANELA DE DESENHO LIVRE COM CENTRAGEM AUTOMÁTICA ---
janela = tk.Tk()
janela.title("Reconhecimento de Dígitos - Inteligente")
janela.geometry("320x420")
janela.resizable(False, False)

# Criar imagem em memória para suportar o desenho (fundo branco, tamanho 280x280)
img_pil = Image.new("L", (280, 280), 255)
draw = ImageDraw.Draw(img_pil)

label_instrucao = tk.Label(janela, text="Desenhe um dígito de 0 a 9 abaixo:", font=("Arial", 11))
label_instrucao.pack(pady=10)

canvas = tk.Canvas(janela, width=280, height=280, bg="white", cursor="cross")
canvas.pack()

label_resultado = tk.Label(janela, text="Previsão: A aguardar desenho...", font=("Arial", 12, "bold"), fg="blue")
label_resultado.pack(pady=10)

def desenhar(event):
    r = 8 # Pincel ligeiramente mais largo para dar corpo ao traço
    x, y = event.x, event.y
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
    draw.ellipse([x - r, y - r, x + r, y + r], fill=0, outline=0)
    
canvas.bind("<B1-Motion>", desenhar)

def prever_desenho():
    # 1. Redimensionar a imagem desenhada para 28x28 pixels
    img_resized = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 2. Converter para matriz NumPy e inverter cores (fundo preto, traço branco)
    img_arr = np.array(img_resized, dtype=float)
    img_arr = 255.0 - img_arr
    img_arr = img_arr / 255.0
    
    # 3. CENTRAGEM AUTOMÁTICA POR CENTRO DE MASSA (Igual ao MNIST)
    if np.sum(img_arr) > 0:
        cy, cx = center_of_mass(img_arr)
        rows, cols = img_arr.shape
        shift_y = np.round(rows / 2.0 - cy).astype(int)
        shift_x = np.round(cols / 2.0 - cx).astype(int)
        img_arr = shift(img_arr, [shift_y, shift_x], cval=0.0)
    
    # 4. Formatar para a entrada do modelo
    img_input = img_arr.reshape(1, 28, 28, 1)
    
    # 5. Executar predição
    predicao = model.predict(img_input, verbose=0)
    digito_previsto = np.argmax(predicao)
    confianca = np.max(predicao) * 100
    
    label_resultado.config(text=f"Previsto: {digito_previsto} (Confiança: {confianca:.1f}%)")
    print(f"-> Desenho avaliado | Dígito Previsto: **{digito_previsto}** (Confiança: {confianca:.2f}%)")

def limpar_tela():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=255)
    label_resultado.config(text="Previsão: A aguardar desenho...")

frame_botoes = tk.Frame(janela)
frame_botoes.pack(pady=5)

btn_prever = tk.Button(frame_botoes, text="Prever", bg="green", fg="white", font=("Arial", 10, "bold"), width=10, command=prever_desenho)
btn_prever.pack(side=tk.LEFT, padx=5)

btn_limpar = tk.Button(frame_botoes, text="Limpar", bg="red", fg="white", font=("Arial", 10, "bold"), width=10, command=limpar_tela)
btn_limpar.pack(side=tk.RIGHT, padx=5)

janela.mainloop()