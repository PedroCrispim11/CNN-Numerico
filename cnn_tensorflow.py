import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Oculta logs desnecessários do TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image

print("=== PASSO 1: A carregar o MNIST via TensorFlow ===")
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

# Limpa o ecrã do terminal no VS Code antes de solicitar os parâmetros de treino
os.system('cls' if os.name == 'nt' else 'clear')

print("=== PASSO 2: Configuração Interativa do Treino ===")
NUM_AMOSTRAS_TREINO = int(input("Quantidade de amostras para o treino (ex: 50): "))
NUM_AMOSTRAS_TREINO = min(max(NUM_AMOSTRAS_TREINO, 10), len(X_train_full))

EPOCAS = int(input("Quantidade de épocas para o treino (ex: 20): "))
EPOCAS = min(max(EPOCAS, 1), 100)

# Preparar os dados para o formato tensorial exigido pelo TensorFlow
X_treino = X_train_full[:NUM_AMOSTRAS_TREINO].astype(float) / 255.0
X_treino = X_treino.reshape((NUM_AMOSTRAS_TREINO, 28, 28, 1))
y_treino = y_train_full[:NUM_AMOSTRAS_TREINO]


# --- PASSO 3: CONSTRUÇÃO DA ARQUITETURA DA CNN (TENSORFLOW / KERAS) ---
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),  # Etapa 1: Convolução
    layers.MaxPooling2D((2, 2)),                  # Etapa 2: Pooling
    layers.Flatten(),                             # Etapa 3: Flattening
    layers.Dense(64, activation='relu'),          # Camada Densa Oculta (MLP)
    layers.Dense(10, activation='softmax')        # Etapa 4: Rede Neural Densa (Saída)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Callback personalizado para apresentar o progresso das épocas em português
class RelogioEmPortugues(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"--- Fim da Época {epoch + 1}/{EPOCAS} ---")
        print(f"  • Exatidão (Treino): {logs.get('accuracy') * 100:.2f}%")
        print(f"  • Erro (Treino):     {logs.get('loss'):.4f}\n")


print(f"\n=== PASSO 4: A treinar o modelo via TensorFlow ({NUM_AMOSTRAS_TREINO} amostras | {EPOCAS} épocas) ===")
model.fit(X_treino, y_treino, epochs=EPOCAS, verbose=0, callbacks=[RelogioEmPortugues()])
print("-> Treino concluído com sucesso! O modelo encontra-se ativo na memória.")


# --- PASSO 5: CICLO INTERATIVO DE TESTES (COM O MESMO MODELO) ---
imagens_teste = ["teste1.jpg", "teste2.jpg", "teste3.jpg", "teste4.jpg", "teste5.jpg"]

while True:
    print("\n" + "="*50)
    print("=== MENU DE TESTES DO MODELO TREINADO ===")
    print("[1] Gerar novas imagens de teste aleatórias e avaliar")
    print("[2] Usar as imagens de teste existentes na pasta e avaliar")
    print("[3] Sair do programa")
    
    escolha_menu = input("Escolha uma opção (1, 2 ou 3): ").strip()
    
    if escolha_menu == '3':
        print("\nA encerrar o programa. Até breve!")
        break
        
    etiquetas_reais = []
    
    if escolha_menu == '1':
        print("\n-> A gerar e a guardar 5 novas imagens de teste aleatórias...")
        indices_aleatorios = np.random.choice(len(X_test_full), 5, replace=False)
        for idx, i in enumerate(indices_aleatorios):
            img_array = X_test_full[i]
            img = Image.fromarray(img_array.astype('uint8'))
            nome_ficheiro = imagens_teste[idx]
            img.save(nome_ficheiro)
            etiqueta = int(y_test_full[i])
            etiquetas_reais.append(etiqueta)
            print(f"  • Guardado: {nome_ficheiro} (Dígito real: {etiqueta})")
        
        with open("etiquetas_reais.txt", "w") as f:
            for etq in etiquetas_reais:
                f.write(f"{etq}\n")
                
    elif escolha_menu == '2':
        if all(os.path.exists(f) for f in imagens_teste) and os.path.exists("etiquetas_reais.txt"):
            print("\n-> A carregar as imagens existentes e respetivas etiquetas da pasta...")
            with open("etiquetas_reais.txt", "r") as f:
                etiquetas_reais = [int(line.strip()) for line in f.readlines()]
            for idx, nome_ficheiro in enumerate(imagens_teste):
                print(f"  • A utilizar: {nome_ficheiro} (Dígito real registado: {etiquetas_reais[idx]})")
        else:
            print("\n[Aviso] Ficheiros de teste ou etiquetas não encontrados na pasta. Por favor, escolha a opção [1] primeiro.")
            continue
    else:
        print("\n[Aviso] Opção inválida. Por favor, escolha 1, 2 ou 3.")
        continue
        
    # Execução da Inferência
    print("\n=== A EXECUTAR INFERÊNCIA NAS 5 IMAGENS ===")
    total_acertos = 0
    total_erros = 0
    
    for indice, nome_ficheiro in enumerate(imagens_teste):
        if os.path.exists(nome_ficheiro):
            img = Image.open(nome_ficheiro).convert('L').resize((28, 28))
            img_arr = np.array(img, dtype=float) / 255.0
            
            if np.mean(img_arr) > 0.5:
                img_arr = 1.0 - img_arr
                
            img_input = img_arr.reshape(1, 28, 28, 1)
            
            predicao = model.predict(img_input, verbose=0)
            digito_previsto = np.argmax(predicao)
            confianca = np.max(predicao) * 100
            etiqueta_real = etiquetas_reais[indice]
            
            if digito_previsto == etiqueta_real:
                total_acertos += 1
                status = "ACERTOU"
            else:
                total_erros += 1
                status = "ERROU"
            
            print(f"-> Ficheiro [{nome_ficheiro}] | Real: {etiqueta_real} | Previsto: **{digito_previsto}** ({status} - Confiança: {confianca:.2f}%)")
    
    # Resumo final do desempenho
    print("\n" + "="*50)
    print("=== RESUMO FINAL DO DESEMPENHO (5 IMAGENS) ===")
    print(f"  • Total de Acertos: {total_acertos} / 5")
    print(f"  • Total de Erros:   {total_erros} / 5")
    print(f"  • Taxa de Sucesso:  {(total_acertos / 5) * 100:.2f}%")
    print("="*50)