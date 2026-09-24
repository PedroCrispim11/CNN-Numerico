import os
import numpy as np
from PIL import Image
import tensorflow as tf

print("=== PASSO 1: A carregar o MNIST via TensorFlow (Apenas para dados) ===")
(X_train_full, y_train_full), (X_test_full, y_test_full) = tf.keras.datasets.mnist.load_data()

# Limpa o ecrã do terminal no VS Code antes da configuração
os.system('cls' if os.name == 'nt' else 'clear')

print("=== PASSO 2: Configuração Interativa do Treino (NumPy) ===")
NUM_AMOSTRAS_TREINO = int(input("Quantidade de amostras para o treino (ex: 1000): "))
NUM_AMOSTRAS_TREINO = min(max(NUM_AMOSTRAS_TREINO, 100), len(X_train_full))

EPOCAS = int(input("Quantidade de épocas para o treino (ex: 3): "))
EPOCAS = min(max(EPOCAS, 1), 20)


# --- FUNÇÕES DE PROCESSAMENTO MANUAL (NUMPY) ---

def convolucao_2d(imagem, kernel):
    """Etapa 1: Convolução 2D (Extração espacial de caraterísticas)."""
    h, w = imagem.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1
    saida = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = imagem[i:i+kh, j:j+kw]
            saida[i, j] = np.sum(patch * kernel)
    return saida

def max_pooling_2d(imagem, pool_size=2):
    """Etapa 2: Max Pooling (Redução da dimensionalidade)."""
    h, w = imagem.shape
    oh, ow = h // pool_size, w // pool_size
    saida = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = imagem[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            saida[i, j] = np.max(patch)
    return saida

def flattening(mapa):
    """Etapa 3: Flattening (Conversão num vetor unidimensional)."""
    return mapa.flatten()

# Filtro Sharpen clássico
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=float)

X_treino_bruto = X_train_full[:NUM_AMOSTRAS_TREINO].astype(float) / 255.0
y_treino = y_train_full[:NUM_AMOSTRAS_TREINO]

# Definição dinâmica do intervalo de exibição no terminal
if NUM_AMOSTRAS_TREINO <= 1000:
    intervalo_progresso = 100
elif NUM_AMOSTRAS_TREINO <= 10000:
    intervalo_progresso = 1000
else:
    intervalo_progresso = 10000

print(f"\n[A aplicar Convolução e Pooling em {NUM_AMOSTRAS_TREINO} amostras (intervalo de {intervalo_progresso})...]")
X_treino_processado = []
for i in range(NUM_AMOSTRAS_TREINO):
    if (i + 1) % intervalo_progresso == 0 or i == 0:
        print(f"  -> A processar imagem {i + 1}/{NUM_AMOSTRAS_TREINO}...")
        
    img = X_treino_bruto[i]
    conv = convolucao_2d(img, kernel_sharpen)
    pool = max_pooling_2d(conv, pool_size=2)
    flat = flattening(pool)
    X_treino_processado.append(flat)

X_treino_processado = np.array(X_treino_processado)
max_val = np.max(X_treino_processado)
if max_val > 0:
    X_treino_processado = X_treino_processado / max_val
print("-> Pré-processamento convolucional concluído com sucesso!")


# --- REDE NEURAL DENSA COM RETROPROPAGAÇÃO MANUAL ---

def softmax(z):
    exps = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exps / np.sum(exps, axis=-1, keepdims=True)

# Inicialização dos pesos da camada densa
np.random.seed(42)
input_dim = X_treino_processado.shape[1]
hidden_dim = 64
output_dim = 10

W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

taxa_aprendizagem = 0.05

print(f"\n=== PASSO 3: Treino por Épocas com Retropropagação ({NUM_AMOSTRAS_TREINO} amostras | {EPOCAS} épocas) ===")

for epoca in range(EPOCAS):
    print(f"-> A iniciar Época {epoca + 1}/{EPOCAS} (Calculando retropropagação...)")
    perda_total = 0
    acertos = 0
    
    for i in range(NUM_AMOSTRAS_TREINO):
        x = X_treino_processado[i:i+1]
        y_real = y_treino[i]
        
        # 1. Forward Pass (Passagem Direta)
        z1 = np.dot(x, W1) + b1
        a1 = np.maximum(0, z1) # Função de ativação ReLU
        z2 = np.dot(a1, W2) + b2
        a2 = softmax(z2)       # Camada de Saída
        
        # Cálculo da perda (Cross-Entropy Loss)
        perda_total += -np.log(a2[0, y_real] + 1e-9)
        
        if np.argmax(a2) == y_real:
            acertos += 1
            
        # 2. Backpropagation (Retropropagação Manual)
        dz2 = a2.copy()
        dz2[0, y_real] -= 1  # Gradiente na saída
        
        dW2 = np.dot(a1.T, dz2)
        db2 = dz2
        
        da1 = np.dot(dz2, W2.T)
        dz1 = da1 * (z1 > 0) # Derivada da ReLU
        
        dW1 = np.dot(x.T, dz1)
        db1 = dz1
        
        # Atualização dos pesos sinápticos
        W2 -= taxa_aprendizagem * dW2
        b2 -= taxa_aprendizagem * db2
        W1 -= taxa_aprendizagem * dW1
        b1 -= taxa_aprendizagem * db1
        
    precisao = (acertos / NUM_AMOSTRAS_TREINO) * 100
    perda_media = perda_total / NUM_AMOSTRAS_TREINO
    print(f"  [Concluído] Época {epoca + 1} -> Exatidão: {precisao:.2f}% | Erro: {perda_media:.4f}\n")

print("-> Treino matemático concluído com sucesso! Modelo ativo na memória.")


# --- PASSO 4: MENU INTERATIVO DE TESTES ---
imagens_teste = ["teste1.jpg", "teste2.jpg", "teste3.jpg", "teste4.jpg", "teste5.jpg"]

while True:
    print("\n" + "="*50)
    print("=== MENU DE TESTES DO MODELO NUMPY ===")
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
            print("\n[Aviso] Ficheiros de teste ou etiquetas não encontrados. Escolha a opção [1] primeiro.")
            continue
    else:
        print("\n[Aviso] Opção inválida. Escolha 1, 2 ou 3.")
        continue
        
    # Execução da Inferência
    print("\n=== A EXECUTAR INFERÊNCIA COM MATEMÁTICA PURA (NUMPY) ===")
    total_acertos = 0
    total_erros = 0
    
    for indice, nome_ficheiro in enumerate(imagens_teste):
        if os.path.exists(nome_ficheiro):
            img = Image.open(nome_ficheiro).convert('L').resize((28, 28))
            img_arr = np.array(img, dtype=float) / 255.0
            
            # Aplicar pipeline manual de extração
            c = convolucao_2d(img_arr, kernel_sharpen)
            p = max_pooling_2d(c, pool_size=2)
            f = flattening(p).reshape(1, -1)
            max_f = np.max(f)
            if max_f > 0:
                f = f / max_f
                
            # Forward Pass de teste
            z1_t = np.dot(f, W1) + b1
            a1_t = np.maximum(0, z1_t)
            z2_t = np.dot(a1_t, W2) + b2
            probabilidades = softmax(z2_t)
            
            digito_previsto = np.argmax(probabilidades)
            confianca = np.max(probabilidades) * 100
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