from PIL import Image
import os

# Caminho da pasta com as imagens
input_folder = '/Users/gregorio/Library/CloudStorage/GoogleDrive-gregoriokoslinskineto@gmail.com/My Drive/TrabalhoRN/Gatos'
output_folder = '/Users/gregorio/Library/CloudStorage/GoogleDrive-gregoriokoslinskineto@gmail.com/My Drive/TrabalhoRN/Gatos1'

# Cria a pasta de saída se ela não existir
os.makedirs(output_folder, exist_ok=True)

# Percorre todos os arquivos na pasta de entrada
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    
    # Tenta abrir o arquivo como uma imagem
    try:
        with Image.open(file_path) as img:
            # Converte a imagem para RGB se necessário
            img = img.convert('RGB')
            
            # Salva a imagem como JPG na pasta de saída
            new_filename = os.path.splitext(filename)[0] + '.jpg'
            output_path = os.path.join(output_folder, new_filename)
            img.save(output_path, 'JPEG')
            print(f"Convertido: {filename} -> {new_filename}")
            
    except Exception as e:
        print(f"Erro ao converter {filename}: {e}")