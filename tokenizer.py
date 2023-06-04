import os
import nltk
from nltk.tokenize import word_tokenize
import pickle

nltk.download('punkt')

all_tokens = []

dir_path = 'TR2'

if not os.path.isdir(dir_path):
    print(f"O diretório {dir_path} não existe!")
else:
    txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]

    if not txt_files:
        print(f"Não há arquivos .txt no diretório {dir_path}!")
    else:
        for filename in txt_files:
            with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                
                if not text:
                    print(f"O arquivo {filename} está vazio!")
                else:
                    tokens = word_tokenize(text)
                    all_tokens.extend(tokens)
                    
        if not all_tokens:
            print("Nenhum token foi criado!")
        else:
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(all_tokens, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("O arquivo tokenizer.pickle foi criado com sucesso!")

