# 🧠 Emollama - Classificador de Emoções com LLM

Este repositório contém uma implementação simples e prática de um modelo baseado em LLaMA para **análise de emoções em textos**.  
Você poderá rodá-lo localmente com o modelo `Emollama-7b` usando a ferramenta [Ollama](https://ollama.com).

---

## 🚀 Como rodar

### ✅ Requisitos

Certifique-se de ter instalado:

- Python 3.10 ou superior
- [Git](https://git-scm.com/)
- [VS Code](https://code.visualstudio.com/) (ou outra IDE de sua preferência)
- [Ollama](https://ollama.com) (executando em segundo plano)
- transformers
- torch
- accelerate
- sentencepiece

Além disso, instale as dependências do Python:

```bash
pip install transformers torch accelerate sentencepiece
```

---

### 📦 Clonando o repositório

```bash
git clone https://github.com/Lual007/Modelo_Sentimentos_Emollama.git
cd Modelo_Sentimentos_Emollama
```

---

### 🧠 Instalando e iniciando o Ollama

1. Baixe o instalador no site: https://ollama.com
2. Instale normalmente conforme o seu sistema.
3. **Execute o Ollama.**
   - No Windows, procure por "Ollama" no menu iniciar.
   - Ele roda em segundo plano. Verifique no **Gerenciador de Tarefas** se está ativo.
4. Agora abra o terminal e rode:

```bash
ollama pull lzw1008/Emollama-7b
```

> Esse comando faz o download do modelo que será usado localmente.

---

### ▶️ Executando o código

Abra o terminal na pasta do projeto e execute o script `main.py`, ou use o seguinte exemplo básico em um script Python:

```python
from transformers import LlamaTokenizer, LlamaForCausalLM

tokenizer = LlamaTokenizer.from_pretrained('lzw1008/Emollama-7b')
model = LlamaForCausalLM.from_pretrained('lzw1008/Emollama-7b', device_map='auto')

entrada = """
Humano:
Tarefa: Categorize o tom emocional do texto como 'neutro ou sem emoção' ou identifique a 
presença de uma ou mais das emoções fornecidas (raiva, ansiedade, nojo, medo, alegria, 
amor, otimismo, pessimismo, tristeza, surpresa, confiança).
Texto: Qualquer coisa que você decida fazer tenha a certeza de que isso te faz feliz.  
Esse texto apresenta as emoções:

Assistente:
"""

```

---

### 🌐 Suporte ao Português

Embora o prompt original esteja em inglês, o modelo suporta entradas em **português** sem problemas.

---

### 📁 Estrutura do Projeto

- `modelo.py`: Código principal que executa o modelo.
- `README.md`: Este arquivo com instruções completas.

---

### 📬 Contato

Contribuições, sugestões ou dúvidas são bem-vindas!  
Abra uma issue ou entre em contato diretamente pelo GitHub.
