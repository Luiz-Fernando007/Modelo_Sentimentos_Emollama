from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained('lzw1008/Emollama-7b')
model = LlamaForCausalLM.from_pretrained('lzw1008/Emollama-7b', device_map='auto')

prompt = '''Humano:
Tarefa: Categorize a emoção expressada no texto como 'neutro ou sem emoção' ou identifique a presença de uma ou mais emoções das listadas (raiva, confusão, ansiedade, desgosto, medo, alegria, amor, otimismo, pessimismo, tristeza, surpresa, confiança).
Texto: Como assim???? Como você fez isso?
Esse texto contem a emoção:

Assistente:
'''

inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs["input_ids"], max_length=256)
response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(response)
