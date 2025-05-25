from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

def analise_sentimento(texto: str) -> str:
    print("CUDA disponível?", torch.cuda.is_available())
    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    tokenizer = LlamaTokenizer.from_pretrained(r"app/emollama_local", local_files_only=True)

    model = LlamaForCausalLM.from_pretrained(
        r"app/emollama_local",
        local_files_only=True,
        device_map='cuda',
        torch_dtype=torch.float16,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt = f'''Humano:
    Tarefa: Categorize a emoção expressada no texto como 'neutro' ou identifique a presença de uma ou mais emoções (satisfação, frustração, confusão, pressão, raiva).
    Texto: {texto}
    Esse texto contém a emoção:

    Assistente:
    '''

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")

    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    resposta_final = response.split("Assistente:")[-1].strip()

    print("Resposta:", resposta_final)

analise_sentimento("Estou feliz!!")
