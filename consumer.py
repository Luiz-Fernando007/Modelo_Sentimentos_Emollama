import pika
from app.models import analise_sentimento 
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
print("PORT =", os.getenv("PORT")) 

class RabbitmqConsumer:
    def __init__(self, callbak) -> None:
        self.__host = os.getenv("RABBITMQ_HOST")
        self.__port = int(os.getenv("PORT"))
        self.__username = os.getenv("RABBITMQ_USER")
        self.__password = os.getenv("RABBITMQ_PASSWORD")
        self.__exchange = os.getenv("RABBITMQ_EXCHANGE")
        self.__queue = "data_queue4"
        self.__callback = callbak
        self.__channel = self.__create_channel()

    def __create_channel(self):
        connection_parameters = pika.ConnectionParameters(
        host=self.__host,
        port=self.__port,
        credentials=pika.PlainCredentials(
            username=self.__username,
            password=self.__password
        )
        )

        channel = pika.BlockingConnection(connection_parameters).channel()

        channel.exchange_declare(
            exchange=self.__exchange,
            exchange_type="direct",
            durable=True
        )

        channel.queue_declare(
            queue=self.__queue,
            durable=True,
            arguments={
                "x-overflow": "reject-publish"
            }
        )

        channel.queue_bind(
            exchange=self.__exchange,
            queue=self.__queue,
            routing_key="minha_routing_key"
        )

        channel.basic_consume(
            queue=self.__queue,
            auto_ack=True,
            on_message_callback=self.__callback
        )

        return channel
    
    def close_connection(self):
        if self.__channel:
            self.__channel.close()
        if self.__channel.connection:
            self.__channel.connection.close()

    def start(self):
        print(f"Listen RabbitMQ on Port 8081")
        self.__channel.start_consuming()


def minha_callback(ch, method, properties, body):
    try:
        body_str = body.decode('utf-8')
        body_dict = json.loads(body_str)
        acao_id = body_dict.get("acao_id")
        user_id = body_dict.get("user_id")
        agent_id = body_dict.get("agent_id")
        descricao = body_dict.get("descricao")
        data = body_dict.get("data_acao")
        print(f"Mensagem JSON como dict: {body_dict}")
        texto = body_dict.get("descricao", "")
        if not texto:
            print("Mensagem vazia recebida.")
            return
        print(f"Mensagem recebida: {texto}")

        resultado = analise_sentimento(descricao)
        print(f"Resultado da análise de sentimento: {resultado}")

        api_url = os.getenv("API_DESTINO_URL") 
     
        payload = {
            "acao_id": acao_id,
            "user_id": user_id,
            "agent_id": agent_id,
            "sentimento": resultado,
            "score": 1,
            "data_analise": data,
        }
        response = requests.post(api_url, json=payload)

        if response.status_code in (200, 201):
            print(f"Resultado enviado com sucesso para a API: {response.json()}")
        else:
            print(f"Erro ao enviar resultado para a API: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Erro ao processar a mensagem: {e}")

if __name__ == "__main__":
    print("Iniciando o consumidor RabbitMQ...")
    rabbitmq = RabbitmqConsumer(minha_callback)
    rabbitmq.start()
