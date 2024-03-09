import boto3
import json
import matplotlib.pyplot as plt
import sys

def bedrock_query(client, field_attributes, processor1, processor2):
    prompt = f"\n\nHuman: Olá, preciso de informações sobre a performance dos processadores {processor1} e {processor2}. Me devolva essas informações em formato de uma lista com dois dicionarios em Python (exemplo: [<dict>, <dict>]), apenas o dicionário, sem mais textos antes ou depois. Dentro de cada dicionário irei precisar dos campos: {field_attributes}.\n\nAssistant:"
    
    body = json.dumps({
        "prompt": prompt,
        "max_tokens_to_sample": 300,
        "temperature": 0.1,
        "top_p": 0.9,
    })

    modelId = 'anthropic.claude-v2'
    accept = 'application/json'
    contentType = 'application/json'

    response = client.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    response_body = json.loads(response.get('body').read())
    processors = response_body.get('completion')

    return processors.strip()

def main():
    FIELD_ATTRIBUTES = ['NomeProcessador', 'Nucleos', 'Threads', 'FrequenciaBase', 'FrequenciaTurbo', 'Cache', 'TDP']
    try:
        processor1 = str(input("Digite o nome do primeiro processador: ")).strip()
        processor2 = str(input("Digite o nome do segundo processador: ")).strip()
        bedrock_client = boto3.client(service_name='bedrock-runtime')
    except Exception as e:
        print(f"Erro ao conectar com o cliente do Boto3: {e}")
        sys.exit(1)

    processors = json.loads(bedrock_query(client=bedrock_client, field_attributes=FIELD_ATTRIBUTES, processor1=processor1, processor2=processor2).replace("'", '"'))
    attribute_names = ['Nucleos', 'Threads', 'FrequenciaBase', 'FrequenciaTurbo', 'Cache', 'TDP']

    processor1_values = [float(processors[0].get(attribute)) for attribute in attribute_names ]
    processor2_values = [float(processors[1].get(attribute)) for attribute in attribute_names ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = range(len(attribute_names))

    # Create bars for each processor
    bar1 = ax.bar(index, processor1_values, bar_width, label=processors[0].get('NomeProcessador'))
    bar2 = ax.bar([i + bar_width for i in index], processor2_values, bar_width, label=processors[1].get('NomeProcessador'))

    for bars in [bar1, bar2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(round(height, 2)),  # Display rounded value
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_xlabel('Attributes')
    ax.set_ylabel('Values')
    ax.set_title(f'Comparison of {processors[0].get("NomeProcessador")} and {processors[1].get("NomeProcessador")}')
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(attribute_names)
    ax.legend()

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

