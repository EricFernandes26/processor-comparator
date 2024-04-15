import os
import boto3
import json
import matplotlib.pyplot as plt
import sys
import pandas as pd


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

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Erro ao ler o arquivo Excel: {e}")
        return None

def get_processor_info(processor, excel_df):
    if excel_df is None:
        return None
    
    processor_info = excel_df.loc[excel_df['CPU_Name'] == processor].to_dict('records')
    if processor_info:
        return processor_info[0]
    else:
        return None

def main():
    FIELD_ATTRIBUTES = ['CPU_Name', 'Cores', 'Threads', 'Base_GHz', 'Turbo_GHz', 'CacheMB', 'Benchmark_Single_Thread']
    try:
        processor1 = str(input("Digite o nome do primeiro processador: ")).strip()
        processor2 = str(input("Digite o nome do segundo processador: ")).strip()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        excel_file_path = os.path.join(script_dir, "dataset02.xlsx")
        bedrock_client = boto3.client(service_name='bedrock-runtime')
    except Exception as e:
        print(f"Erro ao conectar com o cliente do Boto3: {e}")
        sys.exit(1)

    excel_data = read_excel(excel_file_path)

    processor1_info = get_processor_info(processor1, excel_data)
    processor2_info = get_processor_info(processor2, excel_data)

    if processor1_info is None or processor2_info is None:
        print("Informações do processador não encontradas no arquivo Excel. Usando Bedrock para recuperar informações.")
        processors = json.loads(bedrock_query(client=bedrock_client, field_attributes=FIELD_ATTRIBUTES, processor1=processor1, processor2=processor2).replace("'", '"'))
        if processor1_info is None:
            processor1_info = processors[0]
        if processor2_info is None:
            processor2_info = processors[1]

    # Criar três conjuntos de atributos para os gráficos individuais
    attributes_sets = [
        ['Cores', 'Threads'],
        ['Base_GHz', 'Turbo_GHz'],
        ['Benchmark_Single_Thread'],
        ['CacheMB']
        
    ]

    # Converter valores para float, lidando com valores nulos
    processor1_values = []
    processor2_values = []
    for attributes in attributes_sets:
        processor1_values.append([float(processor1_info.get(attribute, 0.0)) for attribute in attributes])
        processor2_values.append([float(processor2_info.get(attribute, 0.0)) for attribute in attributes])

    # Criar três gráficos individuais
    for i, attributes in enumerate(attributes_sets):
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(attributes))

        # Create bars for each processor
        bar1 = ax.bar(index, processor1_values[i], bar_width, label=processor1_info.get('CPU_Name'))
        bar2 = ax.bar([i + bar_width for i in index], processor2_values[i], bar_width, label=processor2_info.get('CPU_Name'))

        for bars in [bar1, bar2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(round(height, 2)),  # Display rounded value
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # Define os nomes ao lado esquerdo de cada gráfico
        #ax.set_ylabel('Qtd_Cores_Threads' if i == 0 else 'Frequência_Base_Turbo' if i == 1 else 'Pontos_Single_Threads' if i == 2 else 'Qtd_CacheMB')

        ax.set_xlabel('Attributes')
        ax.set_ylabel('Qtd_Cores_Threads' if i == 0 else 'Frequência_Base_Turbo' if i == 1 else 'Pontos_Single_Threads' if i == 2 else 'Qtd_CacheMB')

        # Define a margem y para o terceiro gráfico
        if i == 2:
            ax.set_ylim([0, 3000])  # Define a margem dos valores do eixo y para 3000 pontos

        ax.set_title(f'Comparison of {processor1_info.get("CPU_Name")} and {processor2_info.get("CPU_Name")} - Attributes: {", ".join(attributes)}')
        ax.set_xticks([i + bar_width / 2 for i in index])
        ax.set_xticklabels(attributes)
        ax.legend()

        # Salva o gráfico como .png no diretório específico
        plt.savefig(f"C:\\Users\\Administrator\\Desktop\\ok\\saved_images\\comparison_{i + 1}.png")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
