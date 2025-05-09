#!pip install streamlit openai
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
from pycaret.classification import *  # load_model, predict_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import tempfile
import shap
import requests
import os

# Função para interpretar o threshold via IA
def interpretar_threshold(comando_usuario):
    prompt = f"Extraia apenas o valor numérico do threshold (entre 0 e 1) baseado no comando: '{comando_usuario}'."
    try:
        token = st.secrets["gpt_token"]
        client = OpenAI(api_key=token)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system",
                 "content": "Você é um assistente que extrai valores numéricos de thresholds entre 0 e 1."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        resposta = response.choices[0].message.content.strip()
        valor = float(resposta.replace(",", "."))
        if 0 <= valor <= 1:
            return ["Novo threshold interpretado", valor, True]
        else:
            return ["Por favor, informe um valor entre 0 e 1. Utilizando valor padrão", 0.5, False]
    except Exception as e:
        #st.error(f"Erro ao interpretar comando: {e}")
        return ["Por favor, informe um valor entre 0 e 1. Utilizando valor padrão", 0.5, False]

# Configurações iniciais do Streamlit
st.set_page_config(page_title='Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Simulador - Conversão de Vendas')

# Integrantes
with st.expander('Integrantes (RM - NOME)', expanded=True):
    st.write('''
        - 556984 Tiago Toshio Kumagai Gibo
        - 554668 Israel Dalcin Alves Diniz
        - 556213 João Pires da Silva
        - 555762 Ana Carolina Martins da Silva
    ''')

# Explicação
with st.expander('Descrição do App', expanded=False):
    st.write('''
    
        Este aplicativo tem como objetivo apoiar estratégias de marketing e vendas, permitindo simular, de forma simples e interativa, a chance de conversão de um cliente em potencial. A partir do preenchimento de informações de um cliente, o sistema utiliza um modelo de inteligência artificial treinado previamente para indicar se esse perfil tem maior ou menor propensão a adquirir um determinado produto.

        A plataforma também permite que o usuário ajuste o nível de rigor da análise (threshold) com um controle deslizante ou através de comandos em linguagem natural, o que torna a experiência mais personalizada e acessível.

        Além disso, o aplicativo oferece uma aba de análises comparativas, que destaca visualmente as principais diferenças entre os perfis de clientes que costumam comprar e os que não compram. Com gráficos claros e dinâmicos, é possível entender, por exemplo, quais características mais influenciam a decisão de compra — como renda, frequência de compras ou tempo desde a última interação.

        Essa solução foi pensada para ajudar equipes de vendas e marketing a tomar decisões mais informadas, segmentar melhor suas campanhas e aumentar as taxas de conversão, utilizando dados de forma estratégica e acessível.
    
    ''')

# Sidebar
st.sidebar.write('Configurações')
with st.sidebar:
    c1, c2 = st.columns([.3, .7])
    c1.image('./images/logo_fiap.png', width=100)
    c2.subheader('Case Ifood')
    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal=True)

    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

# Tela principal
if database == 'CSV':
    if file:
        # Carregamento do CSV
        Xtest = pd.read_csv(file)

        # Carregamento / instanciamento do modelo pkl
        mdl_rf = load_model('./pickle/pickle_rf_pycaret2')

        # Predict do modelo
        ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)

        with st.expander('Visualizar CSV carregado:', expanded=False):
            c1, _ = st.columns([2, 4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:',
                                   min_value=5,
                                   max_value=Xtest.shape[0],
                                   step=10,
                                   value=5)
            st.dataframe(Xtest.head(qtd_linhas))


        with st.expander('Visualizar Predições / Análises:', expanded=True):

            # Campo de texto para comando
            comando_usuario = st.text_input('Digite o comando para alterar o threshold (ex: "Aumente para 0.7"):',
                                            '')

            # Threshold inicial
            treshold = 0.5

            if comando_usuario:
                msg, treshold, success = interpretar_threshold(comando_usuario)

                if success:
                    st.success(f"{msg}: {treshold}")
                else:
                    st.warning(f"{msg}: {treshold}")

            # Slider para ajuste fino, já usando o threshold interpretado
            treshold = st.slider('Ajuste manual do Threshold:',
                                 min_value=0.0,
                                 max_value=1.0,
                                 step=0.01,
                                 value=treshold)

            predicoes, analises = st.tabs(["Predições", "Análises"])
            with predicoes:

                qtd_true = ypred.loc[ypred['prediction_score_1'] > treshold].shape[0]

                c1, _, c2, c3 = st.columns([.5, .1, .2, .2])
                c2.metric('Qtd comprou', value=qtd_true)
                c3.metric('Qtd não comprou', value=len(ypred) - qtd_true)


                # Função para colorir as predições
                def color_pred(val):
                    color = 'darkgreen' if val > treshold else 'firebrick'
                    return f'background-color: {color}'


                tipo_view = st.radio('Visualizar:', ('Completo', 'Apenas predições'))
                if tipo_view == 'Completo':
                    df_view = ypred.copy()
                else:
                    df_view = pd.DataFrame(ypred.iloc[:, -1].copy())

                st.dataframe(df_view.style.applymap(color_pred, subset=['prediction_score_1']))

                csv = df_view.to_csv(sep=',', decimal=',', index=True)
                st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
                st.download_button(label='Download CSV',
                                   data=csv,
                                   file_name='Predicoes.csv',
                                   mime='text/csv')
                

            with analises:
                st.markdown('### 🎯 Comprou vs Não comprou / Investimento vs Retorno')
                # Pie chart com tamanho e fonte ajustados
                fig, ax = plt.subplots(figsize=(4, 4), dpi=300, facecolor='#1F1D22')  # gráfico menor
                labels = ['Comprou', 'Não comprou']
                sizes = [qtd_true, len(ypred) - qtd_true]
                colors = ['darkgreen', 'firebrick']
                explode = (0, 0.05)

                ax.pie(
                    sizes,
                    labels=labels,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=colors,
                    explode=explode,
                    shadow=True,
                    textprops={'fontsize': 6, 'color': 'white'},      # ← aqui controla o tamanho da fonte
                    labeldistance=1.1                # ← distância do rótulo até o centro
                )
                ax.set_facecolor('white')
                ax.axis('equal')
                col1, col2 = st.columns([0.5, 0.5])
                with col1:
                    st.pyplot(fig)

                # Valores
                investimento = 2000
                retorno = qtd_true * 50
                roi = (retorno - investimento) / investimento

                # Ordem fixa: Retorno em cima, Investimento embaixo
                labels = ['Investimento', 'Retorno']
                valores = [investimento, retorno]
                cores = [ 'gray', 'limegreen']

                # Criação do gráfico
                fig, ax = plt.subplots(figsize=(4, 3.45), dpi=300, facecolor='#1F1D22')
                bars = ax.barh(labels, valores, color=cores)

                # Anotar os valores das barras
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + max(valores) * 0.02, bar.get_y() + bar.get_height() / 2.5,
                            f'R$ {width:,.0f}', va='center', fontsize=6, color='white')

                # Linha vertical no ponto do retorno e ROI como texto
                ax.axvline(x=retorno, color='#23E04A' if roi >= 0 else 'red', linestyle='--', linewidth=1)
                roi_color = '#23E04A' if roi >= 0 else 'red'
                ax.text(retorno + 50, 1.2, f'ROI: {roi * 100:.2f}%', 
                        ha='left', fontsize=8, color=roi_color)

                # Estética
                ax.set_facecolor('#1F1D22')
                fig.patch.set_facecolor('#1F1D22')
                ax.tick_params(axis='x', colors='white', labelsize=6)
                ax.tick_params(axis='y', colors='white', labelsize=6)
                ax.spines[:].set_color('white')
                ax.set_xlabel('Valor (R$)', color='white', fontsize=6)
                ax.set_xlim(0, max(valores) * 1.3)
                ax.grid(color='gray', linestyle=':', linewidth=0.5, axis='x')

                with col2:
                    st.pyplot(fig)


                st.markdown('### 📊 Interpretação do Modelo com SHAP / Feature importances')
                # st.write(mdl_rf.named_steps)  
                pure_model = mdl_rf.named_steps['trained_model']
                explainer = shap.TreeExplainer(pure_model)
                shap_values = explainer.shap_values(Xtest)

                # Plot na memória
                fig, ax = plt.subplots(figsize=(6, 6), dpi=300, facecolor='#1F1D22')
                shap.summary_plot(shap_values, Xtest, show=False)

                # Ajusta os textos para branco
                for text in plt.gca().get_yticklabels():
                    text.set_color("white")
                for text in plt.gca().get_xticklabels():
                    text.set_color("white")

                plt.xlabel(plt.gca().get_xlabel(), color='white')
                plt.ylabel(plt.gca().get_ylabel(), color='white')

                # Acessa o colorbar manualmente e altera as cores
                # OBS: o colorbar é adicionado automaticamente como o último artista do plot
                cbar = plt.gcf().axes[-1]  # último eixo da figura
                cbar.tick_params(colors='white')  # deixa ticks em branco
                cbar.yaxis.label.set_color('white')  # título da barra lateral (Feature value)
                cbar.set_title(cbar.get_title(), color='white')  # redundante por segurança
                
                col3, col4 = st.columns([0.5, 0.5])
                with col3:
                    st.pyplot(fig)

                #st.markdown('### Feature importance')
                importances = mdl_rf.feature_importances_
                features = Xtest.columns
                feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)  # Ascendente para inverter o gráfico horizontal
                feat_imp_values = np.array(feat_imp.values)

                colors_list = ["#49C9E1", "purple", "red"]
                custom_cmap = LinearSegmentedColormap.from_list("custom_shap", colors_list, N=256)
                # Pegando o colormap coolwarm
                cmap = custom_cmap

                # Usa os valores brutos para calcular a intensidade da cor com base em proporção simples
                min_val = feat_imp_values.min()
                max_val = feat_imp_values.max()
                range_val = max_val - min_val if max_val != min_val else 1  # evita divisão por zero

                # Aplica fade manual sem Normalize
                bar_colors = [cmap((val - min_val) / range_val) for val in feat_imp_values]

                # Gráfico
                fig, ax = plt.subplots(figsize=(6, 7.15), dpi=300, facecolor='#1F1D22')
                ax.set_facecolor('#1F1D22')

                ax.barh(feat_imp.index, feat_imp_values, color=bar_colors, height=0.5)

                ax.set_xlabel('Importância', fontsize=12, color='white')
                ax.set_ylabel('Features', fontsize=12, color='white')
                ax.tick_params(axis='x', colors='white', labelsize=10)
                ax.tick_params(axis='y', colors='white', labelsize=8)

                ax.grid(True, linestyle='--', alpha=0.3, color='white')
                for spine in ax.spines.values():
                    spine.set_visible(False)

                fig.tight_layout(pad=2)
                with col4:
                    st.pyplot(fig)
    else:
        st.warning('Arquivo CSV não foi carregado.')

else:
    st.header("Simulação online")
    st.write("Preencha todos os campos para visualizar a previsão do modelo para um cliente específico")
    with st.expander("Formulário para a previsão", expanded=True): 
        AcceptedCmp1 = st.selectbox(
            "Aceitou campanha 1", [None, 0, 1],
            format_func=lambda x: "Selecione…" if x is None else ("Sim" if x == 1 else "Não"),
            key="AcceptedCmp1"
        )
        AcceptedCmp2 = st.selectbox("Aceitou campanha 2", [None, 0, 1],
                                   format_func=lambda x: "Selecione…" if x is None else ("Sim" if x == 1 else "Não"),
                                   key="AcceptedCmp2")
        AcceptedCmp3 = st.selectbox("Aceitou campanha 3", [None, 0, 1],
                                   format_func=lambda x: "Selecione…" if x is None else ("Sim" if x == 1 else "Não"),
                                   key="AcceptedCmp3")
        AcceptedCmp4 = st.selectbox("Aceitou campanha 4", [None, 0, 1],
                                   format_func=lambda x: "Selecione…" if x is None else ("Sim" if x == 1 else "Não"),
                                   key="AcceptedCmp4")
        AcceptedCmp5 = st.selectbox("Aceitou campanha 5", [None, 0, 1],
                                   format_func=lambda x: "Selecione…" if x is None else ("Sim" if x == 1 else "Não"),
                                   key="AcceptedCmp5")
        Complain = st.selectbox("Reclamação", [None, 0, 1],
                                format_func=lambda x: "Selecione…" if x is None else ("Sim" if x == 1 else "Não"),
                                key="Complain")
        age_str = st.text_input("Idade", "", key="Age")
        education = st.selectbox(
            "Escolaridade",
            [None, 1, 2, 3, 4, 5],
            format_func=lambda x: {
                None: "Selecione…",
                1: "Basic",
                2: "2n Cycle",
                3: "Graduation",
                4: "Master",
                5: "PhD"
            }.get(x, "Selecione…"),
            key="Education"
        )
        # Opções do status marital
        marital_options = {
            "Divorced": "Marital_Status_Divorced",
            "Married": "Marital_Status_Married",
            "Single": "Marital_Status_Single",
            "Together": "Marital_Status_Together",
            "Widow": "Marital_Status_Widow"
        }

        # Campo de seleção único
        marital_choice = st.selectbox("Estado Civil", list(marital_options.keys()))

        # Cria as colunas one-hot com True para a escolhida e False para as demais
        marital_status_encoded = {
            column: (column == marital_options[marital_choice])
            for column in marital_options.values()
        }
        kidhome_str = st.text_input("Filhos em Casa", "", key="Kidhome")
        teenhome_str = st.text_input("Adolescentes em Casa", "", key="Teenhome")
        income_str = st.text_input("Renda (R$)", "", key="Income")
        fish_str = st.text_input("Gasto em Peixes", "", key="MntFishProducts")
        fruits_str = st.text_input("Gasto em Frutas", "", key="MntFruits")
        gold_str = st.text_input("Gasto em produtos de ouro", "", key="MntGoldProds")
        meat_str = st.text_input("Gasto em Carnes", "", key="MntMeatProducts")
        sweet_str = st.text_input("Gasto em Doces", "", key="MntSweetProducts")
        wine_str = st.text_input("Gasto em Vinhos", "", key="MntWines")
        cat_str = st.text_input("Compras por Catálogo", "", key="NumCatalogPurchases")
        deals_str = st.text_input("Compras por Promoções", "", key="NumDealsPurchases")
        store_str = st.text_input("Compras na Loja Física", "", key="NumStorePurchases")
        web_str = st.text_input("Compras Online", "", key="NumWebPurchases")
        visits_str = st.text_input("Visitas ao Site no mês", "", key="NumWebVisitsMonth")
        recency_str = st.text_input("Dias desde a última Compra", "", key="Recency")
        time_str = st.text_input("Tempo como cliente (anos)", "", key="Time_Customer")

    if st.button("Prever o resultado"):
        try:
            # Cria o dicionário base com os campos preenchidos
            inputs = {
                "AcceptedCmp1": st.session_state["AcceptedCmp1"],
                "AcceptedCmp2": st.session_state["AcceptedCmp2"],
                "AcceptedCmp3": st.session_state["AcceptedCmp3"],
                "AcceptedCmp4": st.session_state["AcceptedCmp4"],
                "AcceptedCmp5": st.session_state["AcceptedCmp5"],
                "Complain": st.session_state["Complain"],
                "Age": age_str,
                "Education": st.session_state["Education"],
                "Kidhome": kidhome_str,
                "Teenhome": teenhome_str,
                "Income": income_str,
                "MntFishProducts": fish_str,
                "MntFruits": fruits_str,
                "MntGoldProds": gold_str,
                "MntMeatProducts": meat_str,
                "MntSweetProducts": sweet_str,
                "MntWines": wine_str,
                "NumCatalogPurchases": cat_str,
                "NumDealsPurchases": deals_str,
                "NumStorePurchases": store_str,
                "NumWebPurchases": web_str,
                "NumWebVisitsMonth": visits_str,
                "Recency": recency_str,
                "Time_Customer": time_str,
            }

            # Verifica se há campos não preenchidos
            if any(v in (None, "") for v in inputs.values()):
                st.error("Por favor, preencha **TODOS** os campos antes de rodar.")
            else:
                # Converte os tipos e junta com as colunas one-hot de Marital Status
                df_input = pd.DataFrame([{
                    **{f: int(inputs[f]) for f in [
                        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Complain"
                    ]},
                    "Age": int(inputs["Age"]),
                    "Kidhome": int(inputs["Kidhome"]),
                    "Teenhome": int(inputs["Teenhome"]),
                    "Income": float(inputs["Income"]),
                    **{k: int(inputs[k]) for k in [
                        "MntFishProducts", "MntFruits", "MntGoldProds", "MntMeatProducts", "MntSweetProducts", "MntWines"
                    ]},
                    **{k: int(inputs[k]) for k in [
                        "NumCatalogPurchases", "NumDealsPurchases", "NumStorePurchases",
                        "NumWebPurchases", "NumWebVisitsMonth", "Recency"
                    ]},
                    "Time_Customer": float(inputs["Time_Customer"]),
                    "Education": inputs["Education"],
                    **marital_status_encoded  # Adiciona as colunas booleanas one-hot
                }])

                with st.expander("CSV do usuário:", expanded=True):
                    st.dataframe(df_input)

                # Aqui você pode chamar o modelo
                mdl_rf = load_model("./pickle/pickle_rf_pycaret2")
                ypred = predict_model(mdl_rf, data=df_input, raw_score=True)
                prob_true = float(ypred["prediction_score_1"][0])
                st.session_state["prob_true"] = prob_true

        except Exception as e:
            st.error(f"Erro ao processar os dados: {e}")

    
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.5
        
    if "prob_true" in st.session_state:
        with st.expander("Resultado da Simulação", expanded=True):
            # Campo de texto para comando
            comando_usuario = st.text_input('Digite o comando para alterar o threshold (ex: "Aumente para 0.7"):',
                                            '')

            # Threshold inicial
            treshold = 0.5

            if comando_usuario:
                msg, treshold, success = interpretar_threshold(comando_usuario)

                if success:
                    st.success(f"{msg}: {treshold}")
                else:
                    st.warning(f"{msg}: {treshold}")

            prob_true = st.session_state["prob_true"]

            # Slider agora usa key para ler/escrever direto no session_state
            threshold = st.slider(
                "Limiar de corte",
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                value=treshold
            )

            pred_label = "Sim" if prob_true >= threshold else "Não"
            st.metric("Probabilidade de Compra", f"{prob_true:.3f}")
            st.write(f"Classificação: **{pred_label}**")


