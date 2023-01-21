import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd

# =====================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server= app.server
# =====================================================================
path = "Z:/Clientes/WSS/STF/Avaliacao Atuarial/2022/Recebidos/20230119/"
#======================================================================

dist_idade = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='dist_idade')
dist_tipo = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='dist_tipo')
dist_classe = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='dist_classe')
morbidade_media = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='morbidade_media')
severidade_media = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='severidade_media')
morbidade_media_agp = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='morbidade_media_agp')
severidade_media_agp = pd.read_excel(path+'graficos/database/database.xlsx',  engine='openpyxl', sheet_name='severidade_media_agp')

#=========================== Gráfico 1===========================================

fig = px.bar(dist_idade,
             x="Idade",             
             y="Qtd.",
             #color='Plano',
             title='Gráfico 1 - Distribuição de beneficiários por idade',

            )

fig.update_layout(height=480, width=960,
    legend_title_text = None,
    font_family="Neulis Alt",

    title={
        #'text': "Plot Title",
        'y':0.83,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    
    )

fig.update_traces(marker_color='rgb(93,138,167)', marker_line_color='rgb(0,62,76)',
                  marker_line_width=1.5, opacity=0.85)

#fig.show()

#=========================== Gráfico 2 ===========================================

fig2 = px.line(dist_tipo,x='Idade',y='Qtd.',color='TIPO BENEFICIARIO', title='Gráfico 2 - Distribuição de beneficiários por tipo',markers=False)

fig2.update_layout(height=480, width=960,font_family="Neulis Alt",
    title={
        #'text': "Plot Title",
        'y':0.83,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    
    yaxis = dict(
        tickmode = 'linear',
        #tick0 = 0,
        dtick = 25
    ),
    xaxis = dict(
        tickmode = 'linear',
        #tick0 = 0,
        dtick = 3
    ),
   legend=dict(
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        orientation="v",
        bgcolor='rgba(255,255,255,0.6)')
    
)

#fig.show()
#=========================== Gráfico 3 ===========================================

fig3 = px.line(dist_classe,x='Idade',y='Qtd.',color='Classe', title='Gráfico 3 - Distribuição de beneficiários por classe',markers=False)

fig3.update_layout(height=480, width=960,font_family="Neulis Alt",
    title={
        #'text': "Plot Title",
        'y':0.83,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    
    yaxis = dict(
        tickmode = 'linear',
        #tick0 = 0,
        dtick = 25
    ),
    xaxis = dict(
        tickmode = 'linear',
        #tick0 = 0,
        dtick = 3
    ),
   legend=dict(
        yanchor="top",
        y=1,
        xanchor="right",
        x=1,
        orientation="v",
        bgcolor='rgba(255,255,255,0.6)')
    
)
#=========================== Gráfico 4 ===========================================
periodo = [2016,2017,2018,2019,2020,2021,2022]
str_periodo = list(map(str, periodo))

fig4 = px.line(morbidade_media, x=morbidade_media.index, y=periodo, title='Morbidade Média',markers=False,color_discrete_sequence= px.colors.sequential.Teal)

fig4.update_layout(
    height=480, width=960,
    font_family="Neulis Alt",
    
    xaxis_title="Idade",
    yaxis_title=None,
    
    title={
        #'text': "Plot Title",
        'y':0.83,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},

    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 3
    ),
    legend=dict(
        title='Anos',
        yanchor="bottom",
        y=-0.25,
        xanchor="right",
        x=0.8,
        orientation="h",
        bgcolor='rgba(255,255,255,0.6)')

)
#=========================== Layout ===========================================
layout=dict(
    height=480, width=960,
    font_family="Neulis Alt",

    xaxis_title="Idade",
    yaxis_title=None,
    plot_bgcolor='white',

    title={
    #'text': "Plot Title",
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},


    xaxis = dict(
        ticks='outside',
        tickmode = 'linear',
        tick0 = 0,
        dtick = 1
    ),
    legend=dict(
        title='Anos',
        yanchor="top",
        y=1.13,
        xanchor="right",
        x=0.9,
        orientation="h",
        bgcolor='rgba(255,255,255,0.6)')
)

axes=dict(showline=True, linewidth=1, linecolor='gainsboro', gridcolor='aliceblue')

#=========================== Gráfico 5 ===========================================
periodo = [2016,2017,2018,2019,2020,2021,2022]
fig5 = px.line(severidade_media, x=severidade_media.index, y=periodo, title='Severidade Média',markers=False,color_discrete_sequence= px.colors.sequential.Reds)

fig5.update_layout(
    height=480, width=960,
    font_family="Neulis Alt",

    xaxis_title="Idade",
    yaxis_title=None,

    title={
    #'text': "Plot Title",
    'y':0.83,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},


    xaxis = dict(
        tickmode = 'linear',
        tick0 = 0,
        dtick = 3
    ),
    legend=dict(
        title='Anos',
        yanchor="bottom",
        y=-0.25,
        xanchor="right",
        x=0.8,
        orientation="h",
        bgcolor='rgba(255,255,255,0.6)')

)
#=========================== Gráfico 6 ===========================================
periodo = [2016,2017,2018,2019,2020,2021,2022]
fig6 = px.line(morbidade_media_agp, x='FAIXA', y=periodo, title='Morbidade Média',markers=False,color_discrete_sequence= px.colors.sequential.Teal)

fig6.update_layout(layout)
fig6.update_xaxes(axes)
fig6.update_yaxes(axes)
fig6.update_traces(line=dict(width=4))


annotations = []
labels = periodo
y_data= morbidade_media_agp.iloc[-1,1:]

for y_trace, label in zip(y_data, labels):

    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=1, y=y_trace,
                                    xanchor='left', yanchor='middle',
                                    text=label,
                                    font=dict(
                                                size=8),
                                    showarrow=False))

fig6.update_layout(annotations=annotations)


#=========================== Gráfico 7 ===========================================
periodo = [2016,2017,2018,2019,2020,2021,2022]
fig7 = px.line(severidade_media_agp, x='FAIXA', y=periodo, title='Severidade Média',markers=False,color_discrete_sequence= px.colors.sequential.Reds)

fig7.update_layout(layout)

fig7.update_xaxes(axes)
fig7.update_yaxes(axes)
fig7.update_traces(line=dict(width=4))

annotations = []
labels = periodo
y_data= severidade_media_agp.iloc[-1,1:]

for y_trace, label in zip(y_data, labels):

    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=1, y=y_trace,
                                    xanchor='left', yanchor='middle',
                                    text=label,
                                    font=dict(
                                                size=8),
                                    showarrow=False))

fig7.update_layout(annotations=annotations)

#=========================== Gráfico 8 ===========================================
periodo = [2016,2017,2018,2019,2020,2021,2022]

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def format_coefs(coefs):
    equation_list = [f"{coef}x^{i}" for i, coef in enumerate(coefs)]
    equation = " + ".join(equation_list)

    replace_map = {"x^0": "", "x^1": "x", '+ -': '- '}
    for old, new in replace_map.items():
        equation = equation.replace(old, new)

    return equation

df = severidade_media_agp
X = df.index.values.reshape(-1, 1)
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

fig8 = px.scatter(df, x=df.index, y=periodo, opacity=0.65)#,trendline="lowess",trendline_options=dict(frac=0.4))
for degree in [1, 2, 3, 4]:
    poly = PolynomialFeatures(degree)
    poly.fit(X)
    X_poly = poly.transform(X)
    x_range_poly = poly.transform(x_range)

    model = LinearRegression(fit_intercept=False)
    model.fit(X_poly, df[2022])
    y_poly = model.predict(x_range_poly)

    equation = format_coefs(model.coef_.round(2))
    fig8.add_traces(go.Scatter(x=x_range.squeeze(), y=y_poly, name=equation))

# =====================================================================

# Layout 
app.layout = dbc.Container(
    children=[
        dbc.Row([
            dbc.Col([
                    html.Div([
                        
                        html.H5(children="WEDAN APP"),
                        dbc.Button("Avaliação Saúde", color="primary", id="location-button", size="lg")
                    ], ),
                    
                        dcc.Graph(id="line-graph", figure=fig),
                        ], id="grafico1")
                , 

            dbc.Col(
                [
                    dcc.Graph(id="line-graph", figure=fig2),
                        ], id="grafico2")
                ]),
        dbc.Row([     
                dbc.Col(
                    [               
                        dcc.Graph(id="line-graph", figure=fig3),
                        ], id="grafico3")
                , 

                dbc.Col(
                    [
                        dcc.Graph(id="line-graph", figure=fig4),
                        ], id="grafico4"),
                ]),
         dbc.Row([     
                dbc.Col(
                    [               
                        dcc.Graph(id="line-graph", figure=fig5),
                        ], id="grafico5")
                , 

                dbc.Col(
                    [
                        dcc.Graph(id="line-graph", figure=fig6),
                        ], id="grafico6"),
                ]),
         dbc.Row([     
                dbc.Col(
                    [               
                        dcc.Graph(id="line-graph", figure=fig7),
                        ], id="grafico7")
                , 

                dbc.Row(
                    [
                        dcc.Graph(id="line-graph", figure=fig8),
                        ], id="grafico8"),
                ]),
 

            ],)



    #], 
#)
'''
# =====================================================================
# Interactivity
@app.callback(
    [
        Output("casos-recuperados-text", "children"),
        Output("em-acompanhamento-text", "children"),
        Output("casos-confirmados-text", "children"),
        Output("novos-casos-text", "children"),
        Output("obitos-text", "children"),
        Output("obitos-na-data-text", "children"),
    ], [Input("date-picker", "date"), Input("location-button", "children")]
)
def display_status(date, location):
    # print(location, date)
    if location == "BRASIL":
        df_data_on_date = df_brasil[df_brasil["data"] == date]
    else:
        df_data_on_date = df_states[(df_states["estado"] == location) & (df_states["data"] == date)]

    recuperados_novos = "-" if df_data_on_date["Recuperadosnovos"].isna().values[0] else f'{int(df_data_on_date["Recuperadosnovos"].values[0]):,}'.replace(",", ".") 
    acompanhamentos_novos = "-" if df_data_on_date["emAcompanhamentoNovos"].isna().values[0]  else f'{int(df_data_on_date["emAcompanhamentoNovos"].values[0]):,}'.replace(",", ".") 
    casos_acumulados = "-" if df_data_on_date["casosAcumulado"].isna().values[0]  else f'{int(df_data_on_date["casosAcumulado"].values[0]):,}'.replace(",", ".") 
    casos_novos = "-" if df_data_on_date["casosNovos"].isna().values[0]  else f'{int(df_data_on_date["casosNovos"].values[0]):,}'.replace(",", ".") 
    obitos_acumulado = "-" if df_data_on_date["obitosAcumulado"].isna().values[0]  else f'{int(df_data_on_date["obitosAcumulado"].values[0]):,}'.replace(",", ".") 
    obitos_novos = "-" if df_data_on_date["obitosNovos"].isna().values[0]  else f'{int(df_data_on_date["obitosNovos"].values[0]):,}'.replace(",", ".") 
    return (
            recuperados_novos, 
            acompanhamentos_novos, 
            casos_acumulados, 
            casos_novos, 
            obitos_acumulado, 
            obitos_novos,
            )


@app.callback(
        Output("line-graph", "figure"),
        [Input("location-dropdown", "value"), Input("location-button", "children")]
)
'''

if __name__ == "__main__":
    app.run_server(debug=False, port=8051)
