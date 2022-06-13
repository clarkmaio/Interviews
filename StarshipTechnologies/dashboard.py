import streamlit as sl
import plotly.express as px
import pandas as pd
import numpy as np


def curve_builder(salary_input: float, growth_rate: float, inflation_rate: float):
    curve = pd.DataFrame(np.nan, index=range(2022, 2041), columns=['linear', 'exponential', 'logarithmic'])
    curve['linear'] = salary_input * (1 + (curve.index - 2022) * (growth_rate - inflation_rate) / 100)
    curve['exponential'] = salary_input * np.exp((curve.index - 2022) * (growth_rate - inflation_rate) / 100)
    curve['logarithmic'] = salary_input * (1 + np.log(1 + (curve.index - 2022) * (growth_rate - inflation_rate) / 100))

    curve.index.name = 'Year'
    curve = curve.astype(int)
    return curve




sl.set_page_config(layout="wide")


# -------------- SIDEBAR --------------
sl.sidebar.markdown('# Settings')

numbeo_index = sl.sidebar.slider(label='Numbeo index converter', min_value=0., max_value=1., value=0.42)

SALARY_EXPECTATION_CHF = 130952.38095238096
SALARY_EXPECTATION_EUR = SALARY_EXPECTATION_CHF * numbeo_index

sl.sidebar.markdown('#')
salary_input = sl.sidebar.number_input(label='Salary [€]', value=SALARY_EXPECTATION_EUR, min_value=0., max_value=1000000000.)

sl.sidebar.markdown('#')
inflation_rate = sl.sidebar.slider(label='Inflation rare [%]', min_value=0., max_value=5., value=3.)

sl.sidebar.markdown('#')
growth_rate = sl.sidebar.slider(label='Salary growth rate [%]', min_value=0., max_value=10., value=5.)
growth_curve = sl.sidebar.multiselect(label='Growth curve type', options=['linear', 'exponential', 'logarithmic'], default=['linear', 'exponential', 'logarithmic'])


# -------------- MAIN PAGE ------------
salary_curve = curve_builder(salary_input = salary_input, growth_rate = growth_rate, inflation_rate = inflation_rate)

sl.title("Andrea's salary explorer")

sl.markdown('Hi, this is a just a page to play with numbers.')
sl.markdown('**Numbeo index converter** lets you convert the value of CHF in Lugano into € in Tallinn by multiplying CHF by the index.')
sl.markdown('#')

c1, _, c2 = sl.columns([3, 1, 2])

with c1:
    # plot
    fig = px.line(data_frame=salary_curve, y=growth_curve, log_y=True)
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(title="<b>Salary projections</b>", xaxis_title="Year", yaxis_title="Salary [€]", hovermode="x", title_x = .5)
    sl.plotly_chart(fig)

with c2:
    # Display
    delta = 100 * round(1 - SALARY_EXPECTATION_EUR / salary_input, 2)
    sl.metric('Salary', value=f'{round(salary_input)}€', delta=f'{delta}%')



