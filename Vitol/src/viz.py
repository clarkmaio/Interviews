

import polars as pl
import altair as alt
from typing import List
import matplotlib.pyplot as plt


def plot_weekonweek_diff(df1: pl.DataFrame, df2: pl.DataFrame, label1: str = 'df1', label2: str = 'df2'):
    """
    Create a plot composed by 3 subplots:
    Two plots represent timeseries and week on week change.
    One scatter plot that compare timeseries of the two countries
    """

    # Prepare aux dataframe
    df1 = df1.rename({
        'driving': 'Driving', 
        'walking': 'Walking', 
        'transit': 'Transit', 
        'total': 'Total'
    })

    df2 = df2.rename({
        'driving': 'Driving', 
        'walking': 'Walking', 
        'transit': 'Transit', 
        'total': 'Total'
    })

    df1_diff = df1.select(
        pl.col('date'),
        pl.col('Driving').diff(),
        pl.col('Walking').diff(),
        pl.col('Transit').diff(),
        pl.col('Total').diff()
    ).drop_nulls()

    df2_diff = df2.select(
        pl.col('date'),
        pl.col('Driving').diff(),
        pl.col('Walking').diff(),
        pl.col('Transit').diff(),
        pl.col('Total').diff()
    ).drop_nulls()

    df_scatter = df1.unpivot(index='date', value_name='value_1').join(
        df2.unpivot(index='date', value_name='value_2'), 
        on=['date', 'variable']
    ).rename({'variable': 'key'})



    legend_selection = alt.selection_point(
        name='legend_select',
        fields=['key'],

        bind='legend',
        value='Total'
    )

    x_selection = alt.selection_interval(encodings=['x'], empty='all')

    line = alt.Chart(df1).transform_fold(
        ['Total', 'Driving','Walking', 'Transit'],
    ).mark_line().encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('value:Q', axis=alt.Axis(title='Mobility')),
        color=alt.Color('key:N', legend=alt.Legend(title='Mobility type', orient='top', direction='horizontal')),
        opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.2)),
        strokeWidth=alt.condition(legend_selection, alt.value(3), alt.value(1)),
    ).add_params(
        legend_selection,
        x_selection
    )



    bar = alt.Chart(df1_diff).transform_fold(
        ['Total', 'Driving', 'Walking', 'Transit'],
        as_=['key', 'value']
    ).transform_filter(
        legend_selection
        #alt.FieldEqualPredicate(field='key', equal='Total')
    ).mark_bar(cornerRadius=50).encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('value:Q', axis=alt.Axis(title='Week-on-week change'),
        scale=alt.Scale(domain=[-150, 300])
        ),
        color=alt.condition(alt.datum.value > 0, alt.value('green'), alt.value('red')),
        opacity=alt.value(.6),
    ).add_params(
        legend_selection
    )



    line2 = alt.Chart(df2).transform_fold(
        ['Total', 'Driving','Walking', 'Transit'],
    ).mark_line().encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('value:Q', axis=alt.Axis(title='Mobility')),
        color=alt.Color('key:N', legend=alt.Legend(title='Mobility type', orient='top', direction='horizontal')),
        opacity=alt.condition(legend_selection, alt.value(1), alt.value(0.2)),
        strokeWidth=alt.condition(legend_selection, alt.value(3), alt.value(1)),
    ).add_params(
        legend_selection,
        x_selection
    )


    bar2 = alt.Chart(df2_diff).transform_fold(
        ['Total', 'Driving', 'Walking', 'Transit'],
        as_=['key', 'value']
    ).transform_filter(
        legend_selection
        #alt.FieldEqualPredicate(field='key', equal='Total')
    ).mark_bar(cornerRadius=50).encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('value:Q', axis=alt.Axis(title='Week-on-week change'),
        scale=alt.Scale(domain=[-150, 300])
        ),
        color=alt.condition(alt.datum.value > 0, alt.value('green'), alt.value('red')),
        opacity=alt.value(.6),
    ).add_params(
        legend_selection
    )


    scatter = alt.Chart(df_scatter).transform_filter(
        legend_selection
    ).mark_point().encode(
        x=alt.X('value_1:Q', title=label1),
        y=alt.Y('value_2:Q', title=label2),
        opacity=alt.condition(x_selection, alt.value(1), alt.value(0.2)),
        color=alt.value('black')
    )
    scatter_reg = scatter.transform_filter(
        x_selection
    ).transform_regression(
        'value_1', 'value_2'
    ).mark_line(color='black').encode(
        opacity=alt.value(1)
    )

    xy_line = alt.Chart(df_scatter).transform_filter(
        legend_selection
    ).mark_line(color='black').encode(
        x=alt.X('value_1:Q', title=label1),
        y=alt.Y('value_1:Q', title=label2),
        opacity=alt.value(0.2),
    )

    plot1 = (line + bar).resolve_scale(y='independent').properties(width=600, height=200, title=label1)
    plot2 = (line2 + bar2).resolve_scale(y='independent').properties(width=600, height=200, title=label2)
    scatter_corr = (scatter + scatter_reg + xy_line).properties(width=300, height = 450, title='Correlation scatter')
    return (plot1 & plot2) | (scatter_corr)
    



def plot_rolling_correlation(df, country_tag: List[str], max_correlation_window: int = 30):
    '''
    Generate interactive plot to visualize timeseries and window rolling correlation.

    df: must contain columns date, driving_*, walking_*, transit_*, total_* for each country
    '''


    # Build correlation timeseries
    corr_df = []
    WINDOW_RANGE = range(1, max_correlation_window + 1)
    for w in WINDOW_RANGE:
        corr_tmp = df.select(
            pl.col('date'),
            pl.rolling_corr(
                pl.col(f'total_{country_tag[0]}'),
                pl.col(f'total_{country_tag[1]}'),
                window_size=w,
            ).alias('total_correlation'),

            pl.rolling_corr(
                pl.col(f'walking_{country_tag[0]}'),
                pl.col(f'walking_{country_tag[1]}'),
                window_size=w,
            ).alias('walking_correlation'),

            pl.rolling_corr(
                pl.col(f'driving_{country_tag[0]}'),
                pl.col(f'driving_{country_tag[1]}'),
                window_size=w,
            ).alias('driving_correlation'),

            pl.rolling_corr(
                pl.col(f'transit_{country_tag[0]}'),
                pl.col(f'transit_{country_tag[1]}'),
                window_size=w,
            ).alias('transit_correlation'),

            pl.lit(w).alias('window')
        )
        corr_df.append(corr_tmp)
    corr_df = pl.concat(corr_df, how='vertical').sort(['window','date'])


    LEGEND_SELECTION = alt.selection_point(
        fields=['mobility_type'],
        bind='legend',
        name='Select Mobility Type'
    )

    WINDOW_SLIDER = alt.binding_range(
        min=min(WINDOW_RANGE), max=max(WINDOW_RANGE), step=1, name='Correlation window Size'
    )

    WINDOW_SELECTION = alt.selection_point(
        fields=['window'],
        bind=WINDOW_SLIDER,
        name='Select Window Size', value=max_correlation_window//2
    )

    MOBILITY_DROPDOWN= alt.binding_select(options=['Total', 'Walking', 'Driving', 'Transit'], name='Mobility Type')
    MOBILITY_SELECT = alt.selection_point(fields=['mobility_type'], bind=MOBILITY_DROPDOWN, value='Total')



    # ----------------------------------------- PLOTS
    df_unpivot = df.unpivot(
        index=['date']
    ).with_columns(
        mobility_type=pl.col('variable').str.split('_').list.get(0).str.to_titlecase(),
        country=pl.col('variable').str.split('_').list.get(1).str.to_titlecase()
    )

    plot_line = alt.Chart(df_unpivot).transform_filter(
        MOBILITY_SELECT
    ).mark_line().encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('value:Q', title='Mobility index'),
        color=alt.Color('country:N', legend=alt.Legend(title='Country'))
    ).properties(width=800, height=300)


    plot_correlation = alt.Chart(corr_df).transform_filter(
        WINDOW_SELECTION
    ).transform_calculate(
        Total='datum.total_correlation',
        Walking='datum.walking_correlation',
        Driving='datum.driving_correlation',
        Transit='datum.transit_correlation'
    ).transform_fold(
        fold=['Total', 'Walking', 'Driving', 'Transit'],
        as_=['mobility_type', 'correlation']
    ).transform_filter(
        MOBILITY_SELECT
    ).mark_line().encode(
        x=alt.X('date:T', title=None),
        y=alt.Y('correlation:Q', title='Correlation'),
        #color=alt.Color('mobility_type:N', legend=alt.Legend(title='Mobility Type')),
        tooltip=['date:T', 'mobility_type:N', 'correlation:Q']
    ).properties(
        width=800,
        height=150
    ).add_params(
        WINDOW_SELECTION,
        LEGEND_SELECTION,
        MOBILITY_SELECT
    )



    layout = (plot_line & plot_correlation).properties(
            resolve = alt.Resolve(scale=alt.LegendResolveMap(color=alt.ResolveMode('independent'))),
            title=alt.Title(f'{country_tag[0].upper()} vs {country_tag[1].upper()} rolling correlation', fontSize=20, anchor='middle' )
    )
    return layout





def plot_stack(df: pl.DataFrame, x: str, y: List[str], 
               total_y: str = None, 
               labels: List[str] = None,
               colors: List[str] = None):

    plt.stackplot(
        df[x],
        *[df[c] for c in y],
        labels=labels,
        colors=colors)

    if total_y:
        plt.plot(df[x], df[total_y], label='Total', color='black')

    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

