import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import math
import matplotlib.pyplot as plt


def customize_plot_t(fig, x_max, y_max, x_min=0, y_min=0, round_lim=True, log_axis=False):

    if round_lim:
        x_max = math.ceil(x_max / 10) * 10
        y_max = math.ceil(y_max / 10) * 10

        x_min = math.floor(x_min / 10) * 10
        y_min = math.floor(y_min / 10) * 10

    fig.update_layout(
        font=dict(size=18),
        xaxis=dict(
            range=[x_min, x_max],  # axis limits
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,  # Mirror axis line
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
            minor=dict(showgrid=True, ticks="outside", tick0=1, dtick=1)  # fuer die kleinen striche auf der achse
        ),
        yaxis=dict(
            range=[y_min, y_max],
            showline=True,
            linewidth=2,
            linecolor='black',
            mirror=True,
            ticks='outside',
            tickwidth=2,
            tickcolor='black',
            minor=dict(showgrid=True, ticks="outside", tick0=1, dtick=1)
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background color
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            x=0.9,  # position (0 to 1, where 0 is left and 1 is right)
            y=0.9,  # position (0 to 1, where 0 is bottom and 1 is top)
            bordercolor='grey',
            borderwidth=2,
            )
    )
    if log_axis:
        print('log')
        # make this axis log appropiate
    return fig


def customize_plot(fig, x_max=None, y_max=None, x_min=None, y_min=None, round_lim=True, log_axis=False, ticks='time', font_size=28):

    if ticks == 'time':
        x_minor = dict(showgrid=True, ticks="outside", tick0=5, dtick=5)
        y_minor = dict(showgrid=True, ticks="outside", tick0=0.5, dtick=0.5)
    elif ticks == 'konzvar':
        x_minor = dict(showgrid=True, ticks="outside", tick0=5, dtick=5)
        y_minor = dict(showgrid=True, ticks="outside", tick0=1, dtick=1)
    elif ticks == 'spectra':
        x_minor = dict(showgrid=True, ticks="outside", tick0=10, dtick=10)
        y_minor = dict(showgrid=True, ticks="outside", tick0=0.1, dtick=0.1)
    elif ticks == 'umsatz':
        x_minor = dict(showgrid=True, ticks="outside", tick0=5, dtick=5)
        y_minor = dict(showgrid=True, ticks="outside", tick0=0.05, dtick=0.05)
    elif ticks == 'k_werte':
        x_minor = dict(showgrid=True, ticks="outside", tick0=1, dtick=1)
        y_minor = dict(showgrid=True, ticks="outside", tick0=0.01, dtick=0.01)
    else:
        x_minor = dict(showgrid=True, ticks="outside", tick0=1, dtick=1)
        y_minor = dict(showgrid=True, ticks="outside", tick0=0.1, dtick=0.1)

    xaxis_dict = dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=2,
        tickcolor='black',
        minor=x_minor
    )
    yaxis_dict = dict(
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        ticks='outside',
        tickwidth=2,
        tickcolor='black',
        minor=y_minor
    )

    # set axis limits
    if x_max is not None:
        if round_lim:
            x_max = math.ceil(x_max / 10) * 10
            x_min = math.floor(x_min / 10) * 10
        xaxis_dict['range'] = [x_min, x_max]
    if y_max is not None:
        if round_lim:
            y_max = math.ceil(y_max / 10) * 10
            y_min = math.floor(y_min / 10) * 10
        yaxis_dict['range'] = [y_min, y_max]

    fig.update_layout(
        font=dict(size=font_size),
        xaxis=xaxis_dict,
        yaxis=yaxis_dict,
        # plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            x=0.9,  # Position of the legend
            y=0.9,
        )
    )
    return fig
