import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import os
import re
from scipy.optimize import curve_fit
import plotly.colors as colors
import plotly.express as px
from pathlib import Path

import functions


def save_plot(fig, filename: str, path='C:/Users/lborm/OneDrive/Desktop/Bachelorarbeit/Abbildungen', legende_x=0.9, legende_y=0.9):
    """
       Saves a Plotly figure as both an HTML file and an SVG image.

       The function removes the figure's title, as required for the thesis format,
       and saves the plot in HTML und SVG.
       The files are saved as `filename.html` and `filename.svg` in the given directory.

       Parameters:
       fig : plotly.graph_objs.Figure
           The Plotly figure object to be saved.
       filename : str
           The name (without extension) to use when saving the files.
       path: directory where the plots are saved, defaults to desktop/abbildungen
       legende_x: x-position to set the legend
       legende_y: y-postion of the legend

       Prints:
       Confirmation messages with the paths to the saved files.
       """
    fig.update_layout(title='')
    basepath = Path(path)

    html_path = basepath / (filename + '.html')
    svg_path = basepath / (filename + '.png')

    fig.update_layout(
        legend=dict(
            x=legende_x,
            y=legende_y, )
    )

    fig.write_html(html_path, include_plotlyjs='cdn', include_mathjax='cdn')
    pio.write_image(fig, svg_path, format='png', scale=1, width=1000, height=700)

    print(f"Plot saved as HTML to: {html_path}")
    print(f"Plot saved as SVG to: {svg_path}")


def linear_model(x, a):
    return a * x


def linear(x, a, b):
    return a * x + b


def plot_spectra(folder_path, title='UV-Vis Methylenblue', cut_off=500, show=True, max_nm=False, lambda_nm=664.0):
    fig = go.Figure()
    points_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path, sep=';', decimal=',', index_col=0, header=1)
            filtered_data = data[data.index > cut_off]  # cut out left peaks

            try:
                min_a = filtered_data.loc[800.0, 'A']
            except KeyError:
                min_a = filtered_data['A'].min()

            if max_nm:
                max_a = filtered_data['A'].max()
            else:
                max_a = filtered_data.loc[lambda_nm, 'A']

            points_data.append({
                'filename': filename,
                'absorption': max_a - min_a,
            })
            name = filename.split('.')[0]
            name = name.split('_')[1] if '_' in name else name.split('-')[1]
            fig.add_trace(go.Scatter(x=data.index, y=data['A'], mode='lines', name=f'{name} min'))

    fig.add_vline(
        x=lambda_nm,
        line=dict(color='red', dash='dash'),
        annotation_text=f'{lambda_nm} nm',
        annotation_position='top right'
    )
    fig.update_layout(title=title, xaxis_title='λ (nm)',
                      yaxis_title='Extinktion (-)')
    fig = functions.customize_plot(fig, y_min=0, y_max=2, round_lim=False, ticks='spectra')
    # if show:
    #     fig.show()
    points = pd.DataFrame(points_data)
    return points


def calibration(folder_path, concentration, cut_off=500, show=False, max_nm=False, lambda_nm=664.0):
    calibration_curve = plot_spectra(folder_path, f"{folder_path} Calibration", cut_off=cut_off, show=show, max_nm=max_nm, lambda_nm=lambda_nm)
    x = concentration
    y = calibration_curve['absorption'].values

    params, params_covariance = curve_fit(linear_model, x, y)
    slope = params[0]

    if show:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=12, symbol='circle-open', line=dict(width=2)), name='Messpunkt'))
        x_plot = np.linspace(0, 10, 100)
        y_plot = slope * x_plot  
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode='lines', line=dict(dash='dash'), name=f'E = {round(slope, 4)} · c'))  # ,name=f'E = {round(slope,4)} · c')   fr"$\mathrm{{E}} = {round(slope, 4)} \cdot \mathrm{{c}}$"
        fig.update_layout(title=f"{folder_path} Calibration Curve",
                          xaxis_title='c (mg/l)',
                          yaxis_title=f'E(λ={int(lambda_nm)} nm) (-)',
                          xaxis=dict(range=[0, 10]),
                          yaxis=dict(range=[0, 2]),
                          font=dict(size=20))

        fig = functions.customize_plot(fig, ticks='else', font_size=28)
        fig.show()
        save_plot(fig, f'calibration_{folder_path}', legende_x=0.65, legende_y=0.05)

    return slope


def zeitverlauf(experiment, slope, times = [0,5,10,15,20,30,40,60], cut_off=500, dilution=1, max_nm=False, font_size=20):
    df = plot_spectra(experiment, title=experiment, cut_off=cut_off, max_nm=max_nm)
    df['concentration'] = df['absorption'] / slope

    # extract the dilution factor from the filename eg dil2, dil4 etc
    for i, row in df.iterrows():
        filename = row['filename']
        match = re.search(r'dil(\d+)', filename)

        if match:
            dilution_factor = int(match.group(1))
            df.loc[i, 'concentration'] *= dilution_factor

    df['concentration'] = df['concentration'] * dilution

    # df['time'] = df['filename'].str.extract(r'_(-?\d+)').astype(float)
    time_after_underscore = df['filename'].str.extract(r'_(\-?\d+)')[0]
    time_before_min = df['filename'].str.extract(r'(\-?\d+)min')[0]
    df['time'] = time_after_underscore.fillna(time_before_min).astype(float)

    df = umsatz(df)

    df = df.sort_values(by='time')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'],
                             y=df['concentration'],
                             mode='lines+markers',
                             line=dict(color='blue'),
                             marker=dict(
                                size=12,
                                symbol='circle',
                                color='white',
                                line=dict(width=2, color='blue'),
                            ),
                        ))
    fig.update_layout(
        title='',
        xaxis_title='t (min)',
        yaxis_title='c (mg/l)',
        font=dict(size=20),
        ),
    fig = functions.customize_plot(fig, max(df['time']), max(df['concentration']), font_size=font_size)
    # fig.show()
    return fig, df


def umsatz(df):
    c0_series = df[df['time'] == 0]['concentration']
    c0 = c0_series.iloc[0]
    df['umsatz'] = (c0 - df['concentration'])/c0
    return df


def add_photolyse(fig, reactor='batch', mode='concentration'):
    if reactor == 'flow':
        folder = 'L58'

    elif reactor == 'batch':
        folder = 'L4?'
    conc = [1, 2, 5, 10]
    slope = calibration('MB_Kalibrierung', conc, show=False, max_nm=False)

    plot_allinone(fig, folder, slope, times=times, title='MB Konzentrationsvariationen', name=folder, color='dodgerblue', mode=mode)


def plot_allinone(fig, experiment, slope, title, times=[0], cut_off=500, name=None, color=None, dilution=1, max_nm=False, mode='concentration'):
    df = plot_spectra(experiment, title=experiment, show=False, cut_off=cut_off, max_nm=max_nm)

    if name is None:
        name = experiment

    df['concentration'] = df['absorption'] / slope

    for i, row in df.iterrows():
        filename = row['filename']
        match = re.search(r'dil(\d+)', filename)

        if match:
            dilution_factor = int(match.group(1))  # Extract the number after 'dil'
            df.loc[i, 'concentration'] *= dilution_factor
    df['concentration'] = df['concentration'] * dilution
    df = umsatz(df)

    if mode == 'concentration':
        y_column = 'concentration'
        y_label = 'c (mg/l)'
        ticks = 'time'
        round_lim = True
        y_max = max(df[y_column])
    elif mode == 'umsatz':
        y_column = 'umsatz'
        y_label = 'X (-)'
        ticks = 'umsatz'
        round_lim = False
        y_max = 1

    df['time'] = df['filename'].str.extract(r'_(-?\d+)').astype(float)  # the -? is for optional negative numbers

    fig.add_trace(go.Scatter(x=df['time'],
                             y=df[y_column],
                             mode='lines+markers',
                             marker=dict(
                                 size=12,
                                 symbol='circle',
                                 color='white',
                                 line=dict(width=2, color=color)
                             ),
                             name=name,
                             line=dict(color=color),
                             ))
    fig.update_layout(title=title, xaxis_title='t (min)',
                      yaxis_title=y_label,
                      )
    print(df)
    fig = functions.customize_plot(fig, max(df['time']), y_max, ticks=ticks, round_lim=round_lim)

    return df


def dunkelzeiten():
    conc = [0.5, 1, 2, 3, 5, 7, 10]
    slope = calibration('MB_Kalibrierung1206', conc, show=False)


def farbvariation():
    times = [0, 5, 10, 15, 20, 30, 40, 60]
    colors = ['MB', 'MO', 'EY', 'CV']

    fig = go.Figure()
    fig_umsatz = go.Figure()
    fig_int = go.Figure()

    for color in colors:
        if color == 'MB':
            cut_off = 500
            col = 'blue'
            lambda_nm = 664.0

        elif color == 'CV':
            cut_off = 400
            col = 'darkviolet'
            lambda_nm = 590.0

        elif color == 'MO':
            cut_off = 350
            col = 'orange'
            lambda_nm = 465.0

        else:
            col = 'lightcoral'
            cut_off = 400
            lambda_nm = 516.0

        folder_name = f"{color}_Kalibrierung"
        conc = [1, 2, 5, 10]
        slope = calibration(folder_name, conc, cut_off=cut_off, show=False, max_nm=False, lambda_nm=lambda_nm)

        experiment = f"{color}_Farbvariation"
        df = plot_allinone(fig, experiment, slope, times=times, title='Farbvariation', cut_off=cut_off, name=color, color=col, max_nm=True)
        plot_allinone(fig_umsatz, experiment, slope,
                      times=times, title='Farbvariation',
                      cut_off=cut_off, name=color, color=col, max_nm=True,
                      mode='umsatz')
        integrale_methode(df, fig_int, col, name=color)


    fig.show()
    fig_umsatz.show()
    fig_int.show()

    save_plot(fig, 'farbvariation', legende_x=0.8)
    save_plot(fig_umsatz, 'farbvariation_umsatz', legende_x=0.8, legende_y=0.1)
    save_plot(fig_int, 'farbvariation_int', legende_x=0, legende_y=1)
    return fig, fig_umsatz, fig_int


def kinetik():
    thermal_colors = px.colors.sequential.Blues

    conc = [0.5, 1, 2, 3, 5, 7, 10]
    slope = calibration('MB_Kalibrierung1206', conc, show=False)

    times = [0, 5, 10, 15, 20, 30, 40, 60]

    fig = go.Figure()
    fig_umsatz = go.Figure()
    fig_int = go.Figure()
    k_list = []
    konz_list = [5, 10, 20, 30, 40, 50]

    for i, folder in enumerate(os.listdir('MB_konzvariation')):
        folder_path = os.path.join('MB_konzvariation', folder)
        color = thermal_colors[(i+3) % len(thermal_colors)]
        name = f'{int(folder)} mg L⁻¹'
        df = plot_allinone(fig, folder_path, slope, times=times, title='MB Konzentrationsvariationen', name=name, color=color)
        plot_allinone(fig_umsatz, folder_path, slope, times=times, title='MB Konzentrationsvariationen', name=name, mode='umsatz',
                      color=color)

        k = integrale_methode(df, fig_int, color, name=name, x_max=60)

        k_list.append(k)

    # plot for 1/k over c
    fig_truek = go.Figure()
    k_list = 1 / np.array(k_list, dtype=float)
    fig_truek.add_trace(go.Scatter(
        x=konz_list,
        y=k_list,
        mode='markers',
        marker=dict(
            size=12,
            symbol='circle',
            color='white',
            line=dict(width=2, color='royalblue'),
        ),
        showlegend=False,
    ))
    fig_truek.update_layout(
        xaxis_title="c_MB(0) (ppm)",
        yaxis_title="1/k (min)",
        xaxis=dict(
            range=[0, 50]
        )
    )
    functions.customize_plot(fig_truek, ticks='konzvar')
    functions.customize_plot(fig_int, font_size=20)

    params, params_covariance = curve_fit(linear, konz_list, k_list)
    slope_0 = params[0]
    intercept = params[1]

    k_c = 1 / slope_0
    K = slope_0 / intercept

    print('K_LH: ', K,' kc: ', k_c)

    inv_Ce_fit = np.linspace(min(konz_list), max(konz_list), 100)
    y_plot = intercept + slope_0 * inv_Ce_fit

    fig_truek.add_trace(go.Scatter(
        x=inv_Ce_fit, y=y_plot,
        mode='lines',
        line=dict(dash='dash'),
        name=f'1/k = {round(slope_0, 4)} · c + {round(intercept, 4)}'
    ))
    fig_truek.show()

    functions.customize_plot(fig, ticks='konzvar')
    fig.show()
    fig_int.show()
    
    save_plot(fig, 'MB_konzvariation', legende_x=0.75, legende_y=0.9)
    save_plot(fig_umsatz, 'MB_konzvariation_umsatz', legende_x=0.75, legende_y=0.05)
    save_plot(fig_int, 'MB_konzvariation_kinetik', legende_x=0.8, legende_y=0)
    save_plot(fig_truek, 'MB_konzvariation_wahres_k', legende_x=0.45, legende_y=0.05)


def integrale_methode(data, fig_int, color, name, x_max=30):
    df = data.copy()
    df['time'] = df['filename'].str.extract(r'_(\d+)').astype(float)

    ln0 = np.log(df.loc[0, 'concentration'])
    df['lnc'] = ln0 - np.log(df['concentration'])

    # weil man kinetik nur bis 95% umsatz bestimmt
    # 5% der initialkonzentration sind 95% umsatz - wir wollen Werte über dieser Konznetration bzw unter diesem Umsatz behalten
    df = df[df['concentration'] > 0.05 * df['concentration'].iloc[0]]

    params, params_covariance = curve_fit(linear_model, df['time'], df['lnc'])

    k = params[0]
    x_plot = np.linspace(0.1, 60, 100)
    y_plot = k * x_plot

    fig_int.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        name=f'y = {round(k, 4)} · x',
        mode='lines',
        line=dict(dash='dash', color=color)
    ))

    fig_int.add_trace(go.Scatter(x=df['time'],
                                 y=df['lnc'],
                                 name=name,
                                 mode='markers',
                                 marker=dict(
                                     size=12,
                                     symbol='circle',
                                     color='white',
                                     line=dict(width=2, color=color)
                                 ),
                                 ))

    y_max = max(df['lnc']) + 1
    y_max = 3
    fig_int = functions.customize_plot(fig_int, x_max=x_max, y_max=y_max, round_lim=False)
    fig_int.update_layout(
        xaxis_title='t (min)',
        yaxis_title='ln[c₀/c(t)] (-)',
       )

    return k


def plot_reihe(reihe, slope, x_legend=0.8, y_legend=0.9, max_nm=False):
    fig = go.Figure()
    fig_umsatz = go.Figure()
    fig_int = go.Figure()
    thermal_colors = px.colors.qualitative.Plotly

    for i, folder in enumerate(os.listdir(reihe)):
        folder_path = os.path.join(reihe, folder)
        print(folder)
        color = thermal_colors[(i) % len(thermal_colors)]
        df = plot_allinone(fig, folder_path, slope, title=reihe, name=folder, color=color, max_nm=max_nm)
        plot_allinone(fig_umsatz, folder_path, slope,
                      title='Umsatz',
                      name=folder, color=color, max_nm=max_nm,
                      mode='umsatz')
        integrale_methode(df, fig_int, color, name=folder)

    fig.update_layout(
        legend=dict(
            x=x_legend,
            y=y_legend, ))
    fig.show()
    fig_umsatz.show()
    fig_int.show()
    return fig, fig_umsatz, fig_int


def bestrahlungsflachen():
    fig = go.Figure()
    fig_umsatz = go.Figure()
    fig_int = go.Figure()
    fig_k = go.Figure()

    reihe = 'Bestrahlunsgflächen_wiederholung'
    thermal_colors = px.colors.qualitative.Plotly
    k_list = [0]
    area_list = [0, 1, 4, 9, 16]

    for i, folder in enumerate(os.listdir(reihe)):
        name = folder.lstrip('0')
        folder_path = os.path.join(reihe, folder)
        color = thermal_colors[i % len(thermal_colors)]
        df = plot_allinone(fig, folder_path, slope, times=times, title=reihe, name=name, color=color)
        plot_allinone(fig_umsatz, folder_path, slope, times=times, title=reihe, name=name, color=color, mode='umsatz')

        k = integrale_methode(df, fig_int, color, name=name)
        k_list.append(k)

    fig_k.add_trace(go.Scatter(x=area_list,
                               y=k_list,
                               mode='markers',
                               marker=dict(
                                   size=12,
                                   symbol='circle',
                                   color='white',
                                   line=dict(width=2, color='royalblue')
                               ),
                               ))

    params, params_covariance = curve_fit(linear, area_list[:-1], k_list[:-1])  # der letzte punkt ist nicht mehr linear
    x_plot = np.linspace(0.1, 10, 100)
    y_plot = params[0] * x_plot + params[1]

    fig_k = functions.customize_plot(fig_k, x_max=20, x_min=0, y_min=0, y_max=0.18, round_lim=False, ticks='k')
    fig_k.update_layout(
        xaxis_title='Fläche (cm²)',
        yaxis_title='k (min⁻¹)')

    fig.show()
    fig_umsatz.show()
    fig_int.show()
    fig_k.show()

    reihe = 'bestrahlungsflächen'
    save_plot(fig_k, f'k_{reihe}')
    save_plot(fig_int, f'int_{reihe}', legende_x=0, legende_y=1)
    save_plot(fig, f'zeitverlauf_{reihe}', legende_x=0.8, legende_y=1)
    save_plot(fig_umsatz, f'umsatz_{reihe}', legende_x=0.8, legende_y=0.8)


def langmuir( reihe = "AA_15.11", cat_mass=[0.0403, 0.0391, 0.04, 0.0398, 0.04, 0.04], volume=0.04 ):
    fig = go.Figure()

    # volume in L, reaction volume is 40ml
    # massen is g catalyst

    adsorption_data = []

    zeiten = [0, 20]

    df_list = []
    Ce_list = []
    Qe_list = []


    df_list.append(plot_allinone(fig, os.path.join(reihe, '5'), slope, times=zeiten, title=reihe, name='5 mg/l', max_nm=True, dilution=1))
    df_list.append(plot_allinone(fig, os.path.join(reihe, '10'), slope, times=zeiten, title=reihe, name='10 mg/l',max_nm=True, dilution=1.13))
    df_list.append(
        plot_allinone(fig, os.path.join(reihe, '20'), slope, times=zeiten, title=reihe, name='20 mg/l', max_nm=True, dilution=1.18)) # dilution=3.5)
    df_list.append(
        plot_allinone(fig, os.path.join(reihe, '30'), slope, times=zeiten, title=reihe, name='30 mg/l',max_nm=True, dilution=1.1)) # dilution=4
    df_list.append(
        plot_allinone(fig, os.path.join(reihe, '40'), slope, times=zeiten, title=reihe, name='40 mg/l',max_nm=True, dilution=1.2)) #dilution=4.3
    df_list.append(
        plot_allinone(fig, os.path.join(reihe, '50'), slope, times=zeiten, title=reihe, name='50 mg/l',max_nm=True, dilution=0.8)) #dilution=5

    for df, w in zip(df_list, cat_mass):
        c0 = df.loc[0, 'concentration']  # Initial concentration (mg/l)
        ce = df.iloc[-1]['concentration']

        # Qe is amount adsorbed per unit mass of catalyst
        qe = (c0 - ce) * volume / w
        Ce_list.append(ce)
        Qe_list.append(qe)

        adsorption_data.append({'Ce (mg/l)': ce, 'Qe (mg/g)': qe})

    # Plot Qe over Ce
    fig_q = go.Figure()
    fig_iso = go.Figure()

    inv_y = np.array(Ce_list, dtype=float) / np.array(Qe_list, dtype=float)
    fig_iso.add_trace(go.Scatter(x=Ce_list, y=Qe_list, mode='lines+markers',
                                 marker=dict(
                                     size=12,
                                     symbol='circle',
                                     color='white',
                                     line=dict(width=2, color='royalblue')
                                 ),
                                 line=dict(width=2, color='royalblue'),
                                 showlegend=False,
                                 ))

    fig_q.add_trace(go.Scatter(x=Ce_list, y=inv_y, mode='markers',
                               marker=dict(
                                   size=12,
                                   symbol='circle',
                                   color='white',
                                   line=dict(width=2, color='royalblue')
                               ),
                               showlegend=False,
                               ))

    params, params_covariance = curve_fit(linear, Ce_list, inv_y)
    slope_0 = params[0]
    intercept = params[1]

    q_max = 1 / slope_0
    K_ads = slope_0 / intercept

    x_fit = np.linspace(min(Ce_list), max(Ce_list), 100)
    y_fit = intercept + slope_0 * x_fit

    fig_q.add_trace(go.Scatter(
        x=x_fit, y=y_fit,
        mode='lines',
        line=dict(dash='dash'),
        name=f'y = {slope_0:.4f} · x + {intercept:.4f}'
    ))

    print(f"q_max = {q_max:.4f} mg/g")
    print(f"K = {K_ads:.4f} l/mg")

    fig_q.update_layout(
        title='Qe vs Ce',
        xaxis_title='Ce (mg/l)',
        yaxis_title='Ce/Qe (g/l)',
    )

    fig_iso.update_layout(
        title='Qe vs Ce',
        xaxis_title='Ce (mg/l)',
        yaxis_title='Qe (mg/g)',
    )

    functions.customize_plot(fig_iso, x_max=50, x_min=0, y_min=0, y_max=3.5, round_lim=False)
    functions.customize_plot(fig_q, x_max=50, x_min=0, y_min=0, y_max=14, round_lim=False)

    fig_q.show()
    fig_iso.show()

    save_plot(fig_q, 'langmuir_linearisiert', legende_x=0.5, legende_y=0.05)
    save_plot(fig_iso, 'langmuir_Ce_Qe')

    return adsorption_data


def abstandsvariation(reihe, d_list_opt=None):
    fig = go.Figure()
    fig_umsatz = go.Figure()
    fig_int = go.Figure()
    fig_k = go.Figure()

    thermal_colors = px.colors.qualitative.Plotly
    k_list = []
    d_list = []

    for i, folder in enumerate(os.listdir(reihe)):
        folder_path = os.path.join(reihe, folder)
        color = thermal_colors[i % len(thermal_colors)]
        df = plot_allinone(fig, folder_path, slope, times=times, title=reihe, name=folder, color=color)
        plot_allinone(fig_umsatz, folder_path, slope, times=times, title=reihe, name=folder, color=color, mode='umsatz')

        number_match = re.search(r'\d+', folder)
        if number_match:
            d_value = float(number_match.group())
            d_list.append(d_value)
        else:
            d_list = d_list_opt
            print('opt d list verwendet')

        k = integrale_methode(df, fig_int, color, folder)
        k_list.append(k)

    k_list = np.log(k_list)
    d_list = np.log(d_list)

    df_dk = pd.DataFrame({'d_list': d_list, 'k_list': k_list})
    df_dk = df_dk.sort_values(by='d_list')

    d_list_sorted = df_dk['d_list'].values
    k_list_sorted = df_dk['k_list'].values

    params, params_covariance = curve_fit(linear, d_list_sorted, k_list_sorted)
    x_plot = np.linspace(min(d_list), max(d_list), 100)
    y_plot = params[0] * x_plot + params[1]

    params_1, params_covariance = curve_fit(linear, d_list_sorted[:-1], k_list_sorted[:-1])
    x_plot_1 = np.linspace(min(d_list), max(d_list), 100)
    y_plot_1 = params_1[0] * x_plot + params_1[1]

    fig_k.add_trace(go.Scatter(x=df_dk['d_list'],
                               y=df_dk['k_list'],
                               mode='markers',
                               showlegend=False,
                               marker=dict(
                                   size=12,
                                   symbol='circle',
                                   color='white',
                                   line=dict(width=2, color='royalblue')
                               ),
                               ))
    fig_k.add_trace(go.Scatter(
        x=x_plot,
        y=y_plot,
        mode='lines',
        line=dict(dash='dash'),
        name=f'y1 = {round(params[0], 5)} · x + {round(params[1], 4)}'
    ))

    fig_k.add_trace(go.Scatter(
        x=x_plot_1,
        y=y_plot_1,
        mode='lines',
        line=dict(dash='dash', color='green'),
        name=f'y2 = {round(params_1[0], 4)} · x + {round(params_1[1], 4)}'
    ))

    fig_k = functions.customize_plot(
        fig_k,
    )
    fig_k.update_layout(
        xaxis_title='log(d)',
        yaxis_title='log(k)',
        xaxis=dict(
            range=[2, 2.5] # batch
            # range=[0.5, 2.5]
        ),
        yaxis=dict(
            range=[-2.3, -1.7] # batch
            # range=[-6.1, -5.4] # flow
        )
    )

    fig_int.update_layout(
        xaxis=dict(
            range=[0, 20]
            # range=[0, 60] # flow
        ),
        # yaxis=dict(
        #     range=[0, 0.3] # flow
        # )
    )

    fig.show()
    fig_umsatz.show()
    fig_int.show()
    fig_k.show()

    # batch
    save_plot(fig_k, f'k_{reihe}', legende_x=0.45)
    save_plot(fig_int, f'int_{reihe}', legende_x=0.05)
    save_plot(fig, f'zeitverlauf_{reihe}', legende_x=0.8)
    save_plot(fig_umsatz, f'umsatz_{reihe}', legende_x=0.8, legende_y=0.1)

    # save_plot(fig_k, f'k_{reihe}', legende_x=0.05)
    # save_plot(fig_int, f'int_{reihe}', legende_x=0, legende_y=1)
    # save_plot(fig, f'zeitverlauf_{reihe}', legende_x=0.8, legende_y=0.05)
    # save_plot(fig_umsatz, f'umsatz_{reihe}', legende_x=0.8)


def literaturwerte():
    farben = ['MB', 'MO', 'EY', 'CV']
    zno_lit = [0.0334, 0.0205, np.nan, 0.009]
    tio_lit = [0.024, 0.012, np.nan, 0.020]
    zno_exp = [0.0939, 0.0304, 0.0447, 0.051]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=farben, y=zno_lit, mode='markers', marker=dict(size=12, symbol='square-open', line=dict(width=2)),
                             name='ZnO Literatur'))
    fig.add_trace(go.Scatter(x=farben, y=tio_lit, mode='markers', marker=dict(size=12, symbol='diamond-open', line=dict(width=2)),
                             name='TiO2 Literatur'))
    fig.add_trace(go.Scatter(x=farben, y=zno_exp, mode='markers', marker=dict(size=12, symbol='circle-open', line=dict(width=2)),
                             name='ZnO experimentell'))

    fig.update_layout(title=f"literaturwerte",
                      xaxis_title='Farbstoff',
                      yaxis_title='k (min⁻¹)',
                      yaxis=dict(range=[0, 0.12]),)
    fig = functions.customize_plot(fig, round_lim=False, ticks='k_werte')
    # fig.show()
    save_plot(fig, 'k_literaturwerte', legende_x=0.6)

# literaturwerte()

if __name__ == "__main__":
    conc = [1, 2, 5, 10]
    slope = calibration('data/MB_Kalibrierung', conc, show=False)
    times = [0, 5, 10, 15, 20, 30, 40, 60]

# bestrahlungsflachen()

# fig, fig_umsatz = plot_reihe('Dunkelzeiten')
# save_plot(fig, 'Dunkelzeiten', legende_x=0.8)
# save_plot(fig_umsatz, 'Dunkelzeiten_umsatz', legende_x=0.8, legende_y=0.05)

# fig = plot_reihe('Abstandsreihe_batch')
# abstandsvariation('Abstandsreihe_batch', [10, 12, 8])
# abstandsvariation('Abstandsreihe_flow', [10, 12, 6, 8])

# plot_reihe('MB_konzvariation')

# fig, fig_umsatz, fig_int = plot_reihe('Katalysator_variation')
# save_plot(fig, 'Katalysator_variation', legende_x=0.65, legende_y=0.8)
# save_plot(fig_int, 'Katalysator_variation_int', legende_x=0, legende_y=1)
# save_plot(fig_umsatz, 'Katalysator_variation_umsatz', legende_x=0.65, legende_y=0.05)


# farbvariation()


# kinetik()

# fig, df = zeitverlauf('gläser', slope, times)
# fig, df = zeitverlauf('L50', slope, times)
# save_plot(fig, 'uv_led')
#
# fig, df = zeitverlauf('L52', slope, times, dilution=0.52)
# save_plot(fig, 'immokat_batch')

# zeitverlauf('L10', slope, times) # photolyse batch
# zeitverlauf('L55', slope, times)
# fig, df = zeitverlauf('L54', slope, times)
# save_plot(fig, 'l54', legende_x=0.8)

# fig_umsatz = go.Figure()
# plot_allinone(fig_umsatz, 'l54', slope,
#                       times=times, title='Umsatz',
#                       name='', color='royalblue',
#                       mode='umsatz')
# save_plot(fig_umsatz, 'l54_umsatz', legende_x=0.8, legende_y=0.1)

# fig_int = go.Figure()
# df_lf = zeitverlauf('L54', slope, times, max_nm=False)
# integrale_methode(df_lf, fig_int, color='royalblue', name='', x_max=40)
# save_plot(fig_int, 'l54_kinetik', legende_x=0.7, legende_y=0.1)

# adsorption_data = langmuir()

# fig, fig_umsatz, fig_int = plot_reihe('immobilisierterKat_Batch', max_nm=True)
# save_plot(fig, 'immobilisierterKat_Batch', legende_x=0.8)
# save_plot(fig_umsatz, 'immobilisierterKat_Batch_umsatz', legende_x=0.8, legende_y=0.3)

# fig, fig_umsatz, fig_int = plot_reihe('Filter-Fritte-Aufbau')
#
# fig.add_vline(
#         x=5,
#         line=dict(color='red', dash='dash'),
#         # annotation_text=f'{lambda_nm} nm',
#         annotation_text='5 min',
#         annotation_position='top right'
#     )
#
# save_plot(fig, 'filterfritte', legende_x=0.75)
# save_plot(fig_umsatz, 'filterfritte_umsatz', legende_x=0.75, legende_y=0.1)

# # wasserstoffperoxid
# fig, fig_umsatz = plot_reihe('12092024', max_nm=True)
# save_plot(fig, 'H2O2 Reihe batch', legende_x=0.7)
# save_plot(fig_umsatz, 'H2O2_batch_umsatz', legende_x=0.7, legende_y=0.1)
#
# fig, fig_umsatz = plot_reihe('AA_15.11')
# fig, fig_umsatz = plot_reihe('H2O2_flow')
# save_plot(fig, 'H2O2 Reihe flow', legende_x=0.75, legende_y=0.6)
# save_plot(fig_umsatz, 'H2O2_flow_umsatz', legende_x=0.75, legende_y=0.7)


# fig, fig_umsatz, fig_int = plot_reihe('neue Konzepte')
# save_plot(fig, 'schale', legende_x=0.8, legende_y=0.4)
# save_plot(fig_umsatz, 'schale_umsatz', legende_x=0.8, legende_y=0.4)

# fig, fig_umsatz, fig_int = plot_reihe('flow_versuche')
# save_plot(fig, 'flow_versuche', legende_x=0.75, legende_y=0)
# save_plot(fig_umsatz, 'flow_versuche_umsatz', legende_x=0.75, legende_y=1)
