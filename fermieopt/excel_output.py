import logging
import os
from pathlib import Path
from typing import List, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import load_workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.utils.dataframe import dataframe_to_rows

from fermieopt.config import BusLabels, EffectLabels, EnergyPriceLabels, OtherLabels, SinkLabels
from fermieopt.flixPostprocessingXL import FlixPostXL

logger = logging.getLogger('flixOpt')


### Utility functions
def resample_data(
    data_frame: Union[pd.DataFrame, np.ndarray],
    target_years: List[int],
    resampling_by: Literal['YE', 'd', 'h'],
    resampling_method: Literal['sum', 'mean', 'min', 'max'],
    initial_sampling_rate: str = 'h',
) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_frame : Union[pd.DataFrame, np.ndarray]
        DataFrame or array containing data. Number of rows must match the initial sampling rate (safety check):
        8760 ("h") (default) or 365 ("d") per year.
    target_years : List[int]
        Target years for the new index of the DataFrame
    resampling_by : str
        "h" for hourly resampling
        "d" for daily resampling
        "YE" for yearly resampling
    resampling_method : str
        "mean" for mean value
        "sum" for sum value
        "max" for max value
        "min" for min value
    initial_sampling_rate : str
        "h" for hourly data (8760 values per year)
        "d" for daily data (365 values per year)

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(data_frame)
    df.index = range(len(df))  # reset index

    if len(df) / 8760 == len(target_years) and initial_sampling_rate == 'h':
        length_per_year = 8760
    elif len(df) / 365 == len(target_years) and initial_sampling_rate == 'd':
        length_per_year = 365
    elif (len(df) - 1) / 8760 == len(target_years) and initial_sampling_rate == 'h':
        length_per_year = 8760
        df = df.iloc[:-1, :]
    elif (len(df) - 1) / 365 == len(target_years) and initial_sampling_rate == 'd':
        length_per_year = 365
        df = df.iloc[:-1, :]
    else:
        raise ValueError(
            "length of dataframe and initial_sampling_rate must match: 8760 rows/year ('H') or 365 rows/year 'D'."
        )

    if not isinstance(target_years, list):
        target_years = [target_years]

    # create new TS for resampling, without the 29. February (filtering leap years)
    for i, year in enumerate(target_years):
        dt = pd.date_range(start=f'1/1/{year}', end=f'01/01/{year + 1}', freq=initial_sampling_rate)[:-1]
        dt = dt[~((dt.month == 2) & (dt.day == 29))]  # Remove leap year days
        df.loc[i * length_per_year : (i + 1) * length_per_year - 1, 'Timestamp'] = dt
    df = df.set_index('Timestamp')

    if resampling_method == 'sum':
        df = df.resample(resampling_by).sum()
    elif resampling_method == 'mean':
        df = df.resample(resampling_by).mean()
    elif resampling_method == 'min':
        df = df.resample(resampling_by).min()
    elif resampling_method == 'max':
        df = df.resample(resampling_by).max()
    else:
        raise ValueError('Invalid resampling method')

    # Drop all rows that aren't in the years specified in target_years
    lst = [row for row in df.index if row.year not in target_years]
    df = df.drop(index=lst)

    df = df.loc[~((df.index.month == 2) & (df.index.day == 29))]  # Remove leap year days again

    if resampling_by == 'YE':
        df = df.set_index(df.index.year)  # setting the index to the plain year. No datetime anymore

    return df


def rs_in_two_steps(
    data_frame: Union[pd.DataFrame, np.ndarray],
    target_years: List[int],
    resampling_by: Literal['d', 'YE'],
    initial_sampling_rate: str = 'h',
) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_frame : Union[pd.DataFrame, np.ndarray]
        DataFrame or array containing data. Number of rows must match the initial sampling rate (safety check):
        8760 ("h") (default) or 365 ("d") per year.
    target_years : List[int]
        Years for resampling
    resampling_by : str
        "d" for daily resampling
        "YE" for yearly resampling
    initial_sampling_rate : str
        "h" for hourly data
        "d" for daily data
    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with new columns:
        ["Tagesmittel", "Minimum (Stunde)", "Maximum (Stunde)"]
        or new Columns:
        ["Jahresmittel", "Minimum (Tagesmittel)", "Maximum (Tagesmittel)"],
        depending on chosen "resampling_by"
    """

    # Determine base resampling method and new columns based on resampling_by
    if resampling_by == 'd':
        rs_method_base = 'h'
        new_columns = ['Tagesmittel', 'Minimum (Stunde)', 'Maximum (Stunde)']
    elif resampling_by == 'YE':
        rs_method_base = 'd'
        new_columns = ['Jahresmittel', 'Minimum (Tagesmittel)', 'Maximum (Tagesmittel)']
    else:
        raise ValueError("Invalid value for resampling_by. Use 'D' for daily or 'Y' for yearly.")

    # Base resampling
    df_resampled_base = resample_data(data_frame, target_years, rs_method_base, 'mean', initial_sampling_rate)

    # Resample for min, max, and mean
    min_y = resample_data(df_resampled_base, target_years, resampling_by, 'min', rs_method_base)
    max_y = resample_data(df_resampled_base, target_years, resampling_by, 'max', rs_method_base)
    mean_y = resample_data(df_resampled_base, target_years, resampling_by, 'mean', rs_method_base)

    # Concatenate results
    df_result = pd.concat([mean_y, min_y, max_y], axis=1)
    df_result.columns = new_columns

    return df_result


def reorder_columns(df: pd.DataFrame, not_sorted_columns: List[str] = None):
    """
    Order a DataFrame by a custom function, excluding specified columns from sorting, and setting them as the first columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    not_sorted_columns : List[str], optional
        Columns to exclude from sorting and set as the first columns, by default None.

    Returns
    -------
    pd.DataFrame
        DataFrame with the desired column order.
    """
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    sorted_columns = sorted(df.columns, key=lambda x: x.lower())
    sorted_df = df.reindex(columns=sorted_columns)

    # Select the remaining columns excluding the first two
    if not_sorted_columns is None:
        other_columns = [col for col in sorted_df.columns]
        # Create a new DataFrame with the desired column order
        new_order_df = sorted_df[other_columns]
    else:
        other_columns = [col for col in sorted_df.columns if col not in not_sorted_columns]

        # Create a new DataFrame with the desired column order
        new_order_df = pd.concat([sorted_df[not_sorted_columns], sorted_df[other_columns]], axis=1)

    return new_order_df


def df_to_excel_w_chart(
    df: pd.DataFrame,
    filepath: str,
    title: str,
    ylabel: str,
    xlabel: str,
    style: Literal['bar', 'line'] = 'bar',
    bar_style: Literal['stacked', 'clustered'] = 'stacked',
):
    """
    Write DataFrame to an Excel file with a stacked bar chart.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be written.
    filepath : str
        The path to the Excel file. If the file doesn't exist, a new one will be created.
    title : str
        The title of the sheet and chart.
    ylabel : str
        The label for the y-axis of the chart.
    xlabel : str
        The label for the x-axis of the chart.

    Returns
    -------
    None

    Notes
    -----
    This function writes the provided DataFrame to an Excel file and adds a stacked bar chart to a new sheet in the workbook.
    If the sheet with the given title already exists, it is removed before adding the new sheet.
    The stacked bar chart is created based on the DataFrame structure, with columns as categories and rows as data points.
    The chart is positioned at cell "D4" in the sheet.

    """
    filepath = Path(filepath)
    try:
        wb = load_workbook(filepath)
    except Exception:
        template_path = Path(__file__).parent / 'resources' / 'Template_blanco.xlsx'

        wb = load_workbook(template_path)

    if len(title) > 30:
        logger.warning(f'Sheetname "{title}" in file "{filepath}" was shortened in order to work as a Excel sheet name.')
        title = title[:30]
    # Check if the sheet already exists
    if title in wb.sheetnames:
        sheet = wb[title]
        wb.remove(sheet)

    # Add the sheet to the workbook
    sheet = wb.create_sheet(title)

    # Remove the index and save it as a column
    df = df.reset_index()
    # Write the data starting from the second row
    for r in dataframe_to_rows(df, index=False, header=True):
        sheet.append(r)

    # Create the Data and References
    data = Reference(sheet, min_col=2, min_row=1, max_col=df.shape[1], max_row=df.shape[0] + 1)
    labels = Reference(sheet, min_col=1, min_row=2, max_row=df.shape[0] + 1)

    # Create a stacked bar chart
    if style == 'bar':
        chart = BarChart()
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(labels)
        # Stacked bar plot
        chart.type = 'col'
        chart.grouping = bar_style
        if bar_style == 'stacked':
            chart.overlap = 100
            chart.gapWidth = 0  # Adjust the gap between bars (e.g., set gapWidth to 0%)
    elif style == 'line':
        chart = LineChart()
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(labels)
        # Stacked bar plot
        chart.type = 'line'

    # General Chart stuff
    chart.title = title
    chart.y_axis.title = ylabel
    chart.x_axis.title = xlabel
    chart.width = 30
    chart.height = 15

    # Add the chart to the sheet
    sheet.add_chart(chart, 'D4')  # Adjust the position as needed

    # Save the workbook
    wb.save(filepath)


class ExcelEvaluation:
    """
    This class is thightly coulpled with 2 excel templates. Originally designed for a specific use case and predefined system,
    class now accepts labels for components, that will be acessed in the visualization.
    If the used flixResults deffer from the original structure, these values can be adjusted
    """

    def __init__(
        self,
        results: FlixPostXL,
        effect_costs: str = EffectLabels.COSTS,
        effect_funding: str = EffectLabels.FUNDING,
        effect_co2_fw: str = EffectLabels.CO2_HEAT,
        bus_heating: str = BusLabels.HEAT,
        demand_heat: str = SinkLabels.HEAT_DEMAND,
        demand_heat_losses=SinkLabels.HEAT_LOSSES,
        group_label_heat_demand: str = 'Wärmelast',
        bus_electricity_out: str = BusLabels.ELECTRICITY_OUT,
        price_electricity: str = EnergyPriceLabels.ELECTRICITY,
        price_helper_elements: str = OtherLabels.ENERGY_PRICES,
    ):
        self.results = results

        self.effect_costs = effect_costs
        self.effect_funding = effect_funding
        self.effect_co2_fw = effect_co2_fw
        self.bus_heating = bus_heating
        self.demand_heat = demand_heat
        self.demand_heat_losses = demand_heat_losses
        self.group_label_heat_demand = group_label_heat_demand
        self.bus_electricity_out = bus_electricity_out
        self.price_electricity = price_electricity
        self.price_helper_elements = price_helper_elements

    def run_excel_graphics_years(self, short_version=False, custom_output_file_path: str = 'default'):
        """
        Generate detailed annual comparison plots and save them to individual Excel workbooks for each year.

        Parameters:
        - calc (FlixPostXL): Solved calculation of type FlixPostXL.
        - short_version (bool): If True, generate a shortened version of the plots. Default is False.
        - custom_output_file_path (str): A custom output folder path for the Excel workbooks. default: Same as calc.

        Returns:
        None

        The function generates detailed annual comparison plots based on the provided calculation results (FlixPostXL)
        and saves them to individual Excel workbooks. The workbooks are created using a template specified in the calculation.

        Plots and corresponding data are organized into different sheets within each workbook for the specified years:
        - "Wärmeerzeugung": Fernwärme generation data.
        - "Installierte Leistung": Installed capacity data.
        - "Wärmevollkosten": Heat full costs data.
        - "Wärmekosten Variabel": Variable heat costs data.
        - "Emissionen": Emissions data.
        - "Energieträger": Energy carriers data.
        - "Stromerzeugung": Electricity generation data.
        - "Speicherkapazität": Storage capacity data.
        - "Speicherfüllstand D": Daily storage fill level data.
        - "SpeicherFlows D": Daily storage flows data.
        - "WärmeErz-Last-D": Duration curve for heat generation and electricity prices, sorted by heat demand (Daily mean values).
        - "WärmeErz-Strom-D": Duration curves for heat generation and electricity prices, sorted by electricity prices (Daily mean values).

        If short_version is False (default), additional sheets are generated:
        - "WärmeErz-Last": Duration curve for heat generation and electricity prices, sorted by heat demand (Hourly values).
        - "WärmeErz-Strom": Duration curves for heat generation and electricity prices, sorted by electricity prices (Hourly values).
        - "Wärmeerzeugung_Februar": Hourly data for heat generation in February.
        - "Wärmeerzeugung_Juli": Hourly data for heat generation in July.
        - "WärmeErz-Last-DL-H": Annual load duration curves for heat generation. (Individually sorted for every generator)
        - "Speicher Summen": Hourly storage fill level data (Allocated over all storages).
        - "Speicherfüllstand H": Hourly storage fill level data for each individual storage.

        The Excel workbooks are saved in the specified output folder with filenames like
        "Jahr_{year}-{calc.infos['calculation']['name']}.xlsx".

        Example:
        ```
        calc = FlixPostXL(...)  # Create or obtain FlixPostXL instance
        run_excel_graphics_years(calc)  # Save the detailed workbooks in the default location
        run_excel_graphics_years(
            calc, short_version=True
        )  # Save shortened version of the workbooks in the default location
        run_excel_graphics_years(
            calc, custom_output_file_path='path/to/save/folder'
        )  # Save the detailed workbooks in a custom location
        ```

        """
        if custom_output_file_path == 'default':
            output_file_path = self.results.folder
        else:
            output_file_path = custom_output_file_path

        logger.info('Annual Plots to Excel...')

        # computation for the whole calculation

        df_fernwaerme_erz_nach_techn_d = self._get_fernwaerme_erz(resamply_by='d', rs_method='mean')  # Wärmeerzeugung

        df_installierte_leistung_y = self._get_installierte_leistung_pro_jahr(grouped=True)

        df_waermekosten_variable_costs_d = self._get_waermekosten_operation(resamply_by='d')

        df_emissions_d = self._get_emissions(resamply_by='d', rs_method='sum')

        df_eingesetzte_energietraeger_d = self._get_eingesetzte_energietraeger(resamply_by='d', rs_method='mean')

        df_stromerzeugung_d = self._get_stromerzeugung(resamply_by='d')

        df_speicher_kapazitaet_d = self._get_speicher_kapazitaet(
            resamply_by='d', grouped=True, actual_storage_capacity=True
        )

        df_speicher_fuellstand_d = self._get_speicher_fuellstand('d', 'mean', allocated=False)

        df_speicher_flows_d = self._get_speicher_flows('d', 'mean', allocated=False)

        df_fernwaerme_erz_nach_techn_h = self._get_fernwaerme_erz(resamply_by='h', rs_method='mean')

        df_speicher_fuellstand_h = self._get_speicher_fuellstand('h', 'mean', allocated=False)
        df_speicher_fuellstand_h_alloc = self._get_speicher_fuellstand('h', 'mean', allocated=True)

        logger.info('......computation of data for short version finished')
        if not short_version:
            # Erzeugung ungrouped
            df_fernwaerme_erz_h = self._get_fernwaerme_erz_individual()

        # TODO: weitere Grafiken

        logger.info('......computation of data finished')

        templ_path_excel_year = Path(__file__).parent / 'resources' / 'Template_Evaluation_Year.xlsx'

        for year in self.results.years:
            wb = load_workbook(templ_path_excel_year)
            filename = f'{self.results.name}__Jahr_{year}.xlsx'
            path_excel_year = os.path.join(output_file_path, filename)
            wb.save(path_excel_year)

            with pd.ExcelWriter(path_excel_year, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                # Wärmeerzeugung nach Technologie
                df = df_fernwaerme_erz_nach_techn_d[df_fernwaerme_erz_nach_techn_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Wärmeerzeugung')

                # Installierte Leistung nach Technologie
                df = df_installierte_leistung_y[df_installierte_leistung_y.index == year]
                df.to_excel(writer, index=True, sheet_name='Installierte Leistung')

                # Wärmekosten Betrieb
                df = df_waermekosten_variable_costs_d[df_waermekosten_variable_costs_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Wärmekosten Variabel')

                # Emissionen
                df = df_emissions_d[df_emissions_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Emissionen')

                # Energieträger
                df = df_eingesetzte_energietraeger_d[df_eingesetzte_energietraeger_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Energieträger')

                # Stromerzeugung
                df = df_stromerzeugung_d[df_stromerzeugung_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Stromerzeugung')

                # Speicherkapazität allokiert
                df = df_speicher_kapazitaet_d[df_speicher_kapazitaet_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Speicherkapazität')

                # Speicherfüllstand nicht allokiert (Tageswerte)
                df = df_speicher_fuellstand_d[df_speicher_fuellstand_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Speicherfüllstand D')

                # Speicherflows nicht allokiert (Tageswerte)
                df = df_speicher_flows_d[df_speicher_flows_d.index.year == year]
                df.to_excel(writer, index=True, sheet_name='SpeicherFlows D')

                # Wärmeerzeugung als Jahresdauerlinien (Stundenwerte)
                df = df_fernwaerme_erz_nach_techn_h[df_fernwaerme_erz_nach_techn_h.index.year == year]
                df.reset_index(drop=True).to_excel(writer, index=True, sheet_name='Wärmeerzeugung-H')

                # Wärmeerzeugung im Februar und Juli (Stundenwerte)
                df = df_fernwaerme_erz_nach_techn_h[df_fernwaerme_erz_nach_techn_h.index.year == year]
                df.loc[df.index.month == 2].to_excel(writer, index=True, sheet_name='Wärmeerzeugung_Februar')
                df.loc[df.index.month == 7].to_excel(writer, index=True, sheet_name='Wärmeerzeugung_Juli')

                # Jahresdauerlinien der einzelnen Wärmeerzeuger (Stundenwerte)
                df = df_fernwaerme_erz_nach_techn_h[df_fernwaerme_erz_nach_techn_h.index.year == year]
                df = pd.DataFrame(-np.sort(-df.values, axis=0), columns=df.columns)
                df.to_excel(writer, index=True, sheet_name='Wärmeerzeugung-Sorted')

                # Speicherfüllstand (Stundenwerte) allokiert
                df = df_speicher_fuellstand_h_alloc[df_speicher_fuellstand_h_alloc.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Speicher Summen')

                # Speicherfüllstand (Stundenwerte) nicht allokiert
                df = df_speicher_fuellstand_h[df_speicher_fuellstand_h.index.year == year]
                df.to_excel(writer, index=True, sheet_name='Speicherfüllstand H')

                logger.info(f'......Year-{year} finished (short version)')
                if not short_version:
                    df = df_fernwaerme_erz_h[df_fernwaerme_erz_h.index.year == year]
                    df.reset_index(drop=True).to_excel(writer, index=True, sheet_name='Wärmeerzeugung-Einzeln')
            logger.info(f'...Year-{year} finished')

            # TODO: weitere Grafiken

        logger.info('...Annual Plots to Excel finished')

    def run_excel_graphics_main(self, custom_output_file_path: str = 'default'):
        """
        Generate annual comparison plots and save them to an Excel workbook.

        Parameters:
        - calc (FlixPostXL): Solved calculation of type FlixPostXL.
        - custom_output_file_path (str): A custom output file path (full path) for the Excel workbook. Default: Same as calc.

        Returns:
        None

        The function generates various annual comparison plots based on the provided calculation results (FlixPostXL)
        and saves them to an Excel workbook. The workbook is created using a template specified in the calculation.

        Plots and corresponding data are organized into different sheets within the workbook:
        - "Waermelast und Verluste": Fernwärme load and losses data.
        - "Kostenübersicht": Costs overview data.
        - "Wärmeerzeugung": Fernwärme generation data.
        - "Installierte Leistung": Installed capacity data.
        - "Wärmevollkosten": Heat full costs data.
        - "Wärmekosten Variabel": Variable heat costs data.
        - "Emissionen": Emissions data.
        - "Energieträger": Energy carriers data.
        - "Stromerzeugung": Electricity generation data.
        - "Speicherkapazität": Storage capacity data.
        - "Speicher Summen": Summed storage fill level data.

        The Excel workbook is saved in the same folder as the calculation results with the filename
        "Jahresübersicht-{calc.infos['calculation']['name']}.xlsx". If a custom_output_file_path is provided,
        the workbook is saved at that location.

        Example:
        ```
        calc = FlixPostXL(...)  # Create or obtain FlixPostXL instance
        run_excel_graphics_main(calc)  # Save the workbook in the default location
        run_excel_graphics_main(
            calc, custom_output_file_path='path/to/save/file.xlsx'
        )  # Save the workbook in a custom location
        ```

        """
        logger.info('Overview Plots to Excel...')

        if custom_output_file_path == 'default':
            output_file_path = self.results.folder
        else:
            output_file_path = custom_output_file_path

        templ_path_excel_main = Path(__file__).parent / 'resources' / 'Template_Evaluation_Overview.xlsx'

        wb = load_workbook(templ_path_excel_main)
        filename = f'{self.results.name}__Jahresübersicht.xlsx'
        path_excel_main = os.path.join(output_file_path, filename)
        wb.save(path_excel_main)

        with pd.ExcelWriter(path_excel_main, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df = self._get_fernwaerme_last_and_loss('YE', 'sum')
            df.to_excel(writer, index=True, sheet_name='Waermelast und Verluste')

            df = self._get_costs_and_funding_per_year()
            df.to_excel(writer, index=True, sheet_name='Kostenübersicht')

            df = self._get_fernwaerme_erz('YE', 'sum') / 1000
            df.to_excel(writer, index=True, sheet_name='Wärmeerzeugung')

            df = self._get_installierte_leistung_pro_jahr(grouped=True)
            df.to_excel(writer, index=True, sheet_name='Installierte Leistung')

            df = self._get_waermekosten_per_period(with_fix_costs=True)
            df.to_excel(writer, index=True, sheet_name='Wärmevollkosten')

            df = self._get_waermekosten_operation(resamply_by='YE')
            df.to_excel(writer, index=True, sheet_name='Wärmekosten Variabel')

            df = self._get_emissions(resamply_by='YE', rs_method='sum')
            df.to_excel(writer, index=True, sheet_name='Emissionen')

            df = self._get_eingesetzte_energietraeger(resamply_by='YE', rs_method='sum') / 1000
            df.to_excel(writer, index=True, sheet_name='Energieträger')

            df = self._get_stromerzeugung(resamply_by='YE')
            df.to_excel(writer, index=True, sheet_name='Stromerzeugung')

            df_speicher_kapazitaet_y = self._get_speicher_kapazitaet(
                resamply_by='YE', grouped=True, actual_storage_capacity=False
            )
            df_speicher_kapazitaet_y.to_excel(writer, index=True, sheet_name='Speicherkapazität')

            df_speicher_fuellstand_sum_h = self._get_speicher_fuellstand('h', 'mean', allocated=True).reset_index(
                drop=True
            )
            df_speicher_fuellstand_sum_h.to_excel(writer, index=True, sheet_name='Speicher Summen')

        logger.info('...Overview Plots to Excel finished')

    def _get_costs_and_funding_per_year(self):
        funding_var = self.results.get_effect_results(self.effect_funding, origin='operation', as_time_series=True)
        costs_var = self.results.get_effect_results(self.effect_costs, origin='operation', as_time_series=True)

        df = pd.DataFrame(
            data={'Variable Kosten (abzgl. Förderung)': costs_var, 'Förderung Betrieb': -1 * funding_var},
            index=self.results.time,
        )
        data = resample_data(df, self.results.years, 'YE', 'sum').to_dict()

        funding_fix = self.results.get_effect_results(self.effect_funding, origin='invest_per_period')
        costs_fix = self.results.get_effect_results(self.effect_costs, origin='invest_per_period')

        data['Förderung Invest'] = {year: value for year, value in zip(self.results.years, -1 * funding_fix, strict=False)}
        data['Fixkosten (abzgl. Förderung)'] = {
            year: value for year, value in zip(self.results.years, costs_fix, strict=False)
        }

        return pd.DataFrame(data)

    def _get_fernwaerme_erz(self, resamply_by, rs_method):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling

            if "d", Strompreis and Wärmelast are added to the DataFrame in first and second column
            if "YE", Wärmelast and sorages are not included
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value

        Returns
        -------
        pd.DataFrame
        """
        df_fernwaerme = self.results.to_data_frame(self.bus_heating, 'inout', grouped=True)
        if resamply_by == 'YE':
            df_fernwaerme.drop(
                columns=[self.group_label_heat_demand], inplace=True
            )  # ohne Wärmelast, ohne Speicher
        else:
            df_fernwaerme[self.group_label_heat_demand] = (
                -1 * df_fernwaerme[self.group_label_heat_demand]
            )  # reinverting

            try:
                df_fernwaerme = pd.concat([
                    df_fernwaerme, self.results.get_energy_prices(self.price_helper_elements)[self.price_electricity]
                ], axis=1)
            except KeyError:
                logger.warning(f'Electricity price "{self.price_electricity}" was not found and therefore can not be plotted')

        df_fernwaerme_erz_nach_techn = resample_data(df_fernwaerme, self.results.years, resamply_by, rs_method)

        return self._merge_into_dispatch_structure(df_fernwaerme_erz_nach_techn)

    def _get_fernwaerme_erz_individual(self, threshold: Union[float, Literal['auto']] = 'auto') -> pd.DataFrame:
        """
        Parameters
        ----------
        threshhold_for_dropping : None, float
            decide wether not used parts (all 0 values) should be dropped

        Returns
        -------
        pd.DataFrame
        """

        def remove_near_zero(df: pd.DataFrame, tolerance: float) -> pd.DataFrame:
            columns_to_remove = []
            for col in df.columns:
                if df[col].abs().max() <= tolerance:
                    columns_to_remove.append(col)
            return df.drop(columns=columns_to_remove)

        df_fernwaerme = self.results.to_data_frame(self.bus_heating, 'inout', grouped=False)
        if threshold and threshold == 'auto':
            if threshold == 'auto':
                tolerance = 0.01 / 100 * df_fernwaerme.values.max()  # 0.01% as trheshold
                df_fernwaerme = remove_near_zero(df_fernwaerme, tolerance)
            else:
                df_fernwaerme = remove_near_zero(df_fernwaerme, threshold)

        df_fernwaerme = resample_data(df_fernwaerme, self.results.years, 'h', 'mean')

        return df_fernwaerme[sorted(df_fernwaerme.columns, key=lambda col: col.lower())]

    def _get_installierte_leistung_pro_jahr(self, grouped: bool):
        """
        Parameters
        ----------
        calc1 : flix_results
            Calculation Object

        Returns
        -------
        pd.DataFrame
        """
        df_invest = pd.DataFrame(self.results.sizes_per_period_connected_to_bus('Fernwärme', True), index=self.results.years)
        if grouped:
            df_invest = self.results.group_df_by_mapping(df_invest)
        df_invest = reorder_columns(df_invest)

        if df_invest.empty:
            return df_invest
        return self._merge_into_dispatch_structure(df_invest)

    def _get_waermekosten_operation(self, resamply_by: Literal['YE', 'd', 'h']):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value

        Returns
        -------
        pd.DataFrame
        """
        heat = self.results.to_data_frame(self.demand_heat, 'in')

        costs_total = pd.Series(
            self.results.get_effect_results(effect_label=self.effect_costs, origin='operation', as_time_series=True),
            index=self.results.time,
        )

        # Unterschiedung zwischen Resampling
        if resamply_by == 'd':
            rs_method_base = 'h'
            new_columns = ['Tagesmittel', 'Minimum (Stunde)', 'Maximum (Stunde)']
        elif resamply_by == 'YE':
            rs_method_base = 'd'
            new_columns = ['Jahresmittel', 'Minimum (Tagesmittel)', 'Maximum (Tagesmittel)']
        else:
            raise ValueError(f"not implemented for resamply_by parameter: '{resamply_by}'")

        mean_costs_increment = resample_data(costs_total, self.results.years, rs_method_base, 'mean').iloc[:, 0]
        mean_heat_increment = resample_data(heat, self.results.years, rs_method_base, 'mean').iloc[:, 0]
        mean_costs_per_heat_increment = pd.DataFrame(
            mean_costs_increment / mean_heat_increment, columns=['EURvarPerMWh']
        )

        yearly_min = resample_data(mean_costs_per_heat_increment, self.results.years, resamply_by, 'min', rs_method_base)
        yearly_max = resample_data(mean_costs_per_heat_increment, self.results.years, resamply_by, 'max', rs_method_base)
        increment_sum_of_costs_total = resample_data(
            mean_costs_increment, self.results.years, resamply_by, 'sum', rs_method_base
        ).iloc[:, 0]
        increment_sum_of_heat_total = resample_data(
            mean_heat_increment, self.results.years, resamply_by, 'sum', rs_method_base
        ).iloc[:, 0]
        yearly_mean = increment_sum_of_costs_total / increment_sum_of_heat_total

        df = pd.concat([yearly_mean, yearly_min, yearly_max], axis=1)
        df.columns = new_columns

        return df

    def _get_waermekosten_per_period(self, with_fix_costs: bool):
        heat = pd.DataFrame(self.results.to_data_frame(self.demand_heat, 'in'))
        costs_operation = self.results.get_effect_results(effect_label=self.effect_costs, origin='operation', as_time_series=True)

        heat = resample_data(heat, self.results.years, 'YE', 'sum')
        costs_operation = resample_data(costs_operation, self.results.years, 'YE', 'sum')
        if with_fix_costs:
            costs_invest = pd.DataFrame(
                self.results.get_effect_results(effect_label=self.effect_costs, origin='invest_per_period'),
                index=self.results.years,
            )
            costs = costs_operation + costs_invest
        else:
            costs = costs_operation

        return pd.DataFrame(
            {'Wärmevollkosten [€/MWh]': costs.values.flatten() / heat.values.flatten()}, index=self.results.years
        )

    def _get_emissions(self, resamply_by, rs_method):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value

        Returns
        -------
        pd.DataFrame
        """
        heat = self.results.to_data_frame(self.demand_heat, 'in')

        co2 = pd.DataFrame(
            self.results.get_effect_results(effect_label=self.effect_co2_fw, origin='operation', as_time_series=True),
            index=self.results.time,
        )

        co2_per_increment = resample_data(co2, self.results.years, resamply_by, rs_method).iloc[:, 0]
        heat_per_increment = resample_data(heat, self.results.years, resamply_by, rs_method).iloc[:, 0]
        co2_per_heat = co2_per_increment / heat_per_increment * 1000  # from t/MWh to kg/MWh
        df_emissions = pd.concat([co2_per_heat.round(1), co2_per_increment, heat_per_increment], axis=1)
        df_emissions.columns = ['kgCO2PerMWh', 'tCO2absolut', 'MWhabsolut']

        return df_emissions

    def _get_eingesetzte_energietraeger(self, resamply_by, rs_method):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value

        Returns
        -------
        pd.DataFrame
        """
        df_sources = self.results.get_sources_and_sinks(sources=True, sinks=False, source_and_sinks=False)
        df = resample_data(df_sources, self.results.years, resamply_by, rs_method)
        df = reorder_columns(df)

        return df

    def _get_stromerzeugung(self, resamply_by):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value

        Returns
        -------
        resampled DataFrame with new columns:
            if resamply_by = "d": ["Tagesmittel", "Minimum (Stunde)", "Maximum (Stunde)"]
            if resamply_by = "YE": ["Jahresmittel", "Minimum (Tagesmittel)", "Maximum (Tagesmittel)"],
        """
        df_stromerzeugung = self.results.to_data_frame(self.bus_electricity_out, 'out', invert_output=False)
        df = rs_in_two_steps(df_stromerzeugung, self.results.years, resamply_by, 'h')

        return df

    def _get_speicher_kapazitaet(self, resamply_by, grouped, actual_storage_capacity: bool):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling

        Returns
        -------
        resampled DataFrame with capacity of all Storages
        """
        invest_results_speicher = self.results.get_availlability()
        invest_results_speicher = invest_results_speicher.filter(items=self.results.storages)
        if invest_results_speicher.empty:
            invest_results_speicher = pd.DataFrame(np.zeros(len(self.results.time)), index=self.results.time)
            invest_results_speicher.rename(columns={invest_results_speicher.columns[0]: 'Speicher'}, inplace=True)
        elif grouped:
            invest_results_speicher = self.results.group_df_by_mapping(invest_results_speicher)

        df = resample_data(invest_results_speicher, self.results.years, resamply_by, 'max')

        return df

    def _get_speicher_fuellstand(self, resamply_by, rs_method, allocated):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value
        allocated : boolean
            True: sum of all storages in column one and netto flow of all Storages in column 2
            False: charge state of storage separately, no flow values

        Returns
        -------
        resampled DataFrame with total charge_state of all Storages
        """
        df_speicher_charge_state = pd.DataFrame(index=self.results.time)
        df_speicher_netto_flow = pd.DataFrame(index=self.results.time)

        for comp in self.results.storages:
            df_speicher_charge_state[comp] = self.results.component_results[comp].all_results['charge_state'][
                :-1
            ]  # without the last step
            df_speicher_netto_flow[comp] = self.results.component_results[comp].all_results['netto_discharge']

        if allocated:
            charge_state_sum = df_speicher_charge_state.sum(axis=1)
            netto_flow_sum = df_speicher_netto_flow.sum(axis=1) * -1

            df = pd.concat([charge_state_sum, netto_flow_sum], axis=1)
            df.columns = ['Gesamtspeicherstand', 'Nettospeicherflow']
            df = resample_data(df, self.results.years, resamply_by, rs_method)
        else:
            df = resample_data(df_speicher_charge_state, self.results.years, resamply_by, rs_method)

        return df

    def _get_fernwaerme_last_and_loss(self, resamply_by, rs_method):
        data = {
            'Wärmebedarf [MWh]': self.results.component_results[self.demand_heat].inputs[0].all_results['flow_rate'],
            'Wärmeverlust [MWh]': self.results.component_results[self.demand_heat_losses]
            .inputs[0]
            .all_results['flow_rate'],
        }

        df_summed = resample_data(pd.DataFrame(data), self.results.years, resamply_by, rs_method)
        df_verluste_summed = (
            (df_summed['Wärmeverlust [MWh]'] / df_summed.sum(axis=1) * 100).rename('Verlust[%]').round(2)
        )

        return pd.concat([df_summed, df_verluste_summed], axis=1)

    def _get_speicher_flows(self, resamply_by, rs_method, allocated):
        """
        Parameters
        ----------
        resamply_by : string
            "h" for hourly resampling
            "d" for daily resampling
            "YE" for yearly resampling
        rs_method : string
            "mean" for mean value
            "sum" for sum value
            "max" for max value
            "min" for min value
        allocated : boolean
            True: sum of all storages in column one and netto flow of all Storages in column 2
            False: charge state of storage separately, no flow values

        Returns
        -------
        resampled DataFrame with total charge_state of all Storages
        """
        df_speicher_netto_flow = pd.DataFrame(index=self.results.time)

        for comp in self.results.storages:
            df_speicher_netto_flow[comp] = self.results.component_results[comp].all_results['netto_discharge'] * -1

        if allocated:
            df = df_speicher_netto_flow.sum(axis=1)
            df = resample_data(df, self.results.years, resamply_by, rs_method)
            df.columns = ['Nettospeicherflow']
        else:
            df = resample_data(df_speicher_netto_flow, self.results.years, resamply_by, rs_method)

        return df

    def _merge_into_dispatch_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Brings a dataframe into a predefined structure for dispatch evaluation.
        Has space for 9 undefined columns
        """
        # Step 1: Create an empty DataFrame with specific column names
        fixed_columns_1 = [
            'TAB',
            'Geothermie',
            'Abwärme',
            'WP',
            'WP_2',
            'EHK',
            'KWK_Gas',
            'KWK_H2',
            'Kessel_Gas',
            'Kessel_H2',
            'Speicher_S',
            'Speicher_L',
            'Kühler',
        ]  # First 11 fixed columns
        undefined_columns = ['U1', 'U2', 'U3', 'U4', 'U5', 'U6', 'U7', 'U8', 'U9']  # 8 undefined placeholders
        fixed_columns_2 = ['others', self.group_label_heat_demand, self.price_electricity]  # Last 2 fixed columns

        # Combine all parts into the final column structure
        all_columns = fixed_columns_1 + undefined_columns + fixed_columns_2

        # Step 2: Create the target DataFrame with this structure, initially filled with None
        df_target = pd.DataFrame(columns=all_columns, index=df.index)

        # String formattin to prevent unintended behaviour
        df.columns = df.columns.str.replace('ae', 'ä').str.replace('oe', 'ö').str.replace('ue', 'ü').str.strip()
        df.columns = [col[0].upper() + col[1:] for col in df.columns]

        # Merge logic
        # Directly assign matched columns
        for col in df.columns.intersection(df_target.columns):
            df_target[col] = df[col]

        # Handle unmatched columns by placing them into the undefined placeholders
        unmatched_columns = df.columns.difference(df_target.columns)
        unmatched_columns = sorted(unmatched_columns, key=lambda x: x.lower())  # sorting alphabetically
        for i, col in enumerate(unmatched_columns):
            if i < len(undefined_columns):  # Ensure there's an available placeholder
                df_target[undefined_columns[i]] = df[col]
                df_target = df_target.rename(columns={undefined_columns[i]: col})
            else:
                df_target['others'] = df[[col for col in unmatched_columns[i:]]].sum(axis=1)

        # removing all values when all nan values
        nan_columns = df_target.columns[df_target.isnull().all()]
        rename_dict = {col: '.' * i for i, col in enumerate(nan_columns)}
        df_target = df_target.rename(columns=rename_dict)
        return df_target


def write_bus_results_to_excel(
    calc: FlixPostXL, resample_by: Literal['YE', 'd', 'h'] = 'd', custom_output_file_path: str = 'default'
):
    """
    Save the in- and out-flows of every bus to an Excel file.

    Parameters
    ----------
    calc : FlixPostXL
        The flixPostXL object containing the calculation results.
    resample_by : str, optional
        The time frequency for resampling data (e.g., 'd' for daily), by default "d".
        Allowed values are 'YE' (yearly), 'd' (daily), and 'h' (hourly).
    custom_output_file_path : str, optional
        Custom path to save the Excel file

    Returns
    -------
    None
    """
    logger.info(f'...Writing Bus Results ({resample_by}) to Excel...')

    if custom_output_file_path == 'default':
        output_file_path = calc.folder
    else:
        output_file_path = custom_output_file_path

    filename = f'{calc.name}__Buses_{resample_by}.xlsx'
    path_excel = os.path.join(output_file_path, filename)

    for bus_name, bus_results in calc.bus_results.items():
        data = bus_results.to_dataframe('flow_rate')
        data = resample_data(
            data_frame=data, target_years=calc.years, resampling_by=resample_by, resampling_method='sum'
        )
        df_to_excel_w_chart(data, path_excel, bus_name, 'MWh', 'Time')

    logger.info(f'......Buses ({resample_by}) finished')


def write_component_results_to_excel(
    calc: FlixPostXL, resample_by: Literal['YE', 'd', 'h'] = 'd', custom_output_file_path: str = 'default'
):
    """
    Save the in- and out-flows of every component to an Excel file.

    Parameters
    ----------
    calc : FlixPostXL
        The flixPostXL object containing the calculation results.
    resample_by : str, optional
        The time frequency for resampling data (e.g., 'd' for daily), by default "d".
        Allowed values are 'YE' (yearly), 'd' (daily), and 'h' (hourly).
    custom_output_file_path : str, optional
        Custom path to save the Excel file

    Returns
    -------
    None
    """
    logger.info(f'...Writing Components Results ({resample_by}) to Excel...')

    if custom_output_file_path == 'default':
        output_file_path = calc.folder
    else:
        output_file_path = custom_output_file_path

    filename = f'{calc.name}__Comps_{resample_by}.xlsx'
    path_excel = os.path.join(output_file_path, filename)

    for comp_name, comp_results in calc.component_results.items():
        data = comp_results.to_dataframe('flow_rate')
        data = resample_data(
            data_frame=data, target_years=calc.years, resampling_by=resample_by, resampling_method='sum'
        )
        df_to_excel_w_chart(data, path_excel, comp_name, 'MWh', 'Time')

    logger.info(f'......Components ({resample_by}) finished')


def write_effects_per_comp_per_period_to_excel(calc: FlixPostXL, custom_output_file_path: str = 'default'):
    """
    Saving the effects of every component per period to excel
    """
    logger.info('...Writing Effects Results per Component...')

    if custom_output_file_path == 'default':
        output_file_path = calc.folder
    else:
        output_file_path = custom_output_file_path

    filename = f'{calc.name}__Effects_per_comp.xlsx'
    path_excel = os.path.join(output_file_path, filename)

    for effect in calc.effect_results:
        if effect.lower() == 'penalty':
            continue
        data = {}
        for component in calc.component_results:
            operation = calc.get_effects_of_element(component, effect, 'operation')
            invest = calc.get_effects_of_element(component, effect, 'invest_per_period')
            data[component] = invest + resample_data(pd.DataFrame(operation), calc.years, 'YE', 'sum').values.flatten()
        df = pd.DataFrame(data, index=calc.years)
        df_to_excel_w_chart(df, path_excel, effect, 'See Legend', 'Component Name', style='bar', bar_style='stacked')


def visualize_results(
    calc_results: FlixPostXL,
    effects_per_comp_and_year: bool = True,
    buses_yearly: bool = True,
    comps_yearly: bool = True,
    buses_daily: bool = True,
    comps_daily: bool = True,
    buses_hourly: bool = False,
    comps_hourly: bool = False,
) -> None:
    """
    Visualizes the results of a flixPostXL object.

    * The overview results are mainly used to compare yearly mean values
      between different years.

    * The annual results are used to go into detail about the heating
      production and storage usage in each year.

    * The buses results are used to look at all uses of energy balance.

    * The comps results are used to look at all Transformation processes
      in the different components.

    * The effects results are used to look at all effects. Effects are
      Costs, CO2 Funding, etc.

    * Daily mean values are enough for most use cases.

    * Hourly values are good for in-depth examinations, but take a long
      time to extract and save.

    * TAKE CARE: Writing hourly data to excel takes a significant amount of time for
      big Models with many Components.

    Parameters:
        overview (bool): Whether to write overview graphics. Default is True.
        annual_results (bool): Whether to write annual results graphics. Default is True.
        effects_shares (bool): Whether to write effect shares to excel. Default is True.
        buses_yearly (bool): Whether to write annual results for buses to excel. Default is True.
        comps_yearly (bool): Whether to write annual results for components to excel. Default is True.
        effects_yearly (bool): Whether to write annual results for effects to excel. Default is True.
        buses_daily (bool): Whether to write daily results for buses to excel. Default is True.
        comps_daily (bool): Whether to write daily results for components to excel. Default is True.
        effects_daily (bool): Whether to write daily results for effects to excel. Default is True.
        buses_hourly (bool): Whether to write hourly results for buses to excel. Default is False.
        comps_hourly (bool): Whether to write hourly results for components to excel. Default is False.
        effects_hourly (bool): Whether to write hourly results for effects to excel. Default is False.

    Returns:
        FlixPostXL: The calculated results.
    """
    logger.info('Writing Results to Excel (YE)...')
    if buses_yearly:
        write_bus_results_to_excel(calc_results, 'YE')
    if effects_per_comp_and_year:
        write_effects_per_comp_per_period_to_excel(calc_results)
    if comps_yearly:
        write_component_results_to_excel(calc_results, 'YE')
    logger.info('...Results to Excel (YE) finished...')

    logger.info('Writing Results to Excel (d)...')
    if buses_daily:
        write_bus_results_to_excel(calc_results, 'd')
    if comps_daily:
        write_component_results_to_excel(calc_results, 'd')
    logger.info('...Results to Excel (d) finished...')

    logger.info('Writing results to Excel (h)...')
    if buses_hourly:
        write_bus_results_to_excel(calc_results, 'h')
    if comps_hourly:
        write_component_results_to_excel(calc_results, 'h')
    logger.info('...Results to Excel (h) finished...')


################## PDF - OUTPUT ####################
def create_report(calc: FlixPostXL, path: str = 'report.pdf', connected_to: str = 'Fernwaerme', chunk_size: int = 4):
    logger.info(f"Creating Report of Components connected to '{connected_to}'")
    res = calc.bus_results[connected_to]
    flows_to_plot = [
        flow.label_full for flow in res.inputs + res.outputs if flow.to_dataframe('flow_rate').sum().sum() >= 1
    ]
    flows_to_plot.sort()

    with PdfPages(path) as pdf:
        for chunk in [flows_to_plot[i : i + chunk_size] for i in range(0, len(flows_to_plot), chunk_size)]:
            fig, axes = plt.subplots(
                len(chunk), 1, figsize=(8.27, 11.69 / 4 * len(chunk)), sharex=True, sharey=True
            )  # A4 size
            if not isinstance(axes, np.ndarray):  # If only one item in batch, axes is not a list...
                axes = np.array([axes])

            for ax, flow_name in zip(axes.flatten(), chunk, strict=False):
                fig, ax = calc.plot_operation_color_map(
                    flow_name, nb_of_periods=365 * len(calc.years), fig=fig, ax=ax, ylabel='time of day', xlabel='day'
                )
                ax.set_title(flow_name)  # Set individual title for each subplot

            pdf.savefig(fig)
            plt.close()


def create_report_grouped(
    calc: FlixPostXL, path: str = 'report.pdf', connected_to: str = 'Fernwaerme', chunk_size: int = 4
) -> None:
    logger.info(f"Creating Report of Components connected to '{connected_to}'")
    # Filtering and sorting
    grouped_flows = {'others': []}
    res = calc.bus_results[connected_to]
    flows_to_plot = [flow for flow in res.inputs + res.outputs if flow.to_dataframe('flow_rate').sum().sum() >= 1]

    for flow in flows_to_plot:
        group = calc.group_map.get(flow.label_full) or 'others'
        if group in grouped_flows.keys():
            grouped_flows[group].append(flow.label_full)
        else:
            grouped_flows[group] = [flow.label_full]
    for key in grouped_flows:
        grouped_flows[key].sort()
    groups = list(grouped_flows.keys())
    groups.sort()

    with PdfPages(path) as pdf:
        for group in groups:
            items_in_group = len(grouped_flows[group])
            for chunk in [grouped_flows[group][i : i + chunk_size] for i in range(0, items_in_group, chunk_size)]:
                fig, axes = plt.subplots(
                    len(chunk), 1, figsize=(8.27, 11.69 / 4 * len(chunk)), sharex=True, sharey=True
                )  # A4 size
                if not isinstance(axes, np.ndarray):  # If only one item in batch, axes is not a list...
                    axes = np.array([axes])

                for ax, flow_name in zip(axes.flatten(), chunk, strict=False):
                    fig, ax = calc.plot_operation_color_map(
                        flow_name,
                        nb_of_periods=365 * len(calc.years),
                        fig=fig,
                        ax=ax,
                        ylabel='time of day',
                        xlabel='day',
                    )
                    ax.set_title(flow_name)  # Set individual title for each subplot

                fig.text(0, 1, f'Group: {group}', ha='left', va='center', fontsize=14, weight='bold')  # Add big header
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()


def create_report_per_comp(calc: FlixPostXL, path: str = 'report.pdf') -> None:
    # Filtering and sorting
    components = {
        comp_label: sorted([flow.label_full for flow in comp.inputs + comp.outputs])
        for comp_label, comp in calc.component_results.items()
    }

    components_sorted = sorted(components.keys())

    with PdfPages(path) as pdf:
        for comp_name in components_sorted:
            nb_of_subplots = len(components[comp_name])
            fig, axes = plt.subplots(
                nb_of_subplots, 1, figsize=(8.27, 11.69 / 4 * nb_of_subplots), sharex=True, sharey=True
            )  # A4 size
            if not isinstance(axes, np.ndarray):  # If only one item in batch, axes is not a list...
                axes = np.array([axes])
            for ax, flow_name in zip(axes.flatten(), components[comp_name], strict=False):
                fig, ax = calc.plot_operation_color_map(
                    flow_name, nb_of_periods=365 * len(calc.years), fig=fig, ax=ax, ylabel='time of day', xlabel='day'
                )
                ax.set_title(flow_name)  # Set individual title for each subplot
            fig.text(0.5, 0.95, f'{comp_name}', ha='center', va='center', fontsize=14, weight='bold')  # Add big header

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
