# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import List, Dict, Any, Tuple, Optional


class ExcelData:
    """
    A class to handle Excel data related to energy modeling.

    Attributes:
        file_path (str): The path to the Excel file.
        _general_infos (pd.DataFrame): A DataFrame containing general information from the Excel file.
        results_directory (str): The directory where results are stored.
        calc_name (str): The name of the calculation.
        years (list): A list of years for the model.
        co2_limits (dict): A dictionary mapping years to CO2 limits.
        co2_factors (dict): A dictionary mapping sources to CO2 factors.
        time_series_data (pd.DataFrame): A DataFrame containing time series data.
        components_data (dict): A dictionary containing component data.
    """
    def __init__(self, file_path):
        """
        Initialize the ExcelData object with the given file path.

        Args:
            file_path (str): The path to the Excel file.
        """
        self.file_path: str = file_path
        meta_data_columns = ("Erzeuger Sheets",
                             "CO2 Faktor Erdgas [t/MWh_hu]",
                             "Name",
                             "Speicherort")
        yearly_columns = ("Jahre",
                          "Zeitreihen Sheets",
                          "Sonstige Zeitreihen Sheets",
                          "Fahrkurve Fernwärmenetz VL",
                          "Fahrkurve Fernwärmenetz RL",
                          "CO2-limit")
        meta_data, yearly_data = self._process_general_infos(meta_data_columns, yearly_columns)

        # Basic Information
        self.results_directory: str = meta_data["Speicherort"][0]
        self.calc_name: str = str(meta_data["Name"][0])
        self.co2_factors: dict = {"Erdgas": meta_data["CO2 Faktor Erdgas [t/MWh_hu]"][0]}
        self._sheetnames_components: List[str] = meta_data["Erzeuger Sheets"]

        # Information per year of the Model
        self.years: List[int] = yearly_data["Jahre"]
        self.co2_limits: List[int] = yearly_data["CO2-limit"]
        self._heating_network_temperature_curves_ff_info: List[str] = yearly_data["Fahrkurve Fernwärmenetz VL"]
        self._heating_network_temperature_curves_rf_info: List[str] = yearly_data["Fahrkurve Fernwärmenetz RL"]
        self._sheetnames_ts_data: List[str] = yearly_data["Zeitreihen Sheets"]
        sheetnames_ts_data_extra = yearly_data["Sonstige Zeitreihen Sheets"]
        self._sheetnames_ts_data_extra: Optional[List[str]] = None if all(name is None for name in sheetnames_ts_data_extra) else sheetnames_ts_data_extra
        self._validate_and_convert_types()

        # Extracting Information aboutHeating Network Temperature curves
        self.heating_network_temperature_curves = {
            "ff": self.validate_and_extract_factors(yearly_data["Fahrkurve Fernwärmenetz VL"]),
            "rf": self.validate_and_extract_factors(yearly_data["Fahrkurve Fernwärmenetz RL"])}

        # Time Series Data
        self.time_series_data: pd.DataFrame = self._read_time_series_data()
        validate_time_series_data(df=self.time_series_data, years=self.years)

        # Component Data
        self.components_data: Dict = self._read_components(sheet_names=self._sheetnames_components,
                                                     valid_types=(
                                                         'KWK', 'Kessel', 'Speicher', 'EHK', 'Waermepumpe',
                                                         'AbwaermeHT', 'AbwaermeWP', 'Rueckkuehler', 'KWKekt',
                                                         'Geothermie', 'LinearTransformer_1_1', 'Sink','Source'
                                                     ))
        self.further_components_data: Dict = self._read_components(sheet_names=["System"],
                                                     valid_types=('Bus', 'Sink', 'Source'))

    def validate_and_extract_factors(self, factor_infos: List[str]) -> List[Optional[Dict[str, float]]] :
        condition_1 = all(isinstance(info, str) for info in factor_infos)
        condition_2 = all(isinstance(info, type(None)) for info in factor_infos)
        if not (condition_1 or condition_2):
            raise Exception(f"Either specify heating Network curves for all years or for None")
        if condition_1:
            for i, curve in enumerate(factor_infos):
                factor_infos[i] = curve.replace(",", ".").replace(" ", "")
                if not re.match(r'^-?\d+/\d+;\d+/\d+$', curve):
                    raise Exception(f"Use Text to specify the Temperature Curve of the heating network. "
                                    f"Use Form: ' 'lb'/'value_lb';'ub'/'value_ub' '."
                                    f"Example:    '-8/120;10/95'.")
        factors = []
        for infos in factor_infos:
            if not infos:
                factors.append(None)
            else:
                lower, upper = infos.split(";")
                lower_bound, value_below_bound = lower.split("/")
                upper_bound, value_above_bound = upper.split("/")

                factors.append({
                    "lb": float(lower_bound),
                    "ub": float(upper_bound),
                    "value_lb": float(value_below_bound),
                    "value_ub": float(value_above_bound)})
        return factors

    def _validate_and_convert_types(self):
        # self.years
        for i in range(len(self.years)):
            if isinstance(self.years[i], float) and self.years[i]%int(self.years[i]) == 0:
                self.years[i] = int(self.years[i])
            elif isinstance(self.years[i], int):
                continue
            else:
                raise ValueError(f"Every year must be an Integer.")

        # self.results_directory
        if not os.path.exists(self.results_directory):
            raise Exception(f"The path '{self.results_directory}' for saving does not exist. Please create it first.")
        if not os.path.isdir(self.results_directory):
            raise Exception(f"The path '{self.results_directory}' for saving is not a directory.")

        # self.sheetnames_ts_data
        if not all(isinstance(name, str) for name in self._sheetnames_ts_data):
            raise Exception(f"Use Text to specify the Sheetnames of TimeSeries Data")
        if not len(self._sheetnames_ts_data) == len(self.years):
            raise Exception(f"The number of 'years' and the number of 'Zeitreihen Sheets' must match.")

        # self.sheetnames_ts_data_extra
        if self._sheetnames_ts_data_extra:
            if not all(isinstance(name, str) for name in self._sheetnames_ts_data_extra):
                raise Exception(f"Use Text to specify the Sheetnames of TimeSeries Data")
            if len(self._sheetnames_ts_data_extra) != 0 and len(self._sheetnames_ts_data_extra) != len(self.years):
                raise Exception(f"The number of 'years' and the number of 'Sonstige Zeitreihen Sheets' must match. "
                                f"You can also not use 'Sonstige Zeitreihen Sheets' at all. Just leave the lines blank")

        # self._sheetnames_components
        if not all(isinstance(name, str) for name in self._sheetnames_components):
            raise Exception(f"Use Text to specify the Sheetnames of Components")
        if len(self._sheetnames_components) == 0:
            raise Exception("At least One Sheet Name must be given")

    def _process_general_infos(self, meta_data_columns: Tuple, yearly_columns: Tuple) -> Tuple[Dict[str, List], Dict[str, List]]:
        '''
        Gets data from sheet 'Allgemeines' and checks if all needed columns are present
        Returns
        -------

        '''
        general_info = pd.read_excel(self.file_path, sheet_name="Allgemeines")
        general_info = general_info.replace({np.nan: None})

        for column_name in meta_data_columns + yearly_columns:
            if column_name not in general_info:
                raise Exception(f"Column '{column_name}' is missing in sheet 'Allgemeines'.")

        meta_data = general_info[list(meta_data_columns)].to_dict(orient='list')
        meta_data = {k: list(filter(None, v)) for k, v in meta_data.items()}  # Removing None values

        yearly_data = general_info[list(yearly_columns)].copy()
        yearly_data["Jahre"] = pd.to_numeric(yearly_data["Jahre"], errors='coerce')
        yearly_data = yearly_data.dropna(subset=["Jahre"])
        yearly_data = yearly_data.to_dict(orient='list')

        return meta_data, yearly_data

    def _read_time_series_data(self) -> pd.DataFrame:
        li = []
        for sheet_name in self._sheetnames_ts_data:
            df = pd.read_excel(self.file_path, sheet_name=sheet_name, skiprows=[1, 2])
            li.append(df)
        time_series_data = pd.concat(li, axis=0, ignore_index=True)  # Concatenate the DataFrames of the list

        if self._sheetnames_ts_data_extra:
            li = []
            for sheet_name in self._sheetnames_ts_data_extra:
                df = pd.read_excel(self.file_path, sheet_name=sheet_name, skiprows=[1, 2])
                li.append(df)

            time_series_data_extra = pd.concat(li, axis=0, ignore_index=True)  # Concatenate the DataFrames in the list
            time_series_data = pd.concat([time_series_data, time_series_data_extra], axis=1)

        # Adding the Index ain datetime format
        a_time_series = datetime(2021, 1, 1) + np.arange(8760*len(self.years)) * timedelta(hours=1)
        a_time_series = a_time_series.astype('datetime64')
        time_series_data.index = a_time_series

        return time_series_data

    def _read_components_from_sheet(self,  sheet_name: str, valid_types: tuple) -> Dict[str, pd.DataFrame]:
        df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None, nrows=30)
        component_data_by_type = organize_component_data_by_type(df, valid_types)
        print(f"Component Data of Sheet '{sheet_name}' was read sucessfully.")
        return component_data_by_type


    def _read_components(self, sheet_names: List[str], valid_types: tuple):
        component_data_by_type = {}
        for sheet_name in sheet_names:
            component_data = self._read_components_from_sheet(sheet_name, valid_types)
            component_data_by_type = combine_dicts_of_component_data(component_data_by_type, component_data)

        component_data_converted = convert_component_data_types(component_data_by_type)
        component_data_final = seperate_component_data_into_single_dicts(component_data_converted)
        print(f"Component Data from sheets '{sheet_names}' was read and converted")
        return component_data_final


def organize_component_data_by_type(df: pd.DataFrame, valid_types: tuple) -> Dict[str, pd.DataFrame]:
    """
    Processes component data from an Excel file, validating component types and organizing data into separate DataFrames.

    This function iterates through the DataFrame, validating the component types against a predefined list of accepted types.
    It then organizes the data into separate DataFrames for each component type, ensuring each DataFrame is properly formatted.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the component data read from an Excel file.

    Returns
    -------
    dict
        A dictionary where each key is a component type, and the value is a DataFrame containing the data for that component type.
    """

    # Check for invalid Comp types
    for typ in df.iloc[0, :].dropna():
        if typ not in valid_types: raise Exception(
            f"{typ} is not an accepted type of Component. Accepted types are: {valid_types}")

    # Iterate through unique values and create specific DataFrames for each type
    # Create a dictionary to store DataFrames for each unique value
    Erzeugerdaten = {}
    for value in valid_types:
        # Select columns where the first row has the current value
        subset_df = df.loc[:, df.iloc[0] == value]

        if subset_df.shape[1] <= 1: continue  # skip, if no data inside

        # Resetting the index and droping the first column
        subset_df = subset_df.drop(0).reset_index(drop=True)

        # Rename the Columns to the Values of the first row in the created dataframe and drop the first row
        subset_df.columns = subset_df.iloc[0]
        # Rename the column at position 0
        column_names = subset_df.columns.tolist()
        if len(column_names) != len(set(column_names)):
            raise Exception(f"There are Components [{value}] with the same Name. Please rename ({column_names})")
        column_names[0] = "category"
        subset_df.columns = column_names

        # subset_df = subset_df.drop(0).reset_index(drop=True)

        # Drop all unnecessary Rows and Cols from the dataframe
        subset_df = subset_df.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # set index to the first column
        subset_df.set_index('category', inplace=True)

        # Store the subset DataFrame in the dictionary
        Erzeugerdaten[value] = subset_df

    return Erzeugerdaten

def convert_component_data_types(component_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Converts the data types of component data and performs necessary assignments.

    This function iterates through each DataFrame in the provided dictionary, replacing 'nan' values with None and converting
    'ja'/'nein' strings to boolean values. It ensures that the data is in the correct format for further processing.

    Parameters
    ----------
    component_data : dict
        A dictionary where each key is a component type, and the value is a DataFrame containing the data for that component type.

    Returns
    -------
    dict
        The updated dictionary with DataFrames containing data with the correct data types.
    """

    for key, subset_df in component_data.items():
        # Replace all nan values with None
        subset_df.replace({np.nan: None}, inplace=True)

        # replace "ja" and "nein" with True and False
        subset_df.replace({'ja': True, 'Ja': True, 'True': True, 'true': True,
                           'nein': False, 'Nein': False, 'false': False, 'False': False}, inplace=True)

        # check if

    return component_data

def combine_dicts_of_component_data(component_data_1: Dict[str, pd.DataFrame],
                                    component_data_2: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    This function merges the DataFrames from two dictionaries, ensuring that there are no duplicate columns in each DataFrame.
    If duplicates are found, an exception is raised.

    Parameters
    ----------
    component_data_1 : dict
        The first dictionary of component data. component_type as key, value is a pd.DataFrame
    component_data_2 : dict
        The second dictionary of component data. component_type as key, value is a pd.DataFrame

    Returns
    -------
    dict
        A combined dictionary of component data, with DataFrames from both input dictionaries as values and component_types as keys
    """
    result_dict = {}
    for key in set(component_data_1.keys()) | set(component_data_2.keys()):
        if key in component_data_1 and key in component_data_2:
            duplicates = set(component_data_1[key].columns) & set(component_data_2[key].columns)
            if duplicates:  # if there are duplicates
                raise Exception(f"There are following Duplicates of type '{key}': {duplicates}'. Please rename them.")
            else:
                result_dict[key] = pd.concat([component_data_1[key], component_data_2[key]], axis=1)
        elif key in component_data_1:
            result_dict[key] = component_data_1[key].copy()
        elif key in component_data_2:
            result_dict[key] = component_data_2[key].copy()

    return result_dict

def seperate_component_data_into_single_dicts(Erzeugerdaten: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transforms component data into a format suitable for iterative processing.

    This function iterates over each component type in the provided dictionary, converting the corresponding DataFrame into a list of dictionaries.
    Each dictionary represents a component's data, excluding any entries with `None` values.
    This transformation facilitates the creation of components by allowing for easy iteration over the component data.

    Parameters
    ----------
    Erzeugerdaten : dict
        A dictionary mapping component types (as strings) to DataFrames containing the data for each component type.

    Returns
    -------
    dict
        A dictionary where each key is a component type, and the value is a list of dictionaries. Each dictionary within the list represents the data for a single component, with `None` values removed. This structure is optimized for iterative processing to create components.
    """
    ErzDaten = {}
    for typ in Erzeugerdaten:
        ErzDaten[typ] = list()
        for comp in Erzeugerdaten[typ].columns:
            erzeugerdaten_as_dict = Erzeugerdaten[typ][comp].to_dict()
            erzeugerdaten_as_dict_wo_none = {k: v for k, v in erzeugerdaten_as_dict.items() if v is not None}
            ErzDaten[typ].append(erzeugerdaten_as_dict_wo_none)
            if not ErzDaten[typ]:  # if list is empty
                ErzDaten.pop(typ)

    return ErzDaten

def validate_time_series_data(df: pd.DataFrame, years: List[int]) -> None:
    """
    Checks the consistency of a DataFrame with respect to the number of years and the presence of missing values.

    This function verifies that the length of the DataFrame matches the number of years provided and that there are no missing values.
    It raises an exception if either of these conditions is not met.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check for consistency.
    years : List[int]
        A list of years to compare against the DataFrame's length.

    Returns
    -------
    None
    """
    if len(df.index) / 8760 != len(years):
        raise Exception(f"Length of DataFrame: {len(df)}; Number of years: {len(years)}; Doesn't match.")

    columns_with_nan = df.columns[df.isna().any()]
    if not columns_with_nan.empty:
        raise Exception(f"There are missing values in the columns: {columns_with_nan}.")

