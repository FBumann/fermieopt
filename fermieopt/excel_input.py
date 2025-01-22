import logging
import pathlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, ValidationError, field_serializer, field_validator, model_validator

logger = logging.getLogger('flixOpt')


class MetaData(BaseModel, populate_by_name=True):
    """
    A Pydantic model to represent metadata related to Excel data.
    """

    results_directory: pathlib.Path = Field(alias='Speicherort', description='The directory where results are stored.')
    calc_name: str = Field(alias='Name', description='The name of the calculation.')
    co2_factors: Dict[str, float] = Field(
        alias='CO2 Faktor Erdgas [t/MWh_hu]', description='A dictionary mapping sources to CO2 factors.'
    )
    sheets_components: List[str] = Field(alias='Erzeuger Sheets', description='A list of sheet names for components.')

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'MetaData':
        """
        Extracts the metadata from a DataFrame and validates it.

        Args:
            df (pd.DataFrame): The DataFrame the metadata.

        Returns:
            MetaData: A validated MetaData instance.
        """
        # Convert DataFrame to a dictionary with matching aliases
        data_dict = df.to_dict(orient='list')
        # Rename keys using Pydantic aliases
        alias_map = {field_name: field.alias for field_name, field in cls.model_fields.items()}
        attrs_with_single_value = {'results_directory', 'calc_name', 'co2_factors'}
        aliases_with_single_value = {alias_map[attr] for attr in attrs_with_single_value}

        # Extract first value from each entry
        for key in data_dict:
            if key in aliases_with_single_value:
                data_dict[key] = data_dict[key][0]

        # Validate and create an instance of MetaData
        try:
            return cls(**data_dict)
        except ValidationError as e:
            print('Validation error:', e)
            raise

    @field_validator('results_directory', mode='after')
    @classmethod
    def validate_results_directory(cls, path):
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"The path '{path}' does not exist.")
        if not path.is_dir():
            raise NotADirectoryError(f"The path '{path}' is not a directory.")
        return path

    @field_validator('co2_factors', mode='before')
    @classmethod
    def convert_co2_factors(cls, co2_factor_gas):
        if isinstance(co2_factor_gas, Union[float, int]):
            return {'Erdgas': co2_factor_gas}
        return co2_factor_gas

    @field_validator('sheets_components', mode='before')
    @classmethod
    def convert_sheets_components(cls, sheets_components):
        import math

        return [
            value
            for value in sheets_components
            if value is not None and not (isinstance(value, float) and math.isnan(value))
        ]

    @field_serializer('results_directory')
    def serialize_results_directory(self, file_path: pathlib.Path):
        return str(file_path)


class MetaDataTime(BaseModel, populate_by_name=True):
    sheets_time_series: List[str] = Field(
        alias='Zeitreihen Sheets', description='A list of sheet names for time series data.'
    )
    sheets_time_series_others: Optional[List[str]] = Field(
        alias='Sonstige Zeitreihen Sheets', description='An aditional list of sheet names for time series data.'
    )
    years: List[int] = Field(alias='Jahre')
    co2_limit: List[Optional[float]] = Field(alias='CO2-limit')  # TODO: rename to CO2-Limits [t/a]
    green_heat_min: List[Optional[float]] = Field(
        alias='Grüne Wärme Minimum [MWh]'
    )  # TODO: rename to Grüne Wärme Minimum [MWh/a]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'MetaDataTime':
        """
        Extracts the time series metadata from a DataFrame and validates it.

        Args:
            df (pd.DataFrame): The DataFrame with time series metadata.

        Returns:
            MetaDataTime: A validated MetaDataTime instance.
        """
        # Check if 'Jahre' column exists and handle it
        if 'Jahre' in df.columns:
            # Remove rows where 'Jahre' has no value (NaN or None)
            df = df.dropna(subset=['Jahre'])
        else:
            raise ValidationError("No 'Jahre' column found in the DataFrame.")

        # Convert DataFrame to a dictionary with matching aliases
        data_dict = df.to_dict(orient='list')

        # Validate and create an instance of MetaDataTime
        try:
            return cls(**data_dict)
        except ValidationError as e:
            print('Validation error:', e)
            raise

    @model_validator(mode='after')
    def _validate_list_lengths(self):
        """
        Ensures that all list attributes have the same length.
        """
        list_attrs = [
            field for field, field_info in self.__annotations__.items() if isinstance(getattr(self, field), list)
        ]
        lengths = {len(getattr(self, attr)) for attr in list_attrs}

        if len(lengths) > 1:
            raise ValueError(f'Not all list fields have the same length: {list_attrs}.')

    @field_validator('sheets_time_series_others')
    @classmethod
    def _sheetnames_ts_data_extra(cls, value):
        if all(pd.isna(x) or x is None for x in value):
            return None
        return value


class ExcelData(BaseModel, arbitrary_types_allowed=True, populate_by_name=True):
    """
    A Pydantic model to represent Excel data related to energy modeling.
    """

    file_path: pathlib.Path = Field(alias='File Path', description='The path to the Excel file.')

    meta_data: Optional[MetaData] = Field(default=None)
    meta_data_time: Optional[MetaDataTime] = Field(default=None)
    time_series_data: Optional[pd.DataFrame] = Field(None, description='A DataFrame containing time series data.')
    components_data: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None, description='A dictionary containing component data.'
    )
    flow_system_data: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None, description='A dictionary containing flow system data.'
    )
    _skip_read_data: bool = PrivateAttr(default=False)

    @model_validator(mode='after')
    def read_data_from_excel(self):
        """
        Loads the data from an Excel file into the ExcelData model.
        """

        if self._skip_read_data:
            # Skip reading data if flag is set
            return self

        logger.info('Creating ExcelData object from file %s', self.file_path)
        excel_file = pd.ExcelFile(self.file_path)
        if 'Allgemeines' not in excel_file.sheet_names:
            raise ValueError("The Excel file does not contain a 'Allgemeines' sheet.")

        meta_data_df = pd.read_excel(excel_file, sheet_name='Allgemeines')
        meta_data_df = meta_data_df.loc[:, ~meta_data_df.columns.str.startswith('Unnamed')]
        meta_data_df = meta_data_df.replace(
            {
                np.nan: None,
                '': None,
                'NaN': None,
                'None': None,
                'null': None,
                'NULL': None,
                'ja': True,
                'Ja': True,
                'nein': False,
                'Nein': False,
            }
        )

        # Create MetaData and MetaDataTime instances
        self.meta_data = MetaData.from_dataframe(meta_data_df)
        self.meta_data_time = MetaDataTime.from_dataframe(meta_data_df)

        # Extract time series data (assuming the second sheet contains time series data)
        logger.info('Reading data for years %s', self.meta_data_time.years)
        self.time_series_data = self._read_time_series_data(excel_file)

        # Extract component data (assuming it's in separate sheets named by component)
        self.components_data = self._read_components(
            excel_file,
            self.meta_data.sheets_components,
            valid_keys=[
                'KWK',
                'Kessel',
                'Speicher',
                'EHK',
                'Waermepumpe',
                'AbwaermeHT',
                'AbwaermeWP',
                'Rueckkuehler',
                'KWKekt',
                'Geothermie',
                'LinearTransformer_1_1',
            ],
        )
        self.flow_system_data = self._read_components(
            excel_file, sheets=['System'], valid_keys=['Bus', 'Sink', 'Source']
        )
        logger.info('Component Data from all sheets read sucessully.')

        return self

    @model_validator(mode='after')
    def validate_time_series_data(self):
        if len(self.time_series_data) / 8760 != len(self.meta_data_time.years):
            raise Exception(
                f'Length of DataFrame ({len(self.time_series_data)}) and the Number of years '
                f"({len(self.meta_data_time.years)} don't match. Expecting 8760 rows per year."
            )

        columns_with_nan = self.time_series_data.columns[self.time_series_data.isna().any()]
        if not columns_with_nan.empty:
            raise Exception(f'There are missing values in the time series data: {columns_with_nan}.')

        return self

    @model_validator(mode='after')
    def check_used_columns(self):
        if 'Vorlauftemperatur Fernwärmenetz [°C]' not in self.time_series_data.columns:
            logger.warning(
                'Column "Vorlauftemperatur Fernwärmenetz [°C]" was not found in the time series data. It is used as a default for multiple components.'
            )
        if 'Rücklauftemperatur Fernwärmenetz [°C]' not in self.time_series_data.columns:
            logger.warning(
                'Column "Rücklauftemperatur Fernwärmenetz [°C]" was not found in the time series data. It is used as a default for multiple components.'
            )
        return self

    @field_serializer('time_series_data')
    def serialize_time_series_data(self, value: pd.DataFrame):
        return value.to_json(date_format='iso', orient='split')

    @field_serializer('file_path')
    def serialize_file_path(self, file_path: pathlib.Path):
        return str(file_path)

    @field_validator('file_path', mode='after')
    @classmethod
    def validate_file_path(cls, value):
        return pathlib.Path(value) if isinstance(value, str) else value

    # Deserialize DataFrame from JSON when loading the model
    @field_validator('time_series_data', mode='before')
    @classmethod
    def deserialize_dataframe(cls, value: Optional[Union[str, pd.DataFrame]]) -> pd.DataFrame:
        if isinstance(value, str):
            return pd.read_json(value, orient='split')
        return value
    def _read_time_series_data(self, excel_file: pd.ExcelFile) -> pd.DataFrame:
        # Extract time series data (assuming the second sheet contains time series data)
        time_series_data = pd.concat(
            [
                pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=[1, 2])
                for sheet_name in self.meta_data_time.sheets_time_series
            ],
            axis=0,
            ignore_index=True,
        )
        if self.meta_data_time.sheets_time_series_others:
            time_series_data_extra = pd.concat(
                [
                    pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=[1, 2])
                    for sheet_name in self.meta_data_time.sheets_time_series_others
                ],
                axis=0,
                ignore_index=True,
            )
            time_series_data = pd.concat([time_series_data, time_series_data_extra], axis=1)
        # Adding the Index ain datetime format
        a_time_series = datetime(2021, 1, 1) + np.arange(8760 * len(self.meta_data_time.years)) * timedelta(hours=1)
        a_time_series = a_time_series.astype('datetime64')
        time_series_data.index = a_time_series
        return time_series_data

    def _read_components(
        self, excel_file: pd.ExcelFile, sheets: List[str], valid_keys: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        component_data_by_type = {}
        for sheet_name in sheets:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None, nrows=30)
            component_data = organize_component_data_by_type(df, valid_keys)
            component_data_by_type = combine_dicts_of_component_data(component_data_by_type, component_data)
            logger.info(f"Component Data of Sheet '{sheet_name}' was read sucessfully.")
        component_data_converted = convert_component_data_types(component_data_by_type)
        component_data_final = seperate_component_data_into_single_dicts(component_data_converted)
        return component_data_final

    def __eq__(self, other):
        if not isinstance(other, ExcelData):
            return NotImplemented

        # Ignore file_path for comparison

        # Compare meta_data and meta_data_time (assuming these implement __eq__)
        if self.meta_data != other.meta_data or self.meta_data_time != other.meta_data_time:
            return False

        # Compare time_series_data
        if not self.time_series_data.equals(other.time_series_data):
            return False

        # Compare components_data
        if not self._compare_nested_dicts(self.components_data, other.components_data):
            return False

        # Compare flow_system_data
        if not self._compare_nested_dicts(self.flow_system_data, other.flow_system_data):
            return False

        return True

    @staticmethod
    def _compare_nested_dicts(dict1, dict2):
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            if isinstance(dict1[key], list) and isinstance(dict2[key], list):
                if len(dict1[key]) != len(dict2[key]) or any(d1 != d2 for d1, d2 in zip(dict1[key], dict2[key], strict=False)):
                    return False
            elif dict1[key] != dict2[key]:
                return False
        return True


def organize_component_data_by_type(df: pd.DataFrame, valid_types: List[str]) -> Dict[str, pd.DataFrame]:
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
        if typ not in valid_types:
            raise Exception(f'{typ} is not an accepted type of Component. Accepted types are: {valid_types}')

    # Iterate through unique values and create specific DataFrames for each type
    # Create a dictionary to store DataFrames for each unique value
    erzeuger_daten = {}
    for value in valid_types:
        # Select columns where the first row has the current value
        subset_df = df.loc[:, df.iloc[0] == value]

        if subset_df.shape[1] <= 1:
            continue  # skip, if no data inside

        # Resetting the index and droping the first column
        subset_df = subset_df.drop(0).reset_index(drop=True)

        # Rename the Columns to the Values of the first row in the created dataframe and drop the first row
        subset_df.columns = subset_df.iloc[0]
        # Rename the column at position 0
        column_names = subset_df.columns.tolist()
        if len(column_names) != len(set(column_names)):
            raise Exception(f'There are Components [{value}] with the same Name. Please rename ({column_names})')
        column_names[0] = 'category'
        subset_df.columns = column_names

        # subset_df = subset_df.drop(0).reset_index(drop=True)

        # Drop all unnecessary Rows and Cols from the dataframe
        subset_df = subset_df.dropna(axis=0, how='all').dropna(axis=1, how='all')

        # set index to the first column
        subset_df.set_index('category', inplace=True)

        # Store the subset DataFrame in the dictionary
        erzeuger_daten[value] = subset_df

    return erzeuger_daten


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

    for subset_df in component_data.values():
        # Replace all nan values with None
        subset_df.replace({np.nan: None}, inplace=True)

        # replace "ja" and "nein" with True and False
        subset_df.replace(
            {
                'ja': True,
                'Ja': True,
                'True': True,
                'true': True,
                'nein': False,
                'Nein': False,
                'false': False,
                'False': False,
            },
            inplace=True,
        )

        # check if

    return component_data


def combine_dicts_of_component_data(
    component_data_1: Dict[str, pd.DataFrame], component_data_2: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
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


def seperate_component_data_into_single_dicts(
    erzeuger_daten: Dict[str, pd.DataFrame],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transforms component data into a format suitable for iterative processing.

    This function iterates over each component type in the provided dictionary, converting the corresponding DataFrame into a list of dictionaries.
    Each dictionary represents a component's data, excluding any entries with `None` values.
    This transformation facilitates the creation of components by allowing for easy iteration over the component data.

    Parameters
    ----------
    erzeuger_daten : dict
        A dictionary mapping component types (as strings) to DataFrames containing the data for each component type.

    Returns
    -------
    dict
        A dictionary where each key is a component type, and the value is a list of dictionaries. Each dictionary within the list represents the data for a single component, with `None` values removed. This structure is optimized for iterative processing to create components.
    """
    erzeuger_daten_seperated = {}
    for typ in erzeuger_daten:
        erzeuger_daten_seperated[typ] = list()
        for comp in erzeuger_daten[typ].columns:
            erzeugerdaten_as_dict = erzeuger_daten[typ][comp].to_dict()
            erzeugerdaten_as_dict_wo_none = {k: v for k, v in erzeugerdaten_as_dict.items() if v is not None}
            erzeuger_daten_seperated[typ].append(erzeugerdaten_as_dict_wo_none)
            if not erzeuger_daten_seperated[typ]:  # if list is empty
                erzeuger_daten_seperated.pop(typ)

    return erzeuger_daten_seperated


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
        raise Exception(f'There are missing values in the columns: {columns_with_nan}.')


def is_nan(value) -> bool:
    import math

    return value is None or value == '' or value == np.nan or isinstance(value, float) and math.isnan(value)
