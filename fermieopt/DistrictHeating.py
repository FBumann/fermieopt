import datetime
import logging
import os
import pathlib
import shutil
from typing import Dict, List, Optional

import flixOpt as fx
import flixOpt.elements
import flixOpt.structure
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console

from fermieopt.DistrictHeatingComps import ElementFactory, exists, extract_data, numbers_from_str
from fermieopt.excel_input import ExcelData
from fermieopt.flixPostprocessingXL import FlixPostXL

logger = logging.getLogger('flixOpt')


class ExcelModel:
    def __init__(self, excel_file_path: str):
        self.excel_data = ExcelData(file_path=pathlib.Path(excel_file_path))
        self.final_directory = self.excel_data.results_directory / self.excel_data.meta_data.calc_name
        self.final_model = fx.FlowSystem(time_series=self.excel_data.time_series_data.index)
        self._busses = self._create_busses()
        self._effects = self._create_effects()

        self.final_model.add_elements(*self._create_helpers())
        self.final_model.add_effects(*list(self._effects.values()))

        self._create_components()

    def print_comps_in_categories(self):
        # String-resources
        print('###############################################')
        print('Initiated Comps:')
        categorized_comps = {}
        for comp in self.district_heating_system.final_model.components:
            comp: flixOpt.elements.Component
            category = type(comp).__name__
            if category not in categorized_comps:
                categorized_comps[category] = [comp.label]
            else:
                categorized_comps[category].append(comp.label)

        for category, comps in categorized_comps.items():
            print(f'{category}: {comps}')

    def solve_model(self, solver_name: str, gap_frac: float = 0.01, timelimit: int = 3600):
        self.print_comps_in_categories()
        self._adjust_calc_name_and_results_folder()
        self._create_dirs_and_save_input_data()

        calculation = fx.FullCalculation(self.calc_name, self.district_heating_system.final_model, 'pyomo')
        calculation.do_modeling()

        calculation.solve(
            fx.solvers.GurobiSolver(mip_gap=gap_frac, time_limit_seconds=timelimit),
            save_results=os.path.join(self.final_directory, 'SolveResults'),
        )
        self.calc_name = calculation.name
        self.load_results()

        with open(os.path.join(self.final_directory, f'{self.calc_name}__calc_info.txt'), 'w') as log_file:
            calc_info = f"""calc = flixPostXL(nameOfCalc='{self.calc_name}',
            results_folder='{os.path.join(self.final_directory, 'SolveResults')}',
            outputYears={self.years})"""

            log_file.write(calc_info)

    def load_results(self) -> FlixPostXL:
        return FlixPostXL(
            calculation_name=self.calc_name,
            results_folder=os.path.join(self.final_directory, 'SolveResults'),
            output_years=self.years,
        )

    def _create_dirs_and_save_input_data(self):
        os.mkdir(self.final_directory)
        input_data_path = os.path.join(self.final_directory, f'{self.calc_name}__Skript.xlsx')
        shutil.copy2(self.input_excel_file_path, input_data_path)

        with pd.ExcelWriter(input_data_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            df = self.district_heating_system.time_series_data_internal
            df.to_excel(writer, index=True, sheet_name='Internally_computed_data')

        with open(
            os.path.join(self.final_directory, f'{self.calc_name}__Component_data.txt'), 'w', encoding='utf-8'
        ) as log_file:
            console = Console(file=log_file, width=10000)
            console.print(self.excel_data.components_data)

        try:
            with open(
                os.path.join(self.final_directory, f'{self.calc_name}__System_Description.txt'), 'w', encoding='utf-8'
            ) as log_file:
                console = Console(file=log_file, width=10000)
                console.print(self.district_heating_system.final_model)
        except Exception as e:
            logger.warning('Could not write System Description to file')
            logger.warning(f'Exception: {e}')

        try:
            with open(
                os.path.join(self.final_directory, f'{self.calc_name}__Input_and_Preprocessing_Comps.txt'),
                'w',
                encoding='utf-8',
            ) as log_file:
                console = Console(file=log_file, width=1000)
                console.print(self.district_heating_system.factory.print_comps())
        except Exception as e:
            logger.warning('Could not write Input and Preprocessing Components to file')
            logger.warning(f'Exception: {e}')

    def _adjust_calc_name_and_results_folder(self):
        now = datetime.datetime.now()
        self.calc_name = f'{now.strftime("%Y-%m-%d")}_{self.calc_name}'
        self.final_directory = os.path.join(self.excel_data.results_directory, self.calc_name)
        if os.path.exists(self.final_directory):
            for i in range(1, 100):
                calc_name = self.calc_name + '_' + str(i)
                final_directory = os.path.join(os.path.dirname(self.final_directory), calc_name)
                if not os.path.exists(final_directory):
                    self.calc_name = calc_name
                    self.final_directory = final_directory
                    if i >= 5:
                        print(
                            f'There are over {i} different calculations with the same name. '
                            f'Please choose a different name next time.'
                        )
                    if i >= 99:
                        raise Exception(
                            'Maximum number of different calculations with the same name exceeded. Max is 9999.'
                        )
                    break

    def _create_busses(self) -> Dict[str, fx.Bus]:
        busses = {}

        for bus_data in self.excel_data.components_data['Bus']:
            try:
                label = bus_data['Name']
                busses[label] = fx.Bus(label=label, excess_penalty_per_flow_hour=None)
            except KeyError as e:
                raise Exception(f"Every Bus needs a 'Name'! Error: {e}") from e

        return busses

    def _create_effects(self) -> Dict[str, fx.Effect]:
        effects = dict()
        effects['target'] = fx.Effect(
            'target',
            'i.E.',
            'Target',  # name, unit, description
            is_objective=True,
        )  # defining costs as objective of optimiziation
        effects['costs'] = fx.Effect(
            'costs',
            '€',
            'Kosten',
            is_standard=True,
            specific_share_to_other_effects_operation={effects['target']: 1},
            specific_share_to_other_effects_invest={effects['target']: 1},
        )

        effects['funding'] = fx.Effect(
            'funding',
            '€',
            'Funding Gesamt',
            specific_share_to_other_effects_operation={effects['costs']: -1},
            specific_share_to_other_effects_invest={effects['costs']: -1},
        )

        effects['CO2FW'] = fx.Effect('CO2FW', 't', 'CO2Emissionen der Fernwaerme')

        effects['CO2'] = fx.Effect(
            'CO2', 't', 'CO2Emissionen', specific_share_to_other_effects_operation={effects['CO2FW']: 1}
        )

        effects['Gruene_Waerme'] = fx.Effect('Gruene_Waerme', 'MWh', 'Menge an produzierter grüner Wärme')

        # Limit CO2 Emissions per year
        yearly_co2 = add_yearly_effects_with_bounds(
            effects['CO2FW'],
            years=self.years,
            lower_bounds=[None] * len(self.years),
            upper_bounds=self.excel_data.meta_data_time.co2_limits,
            label='CO2Limit',
            unit='t',
            description='Effect to limit the Emissions per year',
        )
        effects.update(yearly_co2)

        # Limit CO2 Emissions per year
        yearly_gw = add_yearly_effects_with_bounds(
            effects['Gruene_Waerme'],
            years=self.years,
            lower_bounds=self.excel_data.meta_data_time.green_heat_min,
            upper_bounds=[None] * len(self.years),
            label='Gruene_Waerme_Limits',
            unit='MWh',
            description='Effect to limit the Gruene_Waerme per year',
        )
        effects.update(yearly_gw)

        effects.update(self._create_invest_groups())
        return effects

    def _create_helpers(self) -> List[flixOpt.structure.Element]:
        p_out1 = fx.Flow(
            label='Strompreis',
            bus=self._busses['StromEinspeisung'],
            size=0,
            effects_per_flow_hour=extract_data('Strom', self.excel_data.time_series_data),
        )
        p_out2 = fx.Flow(
            label='Gaspreis',
            bus=self._busses['Erdgas'],
            size=0,
            effects_per_flow_hour=extract_data('Erdgas', self.excel_data.time_series_data),
        )
        p_out3 = fx.Flow(
            label='Wasserstoffpreis',
            bus=self._busses['Wasserstoff'],
            size=0,
            effects_per_flow_hour=extract_data('Wasserstoff', self.excel_data.time_series_data),
        )
        p_out4 = fx.Flow(
            label='EBSPreis',
            bus=self._busses['EBS'],
            size=0,
            effects_per_flow_hour=extract_data('EBS', self.excel_data.time_series_data),
        )

        return [
            fx.LinearConverter(
                label='HelperPreise',
                inputs=[],
                outputs=[p_out1, p_out2, p_out3, p_out4],
                conversion_factors=[{p_out1: 1, p_out2: 1, p_out3: 1, p_out4: 1}],
            )
        ]

    def _create_components(self) -> None:
        # data manipulation if a range is given for the start year for some components
        self._augment_components_with_several_start_years()

        element_factory = ElementFactory(
            flow_system=self.final_model,
            time_series_data=self.excel_data.time_series_data,
            co2_factors=self.excel_data.meta_data.co2_factors,
            years_of_model=self.excel_data.meta_data_time.years,
            busses=self._busses,
        )

        for comp_type, comp_instances_data in self.component_data.values():
            for comp_props in comp_instances_data:
                element_factory.create_energy_object(comp_type, comp_props)

    def _augment_components_with_several_start_years(self):
        for comp_type in self.excel_data.components_data:
            items_to_remove = []
            for component_data in self.excel_data.components_data[comp_type]:
                years = component_data.get('Startjahr')
                if isinstance(years, str):
                    try:
                        first_year, last_year = numbers_from_str(years)
                    except ValueError as e:
                        raise ValueError('"Startjahr" must be an integer or a string of format "min-max"') from e
                    first_year, last_year = int(first_year), int(last_year)
                    items_to_remove.append(component_data)
                    for year in self.excel_data.meta_data_time.years:
                        if first_year <= year <= last_year:
                            new_comp_data = component_data.copy()
                            new_comp_data['Startjahr'] = year
                            new_comp_data['Name'] = f'{new_comp_data["Name"]}_{year}'
                            self.excel_data.components_data[comp_type].append(new_comp_data)
            for item in items_to_remove:
                self.excel_data.components_data[comp_type].remove(item)

    def _create_invest_groups(self):
        effects = {}
        for comp_infos in self.excel_data.components_data.values():
            for comp in comp_infos:
                label = comp.get('Investgruppe')
                if isinstance(label, str) and label not in effects.keys():
                    limits = label.split(':')[-1]
                    if '-' in limits:
                        lb, ub = limits.split('-')
                        min_sum = float(lb)
                        max_sum = float(ub)
                    else:
                        min_sum = None
                        max_sum = float(limits)
                    label_new = label.replace(':', '')
                    effects[label] = fx.Effect(
                        label=label_new,
                        description='Limiting Investments per group',
                        unit='Stk',
                        minimum_total=min_sum,
                        maximum_total=max_sum,
                    )
        return effects

    @property
    def years(self) -> List[int]:
        return self.excel_data.meta_data_time.years

    @property
    def component_data(self):
        combined_components_data = self.excel_data.components_data.copy()
        for key, value in self.excel_data.flow_system_data.items():
            if key in combined_components_data:
                combined_components_data[key].extend(value)
            elif key == 'Bus':
                continue
            else:
                combined_components_data[key] = value
        return combined_components_data


def calculate_hourly_rolling_mean(series: pd.Series, window_size: int = 24) -> pd.Series:
    """
    Calculate the hourly rolling mean of a time series.

    Parameters:
    - series (pd.Series): Time series data with hourly values. It should be indexed with datetime.
    - window_size (int): Size of the rolling window. Default is 24.

    Returns:
    - pd.Series: Hourly rolling mean of the input time series.

    Raises:
    - ValueError: If the index of the series is not in datetime format or if the hourly step is not 1 hour.

    Example:
    ```
    hourly_data = pd.Series(...)  # Replace ... with your hourly data
    result = calculate_hourly_rolling_mean(hourly_data)
    ```

    """
    # Check if the index is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(series.index):
        raise ValueError('The index of the input series must be in datetime format.')

    # Check if the hourly step is 1 hour for every step
    hourly_steps = (series.index[1:] - series.index[:-1]).total_seconds() / 3600
    if not all(step == 1 for step in hourly_steps):
        raise ValueError('The time series must have a consistent 1-hour hourly step.')

    ser = series.copy()
    # Calculate the rolling mean using the specified window size
    rolling_mean = ser.rolling(window=window_size).mean()

    # Fill the missing values in 'rolling_mean' with the mean values of the series in this area
    rolling_mean.iloc[:window_size] = ser.iloc[:24].mean()

    return rolling_mean


def linear_interpolation_with_bounds(
    input_data: pd.Series, lower_bound: float, upper_bound: float, value_below_bound: float, value_above_bound: float
) -> pd.Series:
    """
    Apply linear interpolation within specified bounds and assign fixed values outside the bounds.

    Parameters:
    - input_data (pd.Series): Input dataset.
    - lower_bound (float): Lower bound for linear interpolation.
    - upper_bound (float): Upper bound for linear interpolation.
    - value_below_bound (float): Value assigned to points below the lower bound.
    - value_above_bound (float): Value assigned to points above the upper bound.

    Returns:
    - pd.Series: New series with linear interpolation within bounds and fixed values outside.

    Example:
    ```
    # Create a sample dataset
    input_series = pd.Series([8, 12, 18, 25, 22, 30, 5, 14], index=pd.date_range('2023-01-01', periods=8, freq='D'))

    # Apply linear interpolation with bounds
    result = linear_interpolation_with_bounds(input_series, 10, 20, 5, 30)
    print(result)
    ```

    """
    output_array = np.zeros_like(input_data)
    for i in range(len(input_data)):
        if input_data.iloc[i] <= lower_bound:
            output_array[i] = value_below_bound
        elif input_data.iloc[i] >= upper_bound:
            output_array[i] = value_above_bound
        else:
            output_array[i] = value_below_bound + (
                (value_below_bound - value_above_bound) / (lower_bound - upper_bound)
            ) * (input_data.iloc[i] - lower_bound)
    return pd.Series(output_array, index=input_data.index)


def add_yearly_effects_with_bounds(
    base_effect: fx.Effect,
    years: List[int],
    lower_bounds: List[Optional[float]],
    upper_bounds: List[Optional[float]],
    label: str,
    unit: str,
    description: str,
) -> Dict[str, fx.Effect]:
    """
    Creates multiple new Effects for yearly allocation of values. Gets values from the base_effect (Factor = 1).
    If no bounds are given, no effect is created.

    Args:
        base_effect: The base effect from which a share is taken (only for the refering year)
        years: All years in the calculation
        lower_bounds: The lower bounds for each year. Can be None.
        upper_bounds: The upper bounds for each year. Can be None.
        label: The label of the new effect
        unit: The unit of the new effect
        description: Descriptionof the new effect

    Returns:
        Dict with the labels of the new effects as keys and the new effects as values

    """
    yearly_effects = {}
    for year, lower_bound, upper_bound in zip(years, lower_bounds, upper_bounds, strict=False):
        if lower_bound is not None or upper_bound is not None:
            full_label = f'{label}{year}'
            yearly_effects[full_label] = fx.Effect(
                full_label, unit, description, minimum_operation=lower_bound, maximum_operation=upper_bound
            )

            base_effect.specific_share_to_other_effects_operation.update(
                {yearly_effects[full_label]: exists(first_year=year, lifetime=1, years_in_model=years)}
            )
    return yearly_effects
