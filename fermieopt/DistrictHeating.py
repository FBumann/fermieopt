# -*- coding: utf-8 -*-
import os
import shutil
import datetime
import numpy as np
import pandas as pd
from pprintpp import pprint as pp
from typing import Dict, List, Union

from flixOpt.flixStructure import cEffectType, cEnergySystem, cCalculation, cME, cFlow, cBus
from flixOpt.flixComps import cBaseLinearTransformer, cBaseComponent

from fermieopt.excel_input import ExcelData
from fermieopt.flixPostXL import SystemVisualization, SystemInfos, flixPostXL
from fermieopt.DistrictHeatingComps import ComponentFactory
from fermieopt.DistrictHeatingComps import check_min_max_format, exists


class ExcelModel:
    def __init__(self, excel_file_path: str):
        self.excel_data = ExcelData(file_path=excel_file_path)
        self.district_heating_system = DistrictHeatingSystem(self.excel_data)

        self.calc_name = self.excel_data.calc_name
        self.final_directory = os.path.join(self.excel_data.results_directory, self.calc_name)
        self.input_excel_file_path = excel_file_path
        self.years = self.excel_data.years

    @property
    def visual_representation(self):
        return SystemVisualization(es=self.district_heating_system.final_model).create_figure()

    def print_comps_in_categories(self):
        # String-resources
        print("###############################################")
        print("Initiated Comps:")
        categorized_comps = {}
        for comp in self.district_heating_system.final_model.listOfComponents:
            comp: cBaseComponent
            category = type(comp).__name__
            if category not in categorized_comps:
                categorized_comps[category] = [comp.label]
            else:
                categorized_comps[category].append(comp.label)

        for category, comps in categorized_comps.items():
            print(f"{category}: {comps}")

    def solve_model(self, solver_name: str, gap_frac: float = 0.01, timelimit: int = 3600):
        self.print_comps_in_categories()
        self._adjust_calc_name_and_results_folder()
        self._create_dirs_and_save_input_data()

        calculation = cCalculation(self.calc_name, self.district_heating_system.final_model, 'pyomo',
                                   pathForSaving=self.final_directory)  # create Calculation
        calculation.doModelingAsOneSegment()

        solver_props = {'gapFrac': gap_frac,  # solver-gap
                        'timelimit': timelimit,  # seconds until solver abort
                        'solver': solver_name,
                        'displaySolverOutput': True,  # ausführlicher Solver-resources.
                        }

        calculation.solve(solver_props, aPath=os.path.join(self.final_directory, "SolveResults"))
        self.calc_name = calculation.nameOfCalc
        calc_results = self.load_results()

        with open(os.path.join(self.final_directory, f"{self.calc_name}__calc_info.txt"), "w") as log_file:
            calc_info = f"""calc = flixPostXL(nameOfCalc='{self.calc_name}', 
            results_folder='{os.path.join(self.final_directory, 'SolveResults')}', 
            outputYears={self.years})"""

            log_file.write(calc_info)

        with open(os.path.join(calc_results.folder, f"{self.calc_name}__Main_Results.txt"), "w") as log_file:
            main_results = calc_results.infos["modboxes"]["info"][0]["main_results"]
            pp(main_results, log_file)

    def load_results(self) -> flixPostXL:
        return flixPostXL(nameOfCalc=self.calc_name,
                          results_folder=os.path.join(self.final_directory, "SolveResults"),
                          outputYears=self.years)

    def _create_dirs_and_save_input_data(self):
        os.mkdir(self.final_directory)
        input_data_path = os.path.join(self.final_directory, f"{self.calc_name}__Skript.xlsx")
        shutil.copy2(self.input_excel_file_path, input_data_path)

        with (pd.ExcelWriter(input_data_path, mode="a", engine="openpyxl", if_sheet_exists='overlay') as writer):
            df = self.district_heating_system.time_series_data_internal
            df.to_excel(writer, index=True, sheet_name="Internally_computed_data")

        with open(os.path.join(self.final_directory, f"{self.calc_name}__Component_data.txt"), "w", encoding='utf-8') as log_file:
            pp(self.excel_data.components_data, log_file)

        try:
            with open(os.path.join(self.final_directory, f"{self.calc_name}__System_Description.txt"), "w", encoding='utf-8') as log_file:
                print(self.district_heating_system.final_model, file=log_file)
        except:
            print()

        try:
            with open(os.path.join(self.final_directory, f"{self.calc_name}__Input_and_Preprocessing_Comps.txt"), "w", encoding='utf-8') as log_file:
                print(self.district_heating_system.factory.print_comps(), file=log_file)
        except:
            print()

    def _adjust_calc_name_and_results_folder(self):
        now = datetime.datetime.now()
        self.calc_name = f"{now.strftime('%Y-%m-%d')}_{self.calc_name}"
        self.final_directory = os.path.join(self.excel_data.results_directory, self.calc_name)
        if os.path.exists(self.final_directory):
            for i in range(1, 100):
                calc_name = self.calc_name + "_" + str(i)
                final_directory = os.path.join(os.path.dirname(self.final_directory), calc_name)
                if not os.path.exists(final_directory):
                    self.calc_name = calc_name
                    self.final_directory = final_directory
                    if i >= 5:
                        print(f"There are over {i} different calculations with the same name. "
                              f"Please choose a different name next time.")
                    if i >= 99:
                        raise Exception(f"Maximum number of different calculations with the same name exceeded. "
                                        f"Max is 9999.")
                    break


class DistrictHeatingSystem:
    def __init__(self, excel_data: ExcelData):
        self.time_series_data = excel_data.time_series_data
        self.time_series_data_internal = pd.DataFrame(index=self.time_series_data.index)
        self.components_data = {**excel_data.components_data,
                                "Sink": excel_data.further_components_data["Sink"],
                                "Source": excel_data.further_components_data["Source"]}
        self.bus_data = excel_data.further_components_data["Bus"]

        self.years = excel_data.years
        self.timeSeries = excel_data.time_series_data.index.to_numpy()
        self.co2_limits = excel_data.co2_limits
        self.co2_factors = excel_data.co2_factors
        self.heating_network_temperature_curves = excel_data.heating_network_temperature_curves

        self._handle_heating_network()

        self.final_model = cEnergySystem(timeSeries=self.timeSeries)
        self.busses = self.create_busses()
        self.effects = self.create_effects()

        self.helpers = self.create_helpers()
        self.factory = ComponentFactory(time_series_data=self.time_series_data,
                                   co2_factors=self.co2_factors,
                                   years_of_model=self.years,
                                   busses=self.busses,
                                   effects=self.effects)
        self.components = self.create_components()

        self.final_model.addEffects(*list(self.effects.values()))
        self.final_model.addElements(*self.helpers)
        self.final_model.addElements(*self.components)


    def create_effects(self) -> Dict[str, cEffectType]:
        effects = {}
        effects['target'] = cEffectType('target', 'i.E.', 'Target',  # name, unit, description
                                        isObjective=True)  # defining costs as objective of optimiziation
        effects['costs'] = cEffectType('costs', '€', 'Kosten', isStandard=True,
                                       specificShareToOtherEffects_operation={effects['target']: 1},
                                       specificShareToOtherEffects_invest={effects['target']: 1})

        effects['funding'] = cEffectType('funding', '€', 'Funding Gesamt',
                                         specificShareToOtherEffects_operation={effects['costs']: -1},
                                         specificShareToOtherEffects_invest={effects['costs']: -1})

        effects['CO2FW'] = cEffectType('CO2FW', 't', 'CO2Emissionen der Fernwaerme')

        effects['CO2'] = cEffectType('CO2', 't', 'CO2Emissionen',
                                     specificShareToOtherEffects_operation={effects['CO2FW']: 1})

        # Limit CO2 Emissions per year
        co2_limiter_shares = {}
        for year, co2_limit in zip(self.years, self.co2_limits):
            if co2_limit is not None:
                effects[f"CO2Limit{year}"] = cEffectType(f"CO2Limit{year}", 't',
                                                         description="Effect to limit the Emissions in that year",
                                                         max_operationSum=co2_limit)
                co2_limiter_shares[effects[f"CO2Limit{year}"]] = exists(first_year=year, lifetime=1, years_in_model=self.years)
        effects['CO2FW'].specificShareToOtherEffects_operation.update(co2_limiter_shares)

        effects.update(self.create_invest_groups())
        return effects

    def create_invest_groups(self):
        effects = {}
        for key, comp_type in self.components_data.items():
            for comp in comp_type:
                label = comp.get("Investgruppe")
                if isinstance(label, str) and label not in effects.keys():
                    limits = label.split(":")[-1]
                    if "-" in limits:
                        lb, ub = limits.split("-")
                        min_sum = float(lb)
                        max_sum = float(ub)
                    else:
                        min_sum = None
                        max_sum = float(limits)
                    label_new = label.replace(":", "")
                    effects[label] = cEffectType(label=label_new, description="Limiting Investments per group",
                                                 unit="Stk", min_Sum=min_sum, max_Sum=max_sum)
        return effects

    def create_busses(self) -> Dict:
        busses = {}

        for bus_data in self.bus_data:
            try:
                name = bus_data["Name"]
            except KeyError as e:
                raise Exception(f"Every Bus needs a 'Name'! Error: {e}")
            try:
                media = bus_data["Medium"]
            except KeyError as e:
                raise Exception(f"Every Bus needs a 'medium'!  Error: {e}")

            busses[name] = cBus(label=name, media=media, excessCostsPerFlowHour=None)

        return busses

    def create_helpers(self) -> List[cME]:
        Pout1 = cFlow(label="Strompreis",
                      bus=self.busses['StromEinspeisung'],
                      nominal_val=0,
                      costsPerFlowHour=self.time_series_data["Strom"])
        Pout2 = cFlow(label="Gaspreis",
                      bus=self.busses['Erdgas'],
                      nominal_val=0,
                      costsPerFlowHour=self.time_series_data["Erdgas"])
        Pout3 = cFlow(label="Wasserstoffpreis",
                      bus=self.busses['Wasserstoff'],
                      nominal_val=0,
                      costsPerFlowHour=self.time_series_data["Wasserstoff"])
        Pout4 = cFlow(label="EBSPreis",
                      bus=self.busses['EBS'],
                      nominal_val=0,
                      costsPerFlowHour=self.time_series_data["EBS"])

        return[cBaseLinearTransformer(label="HelperPreise", inputs=[], outputs=[Pout1, Pout2, Pout3, Pout4],
                                   factor_Sets=[{Pout1: 1, Pout2: 1, Pout3: 1, Pout4: 1}])
                                   ]

    def augment_components_with_several_start_years(self):
        for comp_type in self.components_data:
            items_to_remove = []
            for component_data in self.components_data[comp_type]:
                years = component_data.get("Startjahr")
                if isinstance(years, str):
                    try:
                        first_year, last_year = check_min_max_format(years)
                    except ValueError:
                        raise ValueError(f'"Startjahr" must be an integer or a string of format "min-max"')
                    first_year, last_year = int(first_year), int(last_year)
                    items_to_remove.append(component_data)
                    for year in self.years:
                        if first_year <= year <= last_year:
                            new_comp_data = component_data.copy()
                            new_comp_data["Startjahr"] = year
                            new_comp_data["Name"] = f"{new_comp_data['Name']}_{year}"
                            self.components_data[comp_type].append(new_comp_data)
            for item in items_to_remove:
                self.components_data[comp_type].remove(item)

    def create_components(self) -> List[cME]:
        # data manipulation if a range is given for the start year for some components
        self.augment_components_with_several_start_years()

        pp(self.components_data)
        comps = []

        for comp_type in self.components_data.keys():
            for comp_props in self.components_data[comp_type]:
                comps.extend(self.factory.create_energy_object(comp_type, **comp_props))

        return comps


    def _handle_heating_network(self):
        """
        # TODO: Redo docstring
        Handle heating network parameters in the input DataFrame.

        This function calculates or checks the presence of key parameters related to the heating network,
        including supply temperature (TVL_FWN), return temperature (TRL_FWN), and network losses (SinkLossHeat).
        If not already present in the dataframe, creates them and returns the filled dataframe


        Raises:
        - Exception: If one of "TVL_FWN" or "TRL_FWN" is not present in the input DataFrame and needs calculation.

        Example:
        ```python
        handle_heating_network(my_dataframe)
        ```

        """

        self.time_series_data['Tamb24mean'] = calculate_hourly_rolling_mean(series=self.time_series_data['Tamb'],
                                                                            window_size=24)
        self.time_series_data_internal['Tamb24mean'] = self.time_series_data['Tamb24mean']
        # Check i fTermperatures are given directly as Time Series
        if "TVL_FWN" and "TRL_FWN" in self.time_series_data.keys():
            print("TVL_FWN and TRL_FWN where included in the input data set")
            return
        elif "TVL_FWN" in self.time_series_data.keys() or "TRL_FWN" in self.time_series_data.keys():
            raise Exception("Either include both or None of 'TVL_FWN' and 'TRL_FWN' in the Input Dataset")

        # Check if Fators are given
        if any(item is None for item in self.heating_network_temperature_curves["ff"]):
            raise Exception(f"If 'TVL_FWN' and 'TRL_FWN' are not provided, factors for temperature curves are needed")
        if any(item is None for item in self.heating_network_temperature_curves["rf"]):
            raise Exception(f"If 'TVL_FWN' and 'TRL_FWN' are not provided, factors for temperature curves are needed")

        # Berechnung der Netzwerktemperaturen
        df_tvl = pd.Series()
        for i, factors in enumerate(self.heating_network_temperature_curves["ff"]):
            df = linear_interpolation_with_bounds(input_data=self.time_series_data["Tamb24mean"].iloc[i*8760:(i+1)*8760],
                                                  lower_bound=factors["lb"],
                                                  upper_bound=factors["ub"],
                                                  value_below_bound=factors["value_lb"],
                                                  value_above_bound=factors["value_ub"])
            df_tvl = pd.concat([df_tvl, df])
        self.time_series_data["TVL_FWN"] = df_tvl
        self.time_series_data_internal["TVL_FWN"] = df_tvl

        df_trl = pd.Series()
        for i, factors in enumerate(self.heating_network_temperature_curves["rf"]):
            df = linear_interpolation_with_bounds(input_data=self.time_series_data["Tamb24mean"].iloc[i*8760:(i+1)*8760],
                                                  lower_bound=factors["lb"],
                                                  upper_bound=factors["ub"],
                                                  value_below_bound=factors["value_lb"],
                                                  value_above_bound=factors["value_ub"])
            df_trl = pd.concat([df_trl, df])
        self.time_series_data["TRL_FWN"] = df_trl
        self.time_series_data_internal["TRL_FWN"] = df_trl

        if "SinkLossHeat" not in self.time_series_data.keys():  # Berechnung der Netzverluste
            k_loss_netz = 0.4640  # in MWh/K        # Vereinfacht, ohne Berücksichtigung einer sich ändernden Netzlänge
            # TODO: Factor into excel
            self.time_series_data["SinkLossHeat"] = (k_loss_netz *
                                                     ((self.time_series_data["TVL_FWN"] + self.time_series_data["TRL_FWN"]) / 2 -
                                                      self.time_series_data["Tamb"]))
            self.time_series_data_internal["SinkLossHeat"] = self.time_series_data["SinkLossHeat"]
            print("Heating losses where calculated")
        else:
            print("Heating losses where included in the input data set")


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
        raise ValueError("The index of the input series must be in datetime format.")

    # Check if the hourly step is 1 hour for every step
    hourly_steps = (series.index[1:] - series.index[:-1]).total_seconds() / 3600
    if not all(step == 1 for step in hourly_steps):
        raise ValueError("The time series must have a consistent 1-hour hourly step.")

    ser = series.copy()
    # Calculate the rolling mean using the specified window size
    rolling_mean = ser.rolling(window=window_size).mean()

    # Fill the missing values in 'rolling_mean' with the mean values of the series in this area
    rolling_mean.iloc[:window_size] = ser.iloc[:24].mean()

    return rolling_mean


def linear_interpolation_with_bounds(input_data: pd.Series, lower_bound: float, upper_bound: float,
                                     value_below_bound: float, value_above_bound: float) -> pd.Series:
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
            output_array[i] = (value_below_bound +
                               ((value_below_bound - value_above_bound) / (lower_bound - upper_bound)) *
                               (input_data.iloc[i] - lower_bound))
    return pd.Series(output_array, index=input_data.index)

