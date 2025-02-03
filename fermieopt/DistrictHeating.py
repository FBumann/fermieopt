import datetime
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional

import flixOpt as fx
import flixOpt.structure
from rich.console import Console

from fermieopt.config import BusLabels, EffectLabels, EnergyPriceLabels, SinkLabels, SourceLabels
from fermieopt.DistrictHeatingComps import ModelFactory, exists, extract_data
from fermieopt.excel_input import ExcelData
from fermieopt.flixPostprocessingXL import FlixPostXL
from fermieopt.meta_data import MetaDataFactory

logger = logging.getLogger('flixOpt')


class ExcelModel:
    """"
    Für Vorlagen zur Erstellung der verschiedenen Erzeuegr, siehe Template_Input.xlsx

    Die Vorlagen können auch neu erstellt werden mittels:

    ```python
        from fermieopt.DistrictHeatingComps import ModelFactory
        ModelFactory.model_templates(file_name='Template_Input.xlsx', sheet_name='Templates')
        ModelFactory.model_overview(file_name='Template_Input.xlsx', sheet_name='Doku')
    ```
    """
    _solvers = {
        'gurobi': fx.solvers.GurobiSolver,
        'highs': fx.solvers.HighsSolver,
    }

    def __init__(self, excel_file_path: str):
        self.excel_data = ExcelData(file_path=pathlib.Path(excel_file_path))
        self._timestamp = self._update_timestamp()
        MetaDataFactory.length = len(self.excel_data.period_data.years)
        self.final_model = fx.FlowSystem(time_series=self.excel_data.time_series_data.index)
        self._busses = self._create_busses()
        self._effects = self._create_effects()

        self.final_model.add_elements(*self._create_helpers())
        self.final_model.add_effects(*list(self._effects.values()))

        self._create_components()

        self._validate_predefined_elements()

    def solve_model(self, solver_name: str, gap_frac: float = 0.01, timelimit: int = 3600):
        self._update_timestamp()

        self._create_dirs_and_save_input_data()

        calculation = fx.FullCalculation(self.calc_name, self.final_model, 'pyomo')
        calculation.do_modeling()

        calculation.solve(
            self._solvers[solver_name](mip_gap=gap_frac, time_limit_seconds=timelimit),
            save_results=self._solve_results_folder,
        )

        with open(os.path.join(self.final_directory, f'{self.calc_name}__calc_info.txt'), 'w') as log_file:
            calc_info = f"""results = FlixPostXL(calculation_name='{self.calc_name}',
            results_folder='{os.path.join(self.final_directory, 'SolveResults')}',
            output_years={self.years})"""
            log_file.write(calc_info)

    def load_results(self) -> FlixPostXL:
        return FlixPostXL(
            calculation_name=self.calc_name,
            results_folder=self._solve_results_folder,
            output_years=self.years,
        )

    def _update_timestamp(self) -> None:
        self._timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%HH-%MM')

    def _create_dirs_and_save_input_data(self):
        os.makedirs(self.final_directory, exist_ok=True)
        shutil.copy2(self.excel_data.file_path, self.final_directory / f'{self.calc_name}__Skript.xlsx')

        with open(self.final_directory / f'{self.calc_name}__Excel_Data.json', 'w', encoding='utf-8') as log_file:
            print(self.excel_data.model_dump_json(indent=4, by_alias=True), file=log_file)
            logger.info('Excel Data written to file')

        with open(self.final_directory / f'{self.calc_name}__Component_data.json', 'w', encoding='utf-8') as log_file:
            import json

            json.dump(self.excel_data.components_data, log_file, indent=4, ensure_ascii=False)
            logger.info('Component Data written to file')

        with open(
            self.final_directory / f'{self.calc_name}__System_Description.txt', 'w', encoding='utf-8'
        ) as log_file:
            console = Console(file=log_file, width=10000)
            console.print(self.final_model)
            logger.info('System Description written to file')

    def _create_busses(self) -> Dict[str, fx.Bus]:
        busses = {}

        for bus_data in self.bus_data:
            try:
                label = bus_data['Name']
                busses[label] = fx.Bus(label=label, excess_penalty_per_flow_hour=None)
            except KeyError as e:
                raise Exception(f"Every Bus needs a 'Name'! Error: {e}") from e

        for bus in BusLabels.all_values():
            if bus not in busses:
                logger.critical(f'Bus "{bus}" fehlt. Bitte hinzufügen. Erstellte Busse: {list(busses)}')

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

        effects[EffectLabels.GREEN_HEAT] = fx.Effect(
            EffectLabels.GREEN_HEAT, 'MWh', 'Menge an produzierter grüner Wärme'
        )

        # Limit CO2 Emissions per year
        yearly_co2 = add_yearly_effects_with_bounds(
            effects['CO2FW'],
            years=self.years,
            lower_bounds=[None] * len(self.years),
            upper_bounds=self.excel_data.period_data.co2_limit,
            label='CO2Limit',
            unit='t',
            description='Effect to limit the Emissions per year',
        )
        effects.update(yearly_co2)

        # Limit CO2 Emissions per year
        yearly_gw = add_yearly_effects_with_bounds(
            effects[EffectLabels.GREEN_HEAT],
            years=self.years,
            lower_bounds=self.excel_data.period_data.green_heat_min,
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
            label=EnergyPriceLabels.ELECTRICITY,
            bus=self._busses[BusLabels.ELECTRICITY_OUT],
            size=0,
            effects_per_flow_hour={
                self._effects['costs']: extract_data(EnergyPriceLabels.ELECTRICITY, self.excel_data.time_series_data)
            },
        )
        p_out2 = fx.Flow(
            label=EnergyPriceLabels.GAS,
            bus=self._busses[BusLabels.GAS],
            size=0,
            effects_per_flow_hour={
                self._effects['costs']: extract_data(EnergyPriceLabels.GAS, self.excel_data.time_series_data)
            },
        )
        p_out3 = fx.Flow(
            label=EnergyPriceLabels.HYDROGEN,
            bus=self._busses[BusLabels.HYDROGEN],
            size=0,
            effects_per_flow_hour={
                self._effects['costs']: extract_data(EnergyPriceLabels.HYDROGEN, self.excel_data.time_series_data)
            },
        )

        return [
            fx.LinearConverter(
                label='Energiepreise',
                inputs=[],
                outputs=[p_out1, p_out2, p_out3],
                conversion_factors=[{p_out1: 1, p_out2: 1, p_out3: 1}],
            )
        ]

    def _create_components(self) -> None:
        element_factory = ModelFactory(
            flow_system=self.final_model,
            time_series_data=self.excel_data.time_series_data,
            co2_factors=self.excel_data.meta_data.co2_factors,
            years_of_model=self.excel_data.period_data.years,
            busses=self._busses,
        )

        for comp_type, comp_instances_data in self.components_data.items():
            for comp_props in comp_instances_data:
                element_factory.create_energy_object(comp_type, comp_props)

    def _create_invest_groups(self):
        effects = {}
        for comp_infos in self.excel_data.components_data.values():
            for comp in comp_infos:
                label = comp.get('Investgruppe')
                if label is None:
                    continue
                if not isinstance(label, str):
                    raise TypeError(f'Invest group must be a string, but is {label=}')
                if label in effects.keys():
                    raise ValueError(f'Cant create Invest Group. An Effect wit the label "{label}" already exists')
                items = label.split(':')
                if len(items) != 2:
                    raise ValueError(f'Invalid invest group: {label}. Must be None or of form: "name:limit"')
                label, limits = items
                if '-' in limits:
                    limits = limits.split('-')
                    if len(limits) != 2:
                        raise ValueError(
                            f'Invalid invest group: {label}. If limits are given, they must be of form: "min-max"'
                        )
                    lb, ub = limits
                    try:
                        lb = float(lb)
                        ub = float(ub)
                    except ValueError as e:
                        raise ValueError(f'Invalid invest group: {label}. Convrsion of limits to float failed') from e
                else:
                    lb = None
                    ub = float(limits)

                effects[label] = fx.Effect(
                    label=label,
                    description='Limiting Investments per group',
                    unit='Stk',
                    minimum_invest=lb,
                    maximum_invest=ub,
                )
        return effects

    @property
    def years(self) -> List[int]:
        return self.excel_data.period_data.years

    @property
    def components_data(self):
        combined_components_data = self.excel_data.components_data.copy()
        for key, value in self.excel_data.flow_system_data.items():
            if key in combined_components_data:
                combined_components_data[key].extend(value)
            elif key == 'Bus':
                continue
            else:
                combined_components_data[key] = value
        return combined_components_data

    @property
    def bus_data(self) -> List[Dict[str, Any]]:
        return self.excel_data.flow_system_data['Bus']

    @property
    def final_directory(self) -> pathlib.Path:
        return self.excel_data.meta_data.results_directory / self.calc_name

    @property
    def calc_name(self) -> str:
        return f'{self._timestamp}_{self.excel_data.meta_data.calc_name}'

    @property
    def _solve_results_folder(self) -> pathlib.Path:
        return self.final_directory / 'SolveResults'

    def _validate_predefined_elements(self):
        effect_labels = EffectLabels.all_values()
        bus_labels = BusLabels.all_values()
        sink_labels = SinkLabels.all_values()
        source_labels = SourceLabels.all_values()

        for bus in bus_labels:
            if bus not in self._busses:
                logger.critical(
                    f'Bus {bus} is missing. This might make the automated evaluation fail.'
                    f'The following Buses are needed: {bus_labels}'
                )

        for effect in effect_labels:
            if effect not in self._effects:
                logger.critical(
                    f'Effect {effect} is missing. This might make the automated evaluation fail.'
                    f'The following Effects are needed: {effect_labels}'
                )

        for sink in sink_labels:
            if sink not in self.final_model.components:
                logger.critical(
                    f'Sink {sink} is missing. This might make the automated evaluation fail.'
                    f'The following Sinks are needed: {sink_labels}'
                )

        for source in source_labels:
            if source not in self.final_model.components:
                logger.critical(
                    f'Source {source} is missing. This might make the automated evaluation fail.'
                    f'The following Sources are needed: {source_labels}'
                )


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
