# -*- coding: utf-8 -*-
import os
from typing import Literal, Optional, Dict, List, Tuple, Union
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import flixOpt as fx

logger = logging.getLogger('flixOpt')


class flixPostXL(fx.results.CalculationResults):
    """
    Extending the functionality of fx.results.CalculationResults, this class interacts with meta_data
    and other extra data stored in the results to create extra functionailty.
    The most relevant is the computation of investment effects per period, making the evaluation of effects over a
    mutli-year-periiod way easier.
    Further, it has a function to get the total effects of an Element, including indirect
    effects through shares between effects.
    Further, some comforting attributes for grouping and invetsment results are added.
    """
    def __init__(self, nameOfCalc, results_folder, outputYears):
        super().__init__(calculation_name=nameOfCalc, folder=results_folder)

        self.name = self.calculation_infos['Calculation']['Name']

        self.group_map = self._add_group_mapping()
        self.years = outputYears  # add as attribute
        self.folder = os.path.dirname(results_folder)

        self.investment_effects_per_period = self._get_investment_effects_per_period()

        self.shares_between_effects_operation, self.shares_between_effects_invest = (
            self._get_factors_between_effects())

        self._write_investment_effects_per_period_into_effect_results()

        self._validate_effects_computation()

    def sizes(self) -> Dict[str, Union[int, float]]:
        return {label: element_info['size'] for label, element_info in self.investment_infos().items()}

    @property
    def storages(self) -> List[str]:
        return [comp for comp, comp_results in self.component_results.items()
                if 'Storage' in comp_results.all_infos['class'].split(':')]

    def investment_infos(self) -> Dict[str, Union[int, float, Dict[str, np.ndarray[float]]]]:
        investment_infos = {}

        for component in self.component_results.values():
            if 'Investment' in component.all_results:
                investment_infos[component.label] = {
                    'size': component.all_results['Investment']['size'],
                    'is_invested': component.all_results['Investment'].get('isInvested', 1),
                    'fixed_effects': component.all_infos['meta_data'].get('fixed_effects', {}),
                    'specific_effects': component.all_infos['meta_data'].get('specific_effects', {})
                }

        for flow in self.flow_results().values():
            if 'Investment' in flow.all_results:
                investment_infos[flow.label_full] = {
                    'size': flow.all_results['Investment']['size'],
                    'is_invested': flow.all_results['Investment'].get('isInvested', 1),
                    'fixed_effects': flow.all_infos['meta_data'].get('fixed_effects', {}),
                    'specific_effects': flow.all_infos['meta_data'].get('specific_effects', {})
                }

        return investment_infos

    def _get_investment_effects_per_period(
            self) -> Dict[str, Dict[str, Dict[Literal['fixed_effects', 'specific_effects'], np.ndarray[float]]]]:

        invest_effects_per_period = {effect: {} for effect in self.effect_results}

        for element, element_infos in self.investment_infos().items():
            used_effects = list(
                set(element_infos.get('fixed_effects', {}).keys()).union(
                    element_infos.get('specific_effects', {}).keys())
            )
            for effect in used_effects:
                invest_effects_per_period[effect][element] = {
                    'fixed_effects':
                        element_infos['fixed_effects'].get(effect, np.array([0])) * element_infos['is_invested'],
                    'specific_effects':
                        element_infos['specific_effects'].get(effect, np.array([0])) * element_infos['size']
                }

        # Validate, that the sum of the computed invest effects per Period match the values from the optimization
        logger.debug('Validating Investment Effects...')
        for effect_label, effect_results in self.effect_results.items():
            if effect_label == 'Penalty':
                continue
            for element, new_result in invest_effects_per_period[effect_label].items():
                old_result = {
                    'fixed_effects': effect_results.all_results['invest']['Shares'].get(f'{element}__fix_effects', 0),
                    'specific_effects': effect_results.all_results['invest']['Shares'].get(f'{element}__specific_effects', 0)
                }

                if sum(new_result['fixed_effects']) != old_result['fixed_effects']:
                    logger.critical(
                        f'Getting the fixed investment effects per Period was not succesfull for {element=}.'
                        f'The value from the optimizer {old_result["fixed_effects"]} differs from the self '
                        f'computed value {sum(new_result["fixed_effects"])}.')
                if sum(new_result['specific_effects']) != old_result['specific_effects']:
                    logger.critical(
                        f'Getting the specific investment effects per Period was not succesfull for {element=}.'
                        f'The value from the optimizer {old_result["specific_effects"]} differs from the self '
                        f'computed value {sum(new_result["specific_effects"])}.')

        return invest_effects_per_period

    def _get_factors_between_effects(self) -> Tuple[Dict[Tuple[str, str], Union[int, float, np.ndarray[float]]],
                                                    Dict[Tuple[str, str], Union[int, float, np.ndarray[float]]]]:
        shares_invest = {}
        for effect, effect_results in self.effect_results.items():
            shares_invest[effect] = {
                target: factor for target, factor in
                effect_results.all_infos.get('specific_share_to_other_effects_invest', {}).items()
            }
        shares_invest = compute_conversion_factors(shares_invest)

        shares_operation = {}
        for effect, effect_results in self.effect_results.items():
            shares_operation[effect] = {
                target: factor for target, factor in
                effect_results.all_infos.get('specific_share_to_other_effects_operation', {}).items()
            }
        shares_operation = compute_conversion_factors(shares_operation)
        return shares_operation, shares_invest

    def get_effects_of_element(self,
                               element_label: str,
                               effect_label: str,
                               domain: Literal['invest', 'operation', 'invest_per_period']
                               ) -> Union[int, float, np.ndarray[float]]:
        """
        This function returns the effects introduced by an element.
        If the Element is a Component, effects of sub elements are included.
        The returned value includes direct and indirect effects.
        """
        labels = [element_label]
        if element_label in self.component_results:
            component = self.component_results[element_label]
            labels += [flow.label_full for flow in component.inputs + component.outputs]

        if domain == 'operation':
            total = np.zeros(len(self.time))
            conversion_factors = {key[0]: value for key, value in self.shares_between_effects_operation.items() if
                                  key[1] == effect_label}
            conversion_factors[effect_label] = 1  # Share to itself is 1
            for effect, conversion_factor in conversion_factors.items():
                for origin, value in self.effect_results[effect].all_results['operation']['Shares'].items():
                    if any([origin.startswith(f'{label}__') for label in labels]):
                        total =  total + value * conversion_factor

        elif domain == 'invest':
            total = 0
            conversion_factors = {key[0]: value for key, value in self.shares_between_effects_invest.items() if
                                  key[1] == effect_label}
            conversion_factors[effect_label] = 1  # Share to itself is 1
            for effect, conversion_factor in conversion_factors.items():
                for origin, value in self.effect_results[effect].all_results['invest']['Shares'].items():
                    if any([origin.startswith(f'{label}__') for label in labels]):
                        total =  total + value * conversion_factor
        elif domain == 'invest_per_period':
            total = np.zeros_like(self.years)
            conversion_factors = {key[0]: value for key, value in self.shares_between_effects_invest.items() if
                                  key[1] == effect_label}
            conversion_factors[effect_label] = 1  # Share to itself is 1
            for effect, conversion_factor in conversion_factors.items():
                for origin, value in self.effect_results[effect].all_results['invest']['Shares_per_period'].items():
                    if any([origin.startswith(f'{label}__') for label in labels]):
                        total =  total + value * conversion_factor
        else:
            logger.critical(f'Not allowed domain. Must be in {["invest", "operation", "invest_per_period"]}')
            total = 0

        return total

    def get_effect_results(self, effect_label: str,
                           origin: Literal["operation", "invest", "all", "invest_per_period"],
                           as_TS: bool = False,
                           shares: bool = False
                           ) -> Union[float, np.ndarray, Dict[str, Union[float, np.ndarray]]]:
        '''
        This functions returns the results of the chosen effect
        :param effect_label: Label of effect
        :param origin: Choose from ["operation","invest","all"]
        :param as_TS: Wether to return the values per timestep or a sum
        :param shares: Wether to return the shares to the effect
        :return: np.ndarray
        '''
        if effect_label not in self.effect_results:
            logger.critical(f'Effect {effect_label} not found in results')
            return None
        results: Dict = self.effect_results[effect_label].all_results

        if origin == 'invest':
            if shares:
                return results['invest']['Shares']
            return results['invest']['invest_sum']

        elif origin == 'operation':
            if shares:
                return results['operation']['Shares']
            if as_TS:
                return results['operation']['operation_sum_TS']
            return results['operation']['operation_sum']

        elif origin == 'all':
            if shares:
                return results['all']['Shares']
            return results['all']['all_sum']

        elif origin == 'invest_per_period':
            if shares:
                return results['invest']['Shares_per_period']
            return results['invest']['invest_per_period']

        else:
            raise ValueError(f"Invalid parameter: {origin}")

    def _validate_effects_computation(self):
        """
        This function checks if the method get_effects_of_element() sucessfully allocates all effects for an element.
        It does so by comparing the retrieved effects for all Elements with the total effect value.
        If there is a mismatch, it logs the error

        """
        for effect in self.effect_results:
            if effect == 'Penalty':
                continue
            total = self.effect_results[effect].all_results['invest']['invest_sum']
            individual = [self.get_effects_of_element(component, effect, 'invest') for component in self.component_results]
            computed_total = sum(individual)
            if abs(abs(computed_total)-abs(total)) > 1e-5:
                logger.critical(f'Total of individual results for {effect=:>25} {"invest":<10} doesnt match computation after '
                                f'solve: {computed_total=:>20.5f}     {total=:>20.5f}')

            total = np.sum(self.effect_results[effect].all_results['operation']['operation_sum_TS'])
            individual = [np.sum(self.get_effects_of_element(component, effect, 'operation')) for component in self.component_results]
            computed_total = sum(individual)
            if abs(abs(computed_total)-abs(total)) > 1e-5:
                logger.critical(f'Total of individual results for {effect=:>25} {"operation":<10} doesnt match computation after '
                                f'solve: {computed_total=:>20.5f}     {total=:>20.5f}')

            total = np.sum(self.effect_results[effect].all_results['invest']['invest_sum'])
            individual = [np.sum(self.get_effects_of_element(component, effect, 'invest_per_period')) for component in self.component_results]
            computed_total = sum(individual)
            if abs(abs(computed_total)-abs(total)) > 1e-5:
                logger.critical(f'Total of individual results for {effect=:>25} {"invest_per_period":<10} doesnt match computation after '
                                f'solve: {computed_total=:>20.5f}     {total=:>20.5f}')

            total = np.sum(self.effect_results[effect].all_results['invest']['invest_per_period'])
            individual = [np.sum(list(self.effect_results[effect].all_results['invest']['Shares_per_period'].values()))]
            computed_total = sum(individual)
            if abs(abs(computed_total)-abs(total)) > 1e-5:
                logger.critical(f'Total of individual results for {effect=:>25} {"invest_per_period":<10} doesnt match computation after '
                                f'solve: {computed_total=:>20.5f}     {total=:>20.5f}')
    def _write_investment_effects_per_period_into_effect_results(self):
        """
        This function writes the investment effects per period into the EffectResults.
        For now without shares between effects!
        """

        # Computing individual effects per period
        for effect, effect_results in self.effect_results.items():
            if effect == 'Penalty':
                continue
            effects_per_period = {
                f"{key}__{sub_key}": sub_value
                for key, value in self.investment_effects_per_period[effect].items()
                for sub_key, sub_value in value.items()
                if not np.all(sub_value == 0)
            }
            effect_results.all_results['invest']['Shares_per_period'] = {
                key: value for key, value in effects_per_period.items()
            }

        # Computing total of effect per period
        for effect, effect_results in self.effect_results.items():
            if effect == 'Penalty':
                continue
            effect_results.all_results['invest']['invest_per_period'] = np.sum(
                list(effect_results.all_results['invest']['Shares_per_period'].values()), axis=0
            )

        # TODO: THis might not be entirely viable
        additional_shares = {}
        for effect, effect_results in self.effect_results.items():
            if effect == 'Penalty':
                continue
            factors = {key[0]: value for key, value in self.shares_between_effects_invest.items() if key[1] == effect}
            for origin, factor in factors.items():
                additional_shares[effect] = (
                        additional_shares.get(effect, 0) +
                        self.effect_results[origin].all_results['invest']['invest_per_period'] * factor
                )
        for effect, value in additional_shares.items():
            self.effect_results[effect].all_results['invest']['Shares_per_period'][f'From other effects'] = value
            self.effect_results[effect].all_results['invest']['invest_per_period'] += value

    def _add_group_mapping(self) -> Dict:
        mapping = {}
        for flow_label, flow_results in self.flow_results().items():
            flow_meta_data = flow_results.all_infos.get('meta_data', {})
            if 'Gruppe' in flow_meta_data:
                mapping[flow_label] = flow_meta_data['Gruppe']
        for comp_label, comp_results in self.component_results.items():
            comp_meta_data = comp_results.all_infos.get('meta_data', {})
            if 'Gruppe' in comp_meta_data:
                mapping[comp_label] = comp_meta_data['Gruppe']
        return mapping

    def to_dataFrame(self, busOrComp: str, direction: Literal["in", "out", "inout"],
                     grouped: bool=False, invert_Output: bool = True) -> pd.DataFrame:
        '''
        This Function returns a pd.dataframe containing the Flows of the Bus or Comp.

        Parameters
        ----------
        busOrComp : str
            flows linked to this bus or component are chosen
        direction : str ("in","out","inout")
            Direction of the flows to look at. Choose one of "in","out","inout"
        grouped: bool
            wether the inputs and outputs should be grouped. Inputs abd Outputs are still seperate groups.
        invert_Output : bool
            Wether the output flows should be inverted or not (multiplied by -1)

        Returns
        ---------
        pd.DataFrame
        '''
        output_factor = 1 if direction in ("out", "inout") else 0
        if invert_Output:
            output_factor *= -1
        df = self.to_dataframe(busOrComp, 'flow_rate',
                               input_factor=1 if direction in ("in", "inout") else 0,
                               output_factor=output_factor)

        if grouped:
            df = self.group_df_by_mapping(df)

        return df

    def group_df_by_mapping(self, df: pd.DataFrame, custom_mapping: dict = None) -> pd.DataFrame:
        '''
        Groups the columns of a Dataframe based on a Mapping.
        The mapping is the group mapping of the calculation, but a custom mapping can be used
        Includes unmapped Columns
        :param df:
        :param custom_mapping:
        :return:
        '''
        if custom_mapping:
            mapping = custom_mapping
        else:
            mapping = self.group_map

        ungrouped_columns = [col for col in df.columns if mapping.get(col) is None]
        grouped_df = df.T.groupby(mapping).sum().T
        return pd.concat([grouped_df, df[ungrouped_columns]], axis=1)

    def get_sizes_per_period(self, grouped: bool = False) -> pd.DataFrame:
        size_per_period = {}
        for label, size in self.sizes().items():
            if label in self.component_results:
                start_year = self.component_results[label].all_infos['meta_data']['Startjahr']
                lifetime = self.component_results[label].all_infos['meta_data']['Lebensdauer']
            else:
                flows = self.flow_results()
                if label in flows:
                    start_year = flows[label].all_infos['meta_data']['Startjahr']
                    lifetime = flows[label].all_infos['meta_data']['Lebensdauer']
                else:
                    logger.critical(f'{label=} was not found in Components or Flows!')
                    start_year, lifetime = 0, 10000  # TODO: This is not optimal

            availlability = np.array([1 if start_year <= year < (start_year + lifetime) else 0 for year in self.years])
            size_per_period[label] = size * availlability
        df = pd.DataFrame(size_per_period, index=self.years)
        if grouped:
            df = self.group_df_by_mapping(df)
        return df

    def get_availlability(self, grouped: bool = False) -> pd.DataFrame:
        sizes = self.sizes()
        availlability = {
            flow_label: flow_results.all_infos['relative_maximum'] * flow_results.all_results['Investment']['size']
            for flow_label, flow_results in self.flow_results().items() if flow_label in sizes
        }
        for component_label, component_results in self.component_results.items():
            if 'Storage' in component_results.all_infos['class']:
                if component_label in sizes:
                    availlability[component_label] = component_results.all_infos['relative_maximum_charge_state'] * component_results.all_results['Investment']['size']
                    if len(availlability[component_label]) == len(self.time_with_end):
                        availlability[component_label] = availlability[component_label][:-1]

        df = pd.DataFrame(availlability, index=self.time)
        if grouped:
            return self.group_df_by_mapping(df)
        else:
            return df

    def get_sources_and_sinks(self, sinks=True, sources=True, source_and_sinks=True) -> pd.DataFrame:
        '''
        this returns all the flows of the sources and sinks.
        :return: pd.DataFrame
        '''
        data = {}

        if sinks:
            list_of_sinks = [comp for comp in self.component_results.values() if 'Sink' in comp.all_infos['class'].split(':')]
            for sink in list_of_sinks:
                data[sink.label] = sink.to_dataframe('flow_rate').values.flatten()
        if sources:
            list_of_sources = [comp for comp in self.component_results.values() if 'Source' in comp.all_infos['class'].split(':')]
            for source in list_of_sources:
                data[source.label] = source.to_dataframe('flow_rate').values.flatten()
        if source_and_sinks:
            list_of_source_and_sinks = [comp for comp in self.component_results.values() if 'SourceAndSink' in comp.all_infos['class'].split(':')]
            for source_and_sink in list_of_source_and_sinks:
                for flow in source_and_sink.inputs + source_and_sink.outputs:
                    data[flow.label] = flow.to_dataframe('flow_rate').values.flatten()

        return pd.DataFrame(data, index=self.time)

    def get_fuel_costs(self) -> pd.DataFrame:
        '''
        Returns the costs per flow hour of every medium in a DataFrame. Data saved in a special component ("HelperPreise").

        Parameters
        ----------
        calc : flixPostXL
            Solved calculation of type flixPostXL.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the costs per flow hour for each medium. Columns represent different media,
            and rows represent the time series.
        '''
        flows = self.component_results["HelperPreise"].outputs
        result_dataframe = pd.DataFrame(index=self.time)
        for flow in flows:
            name = flow.label_full.split("_")[-1]
            ar = flow.all_infos['effects_per_flow_hour']
            if isinstance(ar, (float, int)):
                ar = ar * np.ones(len(self.time))

            new_dataframe = pd.DataFrame({name: ar}, index=self.time)
            result_dataframe = pd.concat([result_dataframe, new_dataframe], axis=1)

        return result_dataframe.head(len(self.time))

    def plotOperationColorMap(
            self,
            flow_name: str,
            nbPeriods=365,
            nbTimeStepsPerPeriod=24,
            cmap="jet",
            vmin=0,
            vmax=None,
            xlabel="period",
            ylabel="timestep per period",
            zlabel=None,
            figsize=(12, 4),
            fontsize=12,
            save_as=None,
            xticks=None,
            yticks=None,
            xticklabels=None,
            yticklabels=None,
            monthlabels=False,
            dpi=200,
            pad=0.01,
            aspect=15,
            fraction=0.2,
            orientation="vertical",
            fig=None,
            ax=None,
            **kwargs,
    ):
        """
        Plot operation time series of a component at a location.

        **Required arguments:**

        :param esM: considered energy system model
        :type esM: EnergySystemModel class instance

        :param compName: component name
        :type compName: string

        **Default arguments:**

        :param nbPeriods: number of periods to be plotted
            |br| * the default value is 365
        :type nbPeriods: integer

        :param nbTimeStepsPerPeriod: time steps per period to be plotted (nbPeriods*nbTimeStepsPerPeriod=length of time
            series)
            |br| * the default value is 24
        :type nbTimeStepsPerPeriod: integer

        :param cmap: heat map (color map) (see matplotlib options)
            |br| * the default value is 'jet'
        :type cmap: string

        :param vmin: minimum value in heat map
            |br| * the default value is 0
        :type vmin: integer

        :param vmax: maximum value in heat map. If None, vmax is set to the maximum value of the operation time series.
            |br| * the default value is None
        :type vmax: integer

        :param xlabel: x-label of the plot
            |br| * the default value is 'day'
        :type xlabel: string

        :param ylabel: y-label of the plot
            |br| * the default value is 'hour'
        :type ylabel: string

        :param zlabel: z-label of the plot
            |br| * the default value is 'operation'
        :type zlabel: string

        :param figsize: figure size in inches
            |br| * the default value is (12,4)
        :type figsize: tuple of positive floats

        :param fontsize: font size of the axis
            |br| * the default value is 12
        :type fontsize: positive float

        :param save: indicates if figure should be saved
            |br| * the default value is False
        :type save: boolean

        :param fileName: output file name
            |br| * the default value is 'operation.png'
        :type fileName: string

        :param xticks: user specified ticks of the x axis
            |br| * the default value is None
        :type xticks: list

        :param yticks: user specified ticks of the ý axis
            |br| * the default value is None
        :type yticks: list

        :param xticklabels: user specified tick labels of the x axis
            |br| * the default value is None
        :type xticklabels: list

        :param yticklabels: user specified tick labels of the ý axis
            |br| * the default value is None
        :type yticklabels: list

        :param monthlabels: specifies if month labels are to be plotted (only works correctly if
            365 days are specified as the number of periods)
            |br| * the default value is False
        :type monthlabels: boolean

        :param dpi: resolution in dots per inch
            |br| * the default value is 200
        :type dpi: scalar > 0

        :param pad: pad parameter of colorbar
            |br| * the default value is 0.12
        :type pad: float

        :param aspect: aspect parameter of colorbar
            |br| * the default value is 15
        :type aspect: float

        :param fraction: fraction parameter of colorbar
            |br| * the default value is 0.2
        :type fraction: float

        :param orientation: orientation parameter of colorbar
            |br| * the default value is 'horizontal'
        :type orientation: float

        """
        isStorage = False
        try:
            flow = self.flow_results()[flow_name]
        except KeyError:
            raise KeyError(f'The Flow with the label {flow_name} was not found. '
                           f'Choose from {self.flow_results().keys()}')

        data = flow.variables["flow_rate"]
        unit = "Flow Hours"

        try:
            data = data.reshape(nbPeriods, nbTimeStepsPerPeriod).T
        except ValueError as e:
            raise ValueError(
                "Could not reshape array. Your timeSeries has {} values and it is therefore not possible".format(
                    len(data)
                )
                + " to reshape it to ({}, {}). Please correctly specify nbPeriods".format(
                    nbPeriods, nbTimeStepsPerPeriod
                )
                + " and nbTimeStepsPerPeriod The error was: {}.".format(e)
            )
        vmax = data.max() if not vmax else vmax

        if not fig or not ax:
            fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

        ax.pcolormesh(
            range(nbPeriods + 1),
            range(nbTimeStepsPerPeriod + 1),
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        ax.axis([0, nbPeriods, 0, nbTimeStepsPerPeriod])
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.xaxis.set_label_position("bottom"), ax.xaxis.set_ticks_position("bottom")

        sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm1._A = []
        cb1 = fig.colorbar(
            sm1, ax=ax, pad=pad, aspect=aspect, fraction=fraction, orientation=orientation
        )
        cb1.ax.tick_params(labelsize=fontsize)
        if not zlabel:
            cb1.ax.set_xlabel(zlabel, size=fontsize)
        elif isStorage:
            cb1.ax.set_xlabel("Storage inventory" + " [" + unit + "]", size=fontsize)
        else:
            cb1.ax.set_xlabel("Operation" + " [" + unit + "]", size=fontsize)
        cb1.ax.xaxis.set_label_position("top")

        if xticks:
            ax.set_xticks(xticks)
        if yticks:
            ax.set_yticks(yticks)
        if xticklabels:
            ax.set_xticklabels(xticklabels, fontsize=fontsize)
        if yticklabels:
            ax.set_yticklabels(yticklabels, fontsize=fontsize)

        if monthlabels:
            import datetime

            xticks, xlabels = [], []
            for i in range(1, 13, 2):
                xlabels.append(datetime.date(2050, i + 1, 1).strftime("%b"))
                xticks.append(datetime.datetime(2050, i + 1, 1).timetuple().tm_yday)
                ax.set_xticks(xticks), ax.set_xticklabels(xlabels, fontsize=fontsize)

        fig.tight_layout()

        if save_as:
            plt.savefig(save_as, dpi=dpi, bbox_inches="tight")

        return fig, ax


def compute_conversion_factors(conversion_dict: Dict[str, Dict[str, Union[int, float, np.ndarray]]]
                               ) -> Dict[Tuple[str, str], Union[int, float, np.ndarray]]:
    """
    This function takes a dictionary with conversion factors and computes all indirect conversion factors between nodes.
    """

    # Initialize the result dictionary with the given conversion factors
    result = {}

    # Iterate over each domain in the dictionary
    for origin in conversion_dict:
        # Initialize a queue for Breadth-First Search (BFS)
        queue = [(origin, 1)]
        visited = set()

        # Perform BFS to find all indirect conversion factors
        while queue:
            current_domain, factor = queue.pop(0)
            if current_domain in visited:
                continue
            visited.add(current_domain)

            # Iterate over the neighbors of the current domain
            for target, conversion_factor in conversion_dict.get(current_domain, {}).items():
                if target not in visited:
                    # Calculate the indirect conversion factor
                    indirect_factor = factor * conversion_factor
                    # Update the result dictionary with the indirect conversion factor
                    if (origin, target) not in result:
                        result[(origin, target)] = indirect_factor
                    queue.append((target, indirect_factor))

    return result
