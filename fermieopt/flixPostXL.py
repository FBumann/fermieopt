# -*- coding: utf-8 -*-
import os
from typing import Literal, Optional, Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

from flixOpt.flixPostprocessing import flix_results
from flixOpt.flixStructure import cEnergySystem, cBaseComponent


class flixPostXL(flix_results):
    def __init__(self, nameOfCalc, results_folder, outputYears):
        super().__init__(nameOfCalc=nameOfCalc, results_folder=results_folder, comp_colors = None)

        self._add_InvestTS_to_results()
        self._add_group_mapping()
        self._add_color_mapping()
        self.years = outputYears # add as attribute
        self.folder = os.path.dirname(results_folder)


        # delete duplicate information for save access of the new data
        del self.results_struct
        effects = [fx for fx in self.results["globalComp"]]
        while "penalty" in effects:
            effects.remove("penalty")
        for key in effects:
            self.results.pop(key, None)

    def _add_InvestTS_to_results(self):
        '''
        Add all the investments to the results of calc1 by creating a Timeseries from the total value for each Investment
        '''
        #TODO: This does not work when specific shares is different for operation and investment. This information is n ot availlable.

        exists_coll = self.get_exist_values(normalize=True)
        # <editor-fold desc="Allocate investments over time to each effect">
        for effect_name, effect_results in self.results["globalComp"].items():
            if "invest" not in effect_results:
                continue
            invest_shares = effect_results["invest"]["shares"]
            summed_invest_shares = {}

            # Iterate over items in the original dictionary
            for key, value in invest_shares.items():
                if "specificShareToOtherEffects" in key: continue
                # Extract the common key prefix until the last "_"
                common_key = key.rsplit('_', 1)[0]

                # Sum up values with the same common key or set the value if it's the first occurrence
                summed_invest_shares[common_key] = summed_invest_shares.get(common_key, 0) + value

            invest_TS = {key: summed_invest_shares[key] * exists_coll[key]
                         for key in summed_invest_shares.keys() & exists_coll.keys()}
            sum_TS = np.sum(list(invest_TS.values()),axis=0)

            self.results["globalComp"][effect_name]["invest"]["shares_TS"] = invest_TS # shares as TS
            self.results["globalComp"][effect_name]["invest"]["sum_TS"] = sum_TS  # sum as TS
        # </editor-fold>

        # <editor-fold desc="Update shares_TS with shares by other effects">
        for _ in range(20): # iterative process, spreads like a tree.
            for effect_name, effect_results in self.results["globalComp"].items():
                shares = {key: effect_results[key] for key in effect_results.keys() if "specificShareToOtherEffect" in key}
                # TODO: also specificShareToOtherEffect_invest ??
                if not shares: # if there are no shares
                    continue
                if "invest" not in effect_results:
                    continue

                for key, val in shares.items():
                    effect_origin = effect_name
                    effect_target = key.rsplit('_', 1)[-1]
                    factor = val
                    new_key = f"{effect_origin}_specificShareToOtherEffects_invest"
                    shared_TS = self.results["globalComp"][effect_origin]["invest"]["sum_TS"] * factor
                    if "shares_TS" not in self.results["globalComp"][effect_target]["invest"]:
                        self.results["globalComp"][effect_target]["invest"]["shares_TS"] = {}
                    self.results["globalComp"][effect_target]["invest"]["shares_TS"][new_key] = shared_TS

            # <editor-fold desc="sum up shares_TS to sum_TS">
            for effect_name, effect_results in self.results["globalComp"].items():
                if "invest" not in effect_results:
                    continue
                shared_TS = self.results["globalComp"][effect_name]["invest"]["shares_TS"]
                sum_TS = np.zeros(len(self.timeSeries))
                for array in shared_TS.values():
                    sum_TS += array

                self.results["globalComp"][effect_name]["invest"]["sum_TS"] = sum_TS

        # </editor-fold>

        for effect_name, effect_results in self.results["globalComp"].items():
            if "invest" in effect_results or "operation" in effect_results:

                if "invest" in effect_results:
                    invest_TS = effect_results["invest"]["sum_TS"]
                if "operation" in effect_results:
                    operation_TS = effect_results["operation"]["sum_TS"]

                effect_results["all"]["sum_TS"] = invest_TS + operation_TS


        # Check if TS of invest matches the total costs of invest
        for effect_name, effect_results in self.results["globalComp"].items():
            if "invest" not in effect_results:
                continue
            # <editor-fold desc="check if everything went well">
            # Check if something is missing
            target_value = abs(effect_results["invest"]["sum"])
            calculated_value = abs(effect_results["invest"]["sum_TS"].sum())
            missing_value = abs(calculated_value - target_value)
            if target_value ==0:
                if abs(missing_value) >= 0.01:
                    raise Exception(f"Investments missing in {effect_name}. Missing Value is: {missing_value}")
            else:
                if missing_value / target_value >= 0.00001:  # 0.001% (numeric error)
                    raise Exception(f"Investments missing in {effect_name}. Missing Value is: "
                                    f"{missing_value} [{np.round(missing_value / target_value * 100, decimals=2)} %]")

        # Check if TS of all matches the total costs of all
        for effect_name, effect_results in self.results["globalComp"].items():
            if "invest" not in effect_results:
                continue
            # <editor-fold desc="check if everything went well">
            # Check if something is missing
            target_value = abs(effect_results["all"]["sum"])
            calculated_value = abs(effect_results["all"]["sum_TS"].sum())
            missing_value = abs(calculated_value - target_value)
            if -1e-6 <= target_value <= 1e-6:
                if abs(missing_value) >= 0.01:
                    raise Exception(f"Investments missing in {effect_name}. Missing Value is: {missing_value}")
            else:
                if missing_value / target_value >= 0.00001:  # 0.001% (numeric error)
                    raise Exception(f"Investments missing in {effect_name}. Missing Value is: "
                                    f"{missing_value} [{np.round(missing_value / target_value * 100, decimals=2)} %]")
            # </editor-fold>

        print ("Aggregated all Investments to their effect based on their existance sucessfully")

    def _add_group_mapping(self):
        mapping = {}
        for flow in self.flows:
            if hasattr(flow, "group"):
                mapping[flow.label_full] = flow.group
        for comp in self.comp_posts:
            if hasattr(comp, "group"):
                mapping[comp.label] = comp.group
        self.group_map = mapping

    def _add_color_mapping(self):
        # standard color set:
        n_colors = len(self.group_map.keys())
        colors_basic = px.colors.sample_colorscale("viridis", [n / (n_colors - 1) for n in range(n_colors)])
        color_generator = cycle_values(colors_basic)
        mapping = {}
        for flow in self.flows:
            mapping[flow.label_full] = flow.color
        for comp in self.comp_posts:
            mapping[comp.label] = comp.color

        groups = list(set(self.group_map.values()))
        for group in groups:
            mapping[group] = next(color_generator)

        self.color_map = mapping



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
        if direction not in ("in", "out", "inout"):
            raise TypeError(direction + ' is no valid arg for "direction" ')

        df_in = pd.DataFrame(index=self.timeSeries)
        df_out = pd.DataFrame(index=self.timeSeries)

        (in_flows, out_flows) = self.getFlowsOf(busOrComp)

        if direction in ("in", "inout"):
            for flow in in_flows:
                label = flow.label_full
                df_in[label] = flow.results['val']

        if direction in ("out", "inout"):
            for flow in out_flows:
                label = flow.label_full
                df_out[label] = flow.results['val']

            if invert_Output:
                df_out = -1 * df_out

        df = pd.concat([df_in, df_out], axis=1)

        if grouped:
            df = self.group_df_by_mapping(df)

        return df

    def group_df_by_mapping(self, df:pd.DataFrame, custom_mapping: dict = None) -> pd.DataFrame:
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

    def get_exist_values(self, normalize = False) -> Dict[str, np.ndarray]:
        '''
        Returns all exists_values for all comps in a dict with the name  of the comp as the key.
        When a Comp has no exists_values for, a default is assigned (all ones)
        -----
        :param normalize : bool
            If True, normalize values to a total of 1. This is usefull for Allocation of Sums (of invest f.Ex)
        :return: dict
        '''
        exists_coll = {}
        for flow in self.flows:
            flow_results = flow.results
            exists = flow_results["exists_with_comp"]
            if not isinstance(exists, np.ndarray):
                exists = exists * np.ones(len(self.timeSeries))
            exists_coll[flow.label_full] = exists

        for comp in self.comp_posts:
            exists = self.results[comp.label]["exists"]
            if not isinstance(exists, np.ndarray):
                exists = exists * np.ones(len(self.timeSeries))

            if len(exists) != len(self.timeSeries):
                if comp.type == "cStorage":
                    exists = exists[: -1] #strip the last part because of storage has one TS more
                else:
                    raise ValueError(f"Unexpected length of exists for {comp.label}")

            exists_coll[comp.label] = exists

        if normalize:
            # Divide each element in the array by its sum using element-wise division
            normalized_data = {}
            for key, array in exists_coll.items():
                if np.max(array) == 0:
                    normalized_data[key] = array
                else:
                    normalized_data[key] = array / np.sum(array)
            exists_coll = normalized_data

        return exists_coll
    
    def get_invest_results(self, flows:bool, storages:bool, actual_storage_capacity=False) -> dict:
        '''
        This function returns the investment results as numbers stored in a dict
        :param flows: include flows
        :param storages: include storages
        :param actual_storage_capacity: Wether to use the actual storage capacity (multiply ba max_rel_capacity)
        :return: dict of nominal values or capacity in flow hours chosen with the comp/flow as the key
        '''


        invest_all = {}
        # for flows
        if flows:
            for flow in self.flows:
                invest_data = flow.results.get("invest")
                if invest_data is None:
                    continue
                used_nominal_val = invest_data.get("nominal_val")

                if flow.isInputInComp:  # ignoring all Investments of input flows to comps, (f.Ex. Storage Loading)
                    continue

                # add to DataFrame
                if flow.label_full in invest_all.keys():
                    raise Exception(f"{flow.label_full} already in 'invest_all' dictionary")
                else:
                    invest_all[flow.label_full] = used_nominal_val

        # For Storages
        if storages:
            for comp_label in self.infos_system["components"].keys():
                comp = self.results[comp_label]
                invest_data = comp.get("invest", None)
                if invest_data is None: continue
                used_capacity = invest_data.get("used_capacity_inFlowHours", None)
                if used_capacity is None: continue

                if actual_storage_capacity:
                    if isinstance(comp["max_rel_chargeState"], np.ndarray):
                        max_rel_chargeState = comp["max_rel_chargeState"][:-1]
                    else:
                        max_rel_chargeState = comp["max_rel_chargeState"]
                    used_capacity = used_capacity * max_rel_chargeState

                if comp_label in invest_all: raise Exception(f"{comp_label} already in 'invest_all' dictionary")
                invest_all[comp_label] = used_capacity

        return invest_all

    def get_invest_results_as_TS(self, flows:bool, storages:bool, grouped:bool=False, actual_storage_capacity= False) -> pd.DataFrame:
        '''
        This function returns the investment results as a Dataframe based on the existance of the comp
        :param flows: include flows
        :param storages: include storages
        :param actual_storage_capacity: Wether to use the actual storage capacity (multiply ba max_rel_capacity)
        :return: dict of nominal values or capacity in flow hours chosen with the comp/flow as the key
        '''
        # caluculation of the data
        val = self.get_invest_results(flows=flows, storages = storages, actual_storage_capacity=actual_storage_capacity)
        ex = self.get_exist_values(normalize = False)

        product_dict = {}
        # Iterate through keys that are common to both dictionaries
        for key in set(val.keys()) & set(ex.keys()):
            product_dict[key] = val[key] * ex[key]
        df = pd.DataFrame(product_dict)

        if grouped:
            df = self.group_df_by_mapping(df)
        return df


    def get_effect_results(self, effect_name: str, origin: Literal["operation", "invest", "all"],
                          as_TS: bool = False, shares: bool = False) -> np.ndarray:
        '''
        This functions returns the results of the chosen effect
        :param effect_name: Name of effect
        :param origin: Choose from ["operation","invest","all"]
        :param as_TS: Wether to return the values per timestep or a sum
        :param shares: Wether to return the shares to the effect
        :return: np.ndarray
        '''
        if origin not in ("operation", "invest", "all"):
            raise ValueError(f"Invalid parameter: {origin}")

        key_pre = "shares" if shares else "sum"
        key_post = "_TS" if as_TS else ""
        key = key_pre + key_post

        return self.results["globalComp"][effect_name][origin][key]

    def get_effect_result_of_comp_without_shares(self,
                                         effect: str,
                                         comp_name: str,
                                         operation_or_invest: Literal["operation", "invest"]
                                         ) -> float:
        """
        Calculates the total effect result for a specified component without accounting for shared influences from other effects.
        The resulting value is the sum of the component itself and all sub COmponents like FLows

        Args:
            effect (str): The type of effect to be calculated (e.g., costs, CO2, ...).
            comp_name (str): The name of the component.
            operation_or_invest (Literal["operation", "invest"]): Specifies whether to use operational or investment values.

        Returns:
            float: The calculated result without considering any shared effects.
        """
        result = 0
        for name, value in self.results["globalComp"][effect][operation_or_invest]["shares"].items():
            if name.startswith(f"{comp_name}__"):  # string comparison. Flow name is split from comp name by __
                result += value
        return result

    def get_sources_and_sinks(self, sinks=True, sources = True, sinks_n_sources=True)->pd.DataFrame:
        '''
        this returns all the flows of the sources and sinks.
        :return: pd.DataFrame
        '''
        df = pd.DataFrame()

        if sinks:
            list_of_sinks = [comp.label for comp in self.comp_posts if comp.type == "cSink"]
            for sink in list_of_sinks:
                df[sink] = self.to_dataFrame(sink, grouped=False, direction="in", invert_Output=False) * -1
        if sources:
            list_of_sources = [comp.label for comp in self.comp_posts if comp.type == "cSource"]
            for source in list_of_sources:
                df[source] = self.to_dataFrame(source, grouped=False, direction="out", invert_Output=False)
        if sinks_n_sources:
            list_of_sinks_n_sources = [comp.label for comp in self.comp_posts if comp.type == "cSourceAndSink"]
            for sink_n_source in list_of_sinks_n_sources:
                df[sink_n_source] = self.to_dataFrame(sink_n_source, grouped=False, direction="inout",
                                                      invert_Output=True) * -1


        return df


    def getFuelCosts(self) -> pd.DataFrame:
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
        (discard, flows) = self.getFlowsOf("HelperPreise")
        result_dataframe = pd.DataFrame(index=self.timeSeries)
        for flow in flows:
            name = flow.label_full.split("_")[-1]
            ar = flow.results["costsPerFlowHour_standard"]
            if isinstance(ar, (float, int)):
                ar = ar * np.ones(len(self.timeSeries))

            new_dataframe = pd.DataFrame({name: ar}, index=self.timeSeries)
            result_dataframe = pd.concat([result_dataframe, new_dataframe], axis=1)

        return result_dataframe.head(len(self.timeSeries))

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
        flow = [flow for flow in self.flows if flow.label_full == flow_name]

        timeSeries = flow[0].results["val"]
        print(timeSeries)
        unit = "Flow Hours"

        try:
            timeSeries = timeSeries.reshape(nbPeriods, nbTimeStepsPerPeriod).T
        except ValueError as e:
            raise ValueError(
                "Could not reshape array. Your timeSeries has {} values and it is therefore not possible".format(
                    len(timeSeries)
                )
                + " to reshape it to ({}, {}). Please correctly specify nbPeriods".format(
                    nbPeriods, nbTimeStepsPerPeriod
                )
                + " and nbTimeStepsPerPeriod The error was: {}.".format(e)
            )
        vmax = timeSeries.max() if not vmax else vmax

        if not fig or not ax:
            fig, ax = plt.subplots(1, 1, figsize=figsize, **kwargs)

        ax.pcolormesh(
            range(nbPeriods + 1),
            range(nbTimeStepsPerPeriod + 1),
            timeSeries,
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

    def visual_representation(self, save_to: Optional[str] = None) -> go.Figure:
        figure = SystemVisualization(self).create_figure()
        if save_to:
            figure.write_html(save_to)
        return figure



class SystemInfos:
    '''
    Class to extract the data from a flix_results in preparation for the Model visualization
    '''

    def __init__(self,
                 calc: flix_results=None,
                 es:cEnergySystem=None):
        '''
        Extract the necessary data from a flix_results object
        :param calc: flix_results Object
        :param es: cEnergySystem
        :param custom_colors: pass a color mapping. Else, it will assign colors automatically
        '''
        self.comp_info = {}
        self.bus_info = {}
        self.color_map = {}

        if calc:
            self._extract_data_from_calc(calc)
        elif es:
            self._extract_data_from_es(es)
        else:
            raise ValueError("Either 'calc' or 'es' must be passed")

        self._reorder_to_bus()

    def _extract_data_from_calc(self, calc: flix_results):
        '''
        extract data from flixResults
        :param: calculation of type flixResults
        :return:
        '''
        comp_data = calc.infos_system["components"]
        for i, (comp_name, item) in enumerate(comp_data.items()):
            inputs = []
            outputs = []
            for flow_dict in item["In-Flows"]:  # list
                # Get the first key-value pair
                first_key = next(iter(flow_dict))
                bus_label = flow_dict[first_key]["bus"]
                inputs.append(bus_label)
            for flow_dict in item["Out-Flows"]:  # list
                first_key = next(iter(flow_dict))
                bus_label = flow_dict[first_key]["bus"]
                outputs.append(bus_label)

            self.comp_info[comp_name] = {}
            self.comp_info[comp_name]["inputs"] = inputs.copy()
            self.comp_info[comp_name]["outputs"] = outputs.copy()

    def _extract_data_from_es(self, es: cEnergySystem):
        '''
        extract data from flixResults
        :param: calculation of type flixResults
        :return:
        '''
        comp_data:list
        comp_data = es.listOfComponents
        for i, comp in enumerate(comp_data):
            comp: cBaseComponent
            comp_name = comp.label
            inputs = []
            outputs = []
            for flow in comp.inputs:
                inputs.append(flow.bus.label)
            for flow in comp.outputs:
                outputs.append(flow.bus.label)

            self.comp_info[comp_name] = {}
            self.comp_info[comp_name]["inputs"] = inputs.copy()
            self.comp_info[comp_name]["outputs"] = outputs.copy()

        # colors of es
        self.color_map = {comp: comp.label for comp in comp_data if hasattr(comp, "color")}

    def _reorder_to_bus(self):
        for comp in self.comp_info:
            for bus in self.comp_info[comp]["inputs"]:
                if bus in self.bus_info:
                    self.bus_info[bus]["outputs"].append(comp)
                else:
                    self.bus_info[bus] = {"inputs": [],
                                          "outputs": [comp]}
            for bus in self.comp_info[comp]["outputs"]:
                if bus in self.bus_info:
                    self.bus_info[bus]["inputs"].append(comp)
                else:
                    self.bus_info[bus] = {"inputs": [comp],
                                          "outputs": []}


class SystemVisualization:
    '''
    Visualizer for a flixOpt Model. Visualizes Comps, Busses and Flow direction.
    Application:
    VisuData = cVisuData(calc1)
    model_visualization = cModelVisualizer(VisuData)
    model_visualization.show()
    '''
    def __init__(self,
                 calc: flix_results = None,
                 es: cEnergySystem = None,
                 comp_colorscale: str = 'algae',
                 bus_colorscale: str = 'jet',
                 use_bus_colors_for_edges: bool = True,
                 custom_color_map: dict = None,
                 figure_scaling: float = "auto"):
        '''

        Parameters
        ----------
        data
        custom_color_map
        comp_colorscale: Uses plotly.express. Supported colorscales: https://plotly.com/python/builtin-colorscales/
        bus_colorscale: Uses plotly.express. Supported colorscales: https://plotly.com/python/builtin-colorscales/
        figure_scaling
        '''
        self.data = SystemInfos(calc, es)
        self.busses: Tuple[str, ...] = tuple(sorted(self.data.bus_info.keys()))  # Sorted collection of busses
        self.comps: Tuple[str, ...] = tuple(sorted(self.data.comp_info.keys()))   # Sorted collection of comps

        # Using the colors from SystemInfos, but overwriting it with the new custom color map
        if not custom_color_map:
            custom_color_map = {}
        custom_color_map = {**self.data.color_map, **custom_color_map}

        bus_colors = create_color_mapping(self.busses, bus_colorscale, custom_color_map)  # Mapping busnames to colors
        comp_colors = create_color_mapping(self.comps, comp_colorscale, custom_color_map)  # Mapping compnames to colors
        self.color_map = {**comp_colors, **bus_colors}

        if figure_scaling == "auto":
            self.figure_scaling = 6 / len(self.data.bus_info.keys())
        elif isinstance(figure_scaling, (int, float)):
            self.figure_scaling = figure_scaling
        else:
            raise TypeError(f"custom_scale has to be 'auto' or 'int' or 'float'")

        self.positions = self._get_positions()
        self.graph = self._create_graph(use_bus_colors_for_edges)

    def create_figure(self,
                      comp_marker: str = 'square-dot',
                      bus_marker: str = 'circle-dot',
                      comp_marker_size=20,
                      bus_marker_size=60,
                      arrow_size=20,
                      edge_width=5,
                      font_size_comps=18) -> go.Figure:
        '''
        Function to visualize the Graph.
        here, the logic for the visualization of each component is written
        :param G:
        :param positions: a dict with a mapping from the comp_name to x and y positions:  {comp_name: [x, y]}. The x and y positions of the objects used relative to each other
        :return:
        '''
        # Create a Plotly graph
        fig = go.Figure()

        # Add edges with arrows.
        # Start and end point is defined by data in positions, key is name of Comp/bus
        for edge in self.graph.edges(data=True):
            x0, y0 = self.positions[edge[0]]
            x1, y1 = self.positions[edge[1]]

            fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1],
                                     marker=dict(size=arrow_size * self.figure_scaling,
                                                 symbol="arrow-bar-up", angleref="previous",
                                                 line=dict(width=1 * self.figure_scaling, color='black')),
                                     mode="lines+markers+text",
                                     line=dict(width=edge_width * self.figure_scaling, color=edge[2]["color"]),
                                     textposition="top center",
                                     hoverinfo="none",
                                     opacity=0.7
                                     )
                          )

        # Add nodes
        for node, (x, y) in self.positions.items():  # the node is a str
            if self.graph.nodes[node]["type"] == "component":
                marker_symbol = comp_marker
                marker_size = comp_marker_size * self.figure_scaling  # Adjust the size as needed
                textsize = font_size_comps * self.figure_scaling
                textposition = 'top center'
                opacity = 0.7
            else:
                marker_symbol = bus_marker
                marker_size = bus_marker_size * self.figure_scaling  # Adjust the size as needed
                textsize = font_size_comps * 1.2 * self.figure_scaling
                textposition = 'bottom center'
                opacity = 0.5
            label = node
            color = self.graph.nodes[node]["color"]
            fig.add_trace(go.Scatter(x=[x], y=[y],
                                     marker=dict(size=marker_size, symbol=marker_symbol, color=color,
                                                 line=dict(width=2, color='black'), opacity=opacity),
                                     mode="markers+text",
                                     text=[label], textfont=dict(size=textsize),
                                     textposition=textposition, hoverinfo=["none"],
                                     )
                          )

        # Set layout options
        fig.update_layout(
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hovermode="closest",
        )
        return fig

    def _create_graph(self, use_bus_colors_for_edges=False) -> nx.DiGraph:
        '''
        Creates the graph object for later plotting.
        Colors are given as an attribute to each node an edge
        :return:
        '''
        G = nx.DiGraph()
        for bus in self.busses:
            G.add_node(bus, type="bus", color=self.color_map[bus])
        for comp in self.comps:
            G.add_node(comp, type="component",
                       inputs=self.data.comp_info[comp]["inputs"],
                       outputs=self.data.comp_info[comp]["outputs"],
                       color=self.color_map[comp])

        # Connect components to buses
        for node in list(G.nodes()):
            if G.nodes[node]["type"] == "component":
                for input_bus in G.nodes[node]["inputs"]:
                    if use_bus_colors_for_edges:
                        edge_color = self.color_map[input_bus]
                    else:
                        edge_color = self.color_map[node]
                    G.add_edge(input_bus, node, edge_type="input", color=edge_color)
                for output_bus in G.nodes[node]["outputs"]:
                    if use_bus_colors_for_edges:
                        edge_color = self.color_map[output_bus]
                    else:
                        edge_color = self.color_map[node]
                    G.add_edge(node, output_bus, edge_type="output", color=edge_color)
        return G

    def _get_positions(self) -> Dict[str, Tuple[float, float]]:
        # create a layout with columns
        cols = [[] for _ in range(len(self.busses) * 2 + 1)]

        # get position mappings for each bus
        pos = {}
        cols_bus = [[] for _ in range(len(self.busses) * 2 + 1)]
        for i, bus in enumerate(self.busses):
            # TODO: Order the buses by a logic?
            cols_bus[i * 2 + 1].append(bus)
        for col_nr in range(len(cols_bus)):
            pos_map = self._get_position_with_y_spread(keys=cols_bus[col_nr], x_pos=col_nr, y_pos=-3)
            pos = {**pos, **pos_map}

        # create a mapping to calculate the position of each component, based on its output and input bus
        bus_map = {key: i * 2 + 1 for i, key in enumerate(self.busses)}

        # calculate column for each component
        for comp_name in self.comps:
            comp = self.data.comp_info[comp_name]
            output_val = bus_map[comp["outputs"][0]] if len(comp["outputs"]) > 0 else None
            input_val = bus_map[comp["inputs"][0]] if len(comp["inputs"]) > 0 else None
            if output_val is None and input_val is None:
                raise Exception("Comp must have one input or output")
            if output_val is None:
                col_nr = input_val + 1
            elif input_val is None:
                col_nr = output_val - 1
            else:
                col_nr = int((input_val + output_val) / 2)
            cols[col_nr].append(comp_name)

        # get position mappings for each column for the comps
        for col_nr in range(len(cols)):
            pos_map = self._get_position_with_y_spread(keys=cols[col_nr], x_pos=col_nr, y_pos=0, y_spread_ampl=2)
            pos = {**pos, **pos_map}

        return pos

    def _get_position_with_y_spread(self,
                                    keys: List[str],
                                    x_pos: float,
                                    y_pos: float,
                                    y_spread_ampl: float = 0) -> Dict[str, Tuple[float, float]]:
        '''
        Takes a dict, whose values are a list of 2 numbers
        used to spread multiple comps in the same column across the yaxis
        :param keys: labels, new keys of the returned dict
        :param x_pos: x position
        :param y_pos: y position
        :param y_spread_ampl: maximal amplitude of spread.
        :return: dict with keys and updated values
        '''
        if not isinstance(keys, list):
            keys = [keys]
        keys = sorted(keys)
        if len(keys) == 1:
            y_pos = np.array([y_pos])
        else:
            y_pos = y_pos + np.linspace(y_spread_ampl, -1*y_spread_ampl, len(keys))

        y_pos = y_pos * self.figure_scaling

        # Assign the new values to the second elements in each list
        dict = {key: (x_pos, value) for key, value in zip(keys, y_pos)}

        return dict

def cycle_values(values):
    index = 0
    while True:
        yield values[index]
        index = (index + 1) % len(values)

def create_color_mapping(values: Union[List[str], Tuple[str, ...]], colorscale: str = 'viridis', custom_color_map=None) -> dict:
    '''
    This Function creates a color mapping for a Collection of values based on a color scale.
    A custom color mapping can be used, which overwrites colors from the colorscale.

    Parameters
    ----------
    keys
    colorscale
    custom_color_map

    Returns
    -------

    '''
    from plotly.express.colors import sample_colorscale
    n_colors = len(values)
    colors_basic = sample_colorscale(colorscale, [n / (n_colors - 1) for n in range(n_colors)])
    color_generator = cycle_values(colors_basic)
    mapping = {}
    for key in values:
        if custom_color_map and values in custom_color_map:
            mapping[key] = custom_color_map[key]
        else:  # fill up with basic colors
            mapping[key] = next(color_generator)
    return mapping