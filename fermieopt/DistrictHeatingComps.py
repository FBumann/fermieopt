# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import textwrap
from typing import Union, List, Dict, Optional, Literal, Any, Tuple
from flixOpt.flixComps import (cKessel, cInvestArgs, cHeatPump, cStorage, cKWK, cAbwaermeHP, cBaseLinearTransformer,
                               cEHK, cCoolingTower, cSink, cSource)
from flixOpt.flixCompsExperimental import KWKektB
from flixOpt.flixStructure import cME, cFlow, cEffectType, cBus

NO_DEFAULT = object()  # Unique object as signal that no default value exists


class Validator:
    """
    This class is Used to provide a general structure and functionality to reliably create Objects from kwargs.
    It provides type validation and properties to distinguish between optional and mandatory kwargs

    Attributes:
        props (dict): A dictionary holding the properties of the object, initialized from keyword arguments.
        kwargs (dict): Additional keyword arguments not directly assigned to properties.
        computed_props (dict): Computed properties derived from the object's attributes.
        flix_comps (List[cME]): A list of components associated with the object, where cME is a custom type
                                 representing a component in the energy system.

    Methods:
        __init__(**props): Initializes an instance of Validator with properties passed as keyword arguments.
        __str__(): Provides a string representation of the Validator instance.
        _property_defaults(): Returns a dictionary of all properties which have a default value.
        _property_types(): Returns a dictionary mapping property names to their allowed types.
        _mandatory_properties(): Identifies and returns a list of required props for the Validator.
        _kwargs(): Extracts and validates additional keyword arguments not directly assigned to properties.
        validate_properties(): Validates the properties against the definitions and types specified in _property_definitions and _allowed_kwargs.
        setup_default_properties(): Sets up default values for properties that have them defined in _property_definitions.
        finalize_kwargs(time_series_data: pd.DataFrame) -> Dict[str, Any]: Finalizes kwargs and saves them in Dictionaries, Overwriting the placeholder in self.kwargs.
        computation(years_of_model: List[int], co2_factors: Dict[str, float], time_series_data: pd.DataFrame) -> None: Placeholder for computation logic.
        connect_to_system(time_series_data: pd.DataFrame, co2_factors: Dict[str, float], years_of_model: List[int], effects: Dict[str, cEffectType], busses: Dict[str, cBus]) -> List[cME]: Initializes a flixOpt Component from the computed data.
    """

    # Defining allowed properties, default values and allowed types. Needs to be extended by Child class
    _property_definitions = {
    }
    # Defining allowed kwargs and types. Needs to be extended by Child class
    _allowed_kwargs = {
    }

    def __init__(self, **props):
        """
          Initializes an instance of EnergySystemObject with properties passed as keyword arguments.

          Args:
              props (dict): Keyword arguments representing the properties of the object.
          """
        self.props: Dict[str, Any] = props
        self.kwargs = None  #Placeholder and Validation that the function finalize kwargs is called
        self.computed_props = {}
        self.flix_comps: List[cME] = []

        self.setup_default_properties()
        self.validate_properties()

    def __str__(self):
        props_str = f"props=\n{textwrap.indent(print_dict(self.props), ' ' * 3)}"
        computed_props_str = f"computed_props=\n{textwrap.indent(print_dict(self.computed_props), ' ' * 3)}"
        kwargs_str = f"kwargs=\n{textwrap.indent(print_dict(self.kwargs), ' ' * 3)}"

        return (f"<{self.__class__.__name__}> {self.props['Name']}:\n"
                    f"{textwrap.indent(props_str, ' ' * 3)}\n"
                    f"{textwrap.indent(computed_props_str, ' ' * 3)}\n"
                    f"{textwrap.indent(kwargs_str, ' ' * 3)}\n"
                    )

    @property
    def _property_defaults(self) -> Dict[str, Any]:
        """
        Returns a dictionary of all properties which have a default value

        Returns:
            dict: A dictionary mapping property names to their default values.
        """
        return {k: default for k, (default, types) in self._property_definitions.items() if default is not NO_DEFAULT}

    @property
    def _property_types(self) -> Dict[str, Tuple[type, type]]:
        """
        Returns a dictionary mapping property names to their allowed types.

        Returns:
            dict: A dictionary mapping property names to their allowed types.
        """
        return {k: types for k, (defaults, types) in self._property_definitions.items()}

    @property
    def _mandatory_properties(self) -> List[str]:
        """
        Identifies and returns a list of required props for the EnergySystemObject.

        Returns:
            list: A list of properties which are mandatory
        """
        return [prop for prop, (default, prop_type) in self._property_definitions.items()
                if default is NO_DEFAULT]

    @property
    def _kwargs(self) -> dict:
        """
        Extracts and validates additional keyword arguments not directly assigned to properties.

        Returns:
            dict: A dictionary of validated additional keyword arguments.
        """
        kwargs = {}
        for key, allowed_types in self._allowed_kwargs.items():
            if key in self.props:
                kwargs[key] = self.props[key]

        return kwargs

    def validate_properties(self):
        """
        Validates the properties against the definitions and types specified in _property_definitions and _allowed_kwargs.

        Raises:
            ValueError: If a mandatory property is missing or an invalid property is provided.
            TypeError: If a property has an incorrect type.
        """

        # Check for mandatory properties
        for prop in self._mandatory_properties:
            if prop not in self.props.keys():
                raise ValueError(f"{prop} is required for {self.__class__.__name__}")

        allowed_properties = {**self._property_types, **self._allowed_kwargs}
        for prop, prop_type in self.props.items():
            # Check or excess properties
            if prop not in allowed_properties.keys():
                raise ValueError(f"{prop} is not a valid parameter for {self.__class__.__name__}")

            # Check for invalid types
            if not isinstance(prop_type, allowed_properties[prop]):
                raise TypeError(f"{prop} must be {allowed_properties[prop]}, got {prop_type} instead")

    def setup_default_properties(self):
        """
        Sets up default values for properties that have them defined in _property_definitions.

        """
        # Set up default values or perform transformations
        for key, value in self._property_defaults.items():
            self.props.setdefault(key, value)

    def finalize_kwargs(self, time_series_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Finalizes kwargs and saves them in self.kwargs, overwriting the placeholder.
        Converts certain kwargs into time series data based on the provided DataFrame.

        Args:
            time_series_data (pd.DataFrame): The DataFrame containing the time series data.
        """
        self.kwargs = {
            k: (as_time_series(v, time_series_data) if
                k in ["min_rel", "max_rel", "costsPerRunningHour", "costsPerFlowHour", "switchOnCosts"]
                else v)
            for k, v in self._kwargs.items()}
        return self.kwargs


    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:

        raise Exception(f"Not implemented for class 'Validator'. "
                        f"Needs to be implemented in class {self.__class__.__name__} itself")

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        """
        Needs to be implemented by Child class
        Connects the energy system object to the overall system, creating a component representation and integrating it into the system model.

        Args:
            time_series_data (pd.DataFrame): A DataFrame containing the time series data for the computation.
            co2_factors (Dict[str, float]): A dictionary mapping CO2 factors to their values.
            years_of_model (List[int]): A list of years considered existing in the model.
            effects (Dict[str, cEffectType]): A dictionary mapping effect type labels to effect type objects.
            busses (Dict[str, cBus]): A dictionary mapping bus labels to their bus objects.

        Returns:
            List[cME]: A list of components to be added to the flixOpt Model.
        """
        raise Exception(f"Not implemented for class 'Validator'. "
                        f"Needs to be implemented in class {self.__class__.__name__} itself")

class EnergySystemObject(Validator):
    """
    Represents an object within an energy system.
    Introduced Concepts:
    Basics (Name, Gruppe):
        - The name of the Component and a group the Component belongs to (for. ex. a technology, a location,...)
    Existence (Startjahr, Lebensdauer):
        - Limiting the existance of the Component in the Model
    Investment:
        - Investment into Components specified by several costs.
        - Funding a part of the Investment
        - Limiting Investments across multiple Components
    Kwargs:
        - Several optional attributes, which are directly passed to cFlow of flixOpt.flixStructure

    Attributes:
        props (dict): A dictionary holding the properties of the object, initialized from keyword arguments.
        kwargs (dict): Additional keyword arguments not directly assigned to properties.
        computed_props (dict): Computed properties derived from the object's attributes.
        flix_comps (List[cME]): A list of components associated with the object, where cME is a custom type representing a component in the energy system.

    """

    # Defining allowed properties, default values and allowed types
    _property_definitions = {
        # Basics
        "Name": (NO_DEFAULT, str),
        "Gruppe": (None, Optional[str]),
        # Existance (& Investment)
        "Startjahr": (None, Optional[int]),
        "Lebensdauer": (None, Optional[int]),
        #Investment
        "Optional": (False, bool),
        "Investkosten [€]": (0, Union[int, float]),
        "Investkosten [€/MW]": (0, Union[int, float]),
        "Zinssatz": (0, Union[int, float]),
        "Sonstige Fixkosten [€/a]": (0, Union[int, float]),
        "Sonstige Fixkosten [€/(MW*a)]": (0, Union[int, float]),
        "Fördersatz": (0, Union[int, float]),
        "Investgruppe": (None, Optional[str]),
    }
    # Defining allowed kwargs, and types
    _allowed_kwargs = {
        "min_rel": Union[int, float, str],
        "max_rel": Union[int, float, str],
        "costsPerRunningHour": Union[int, float, str],
        "costsPerFlowHour": Union[int, float, str],
        "switchOnCosts": Union[int, float, str],
        "loadFactor_min": Union[int, float],
        "loadFactor_max": Union[int, float],
        "onHoursSum_min": int,
        "onHoursSum_max": int,
        "onHours_min": int,
        "onHours_max": int,
        "offHours_min": int,
        "offHours_max": int,
        "switchOn_maxNr": int,
        "sumFlowHours_min": int,
        "sumFlowHours_max": int,
        "iCanSwitchOff": bool,
    }

    _invest_prop = None

    def validate_properties(self):
        """
        Validates the properties against the definitions and types specified in _property_definitions and _allowed_kwargs.

        Raises:
            ValueError: If a mandatory property is missing or an invalid property is provided.
            TypeError: If a property has an incorrect type.
        """
        super().validate_properties()

        # Logical Check
        if not (self.props["Startjahr"] is None) == (self.props["Lebensdauer"] is None):
            raise ValueError(f"Either set BOTH or NONE of 'Startjahr' and 'Lebensdauer'!")

        # Check for not computable investement props
        if not self.invest_args_viable:
            for prop in ['Investkosten [€]',
                         'Investkosten [€/MW]',
                         'Zinssatz', 'Sonstige Fixkosten [€/a]',
                         'Sonstige Fixkosten [€/(MW*a)]',
                         'Fördersatz',
                         'Investgruppe']:
                if self.props[prop] != self._property_defaults[prop]:
                    raise ValueError(f"If {prop} is used, 'Startjahr' and 'Lebensdauer' must be set!")

    def years_in_model(self, years: List[int]) -> int:
        """
        Computes the total number of years the object will be present in the model.

        Args:
            years (list): A list of years representing the years of the model

        Returns:
            int: The total number of years the object will be present in the model.
        """
        if "Jahre im Modell" not in self.computed_props:
            self.computed_props["Jahre im Modell"] = sum(
                index_per_year_in_model(first_year=self.props["Startjahr"],
                                        lifetime=self.props["Lebensdauer"],
                                        years_of_model=years)
            )
        return self.computed_props["Jahre im Modell"]

    def compute_investment(self, years_of_model: List[int]):
        self.computed_props["exists"] = exists(self.props["Startjahr"], self.props["Lebensdauer"], years_of_model)
        self.computed_props[self._invest_prop], min_invest, max_invest = (
            handle_invest_parameter(self.props[self._invest_prop]))

        self.computed_props[f"Investment {self._invest_prop}"] = None

        if self.invest_args_viable:
            self.computed_props["fixed_effects"], self.computed_props["specific_effects"] = costs_and_funding(
                interest_rate=self.props["Zinssatz"],
                lifetime=self.props["Lebensdauer"],
                specific_invest_costs=self.props["Investkosten [€/MW]"],
                specific_annual_costs=self.props["Sonstige Fixkosten [€/(MW*a)]"],
                invest_costs=self.props["Investkosten [€]"],
                annual_costs=self.props["Sonstige Fixkosten [€/a]"],
                funding_rate=self.props["Fördersatz"],
                years_in_model=self.years_in_model(years_of_model)
            )
            if self.props["Investgruppe"]:
                self.computed_props["specific_effects"][self.props["Investgruppe"]] = 1

            self.computed_props[f"Investment {self._invest_prop}"] = cInvestArgs(
                fixCosts={key: value for key, value in self.computed_props["fixed_effects"].items() if value},
                specificCosts={key: value for key, value in self.computed_props["specific_effects"].items() if
                               value},
                investmentSize_is_fixed=isinstance(self.computed_props[self._invest_prop], (int, float)),
                investment_is_optional=self.props["Optional"],
                min_investmentSize=min_invest,
                max_investmentSize=max_invest)

    def insert_effects_into_investargs(self, effects: Dict[str, cEffectType]) -> None:
        # Inserting effects as keys
        invest_key = f"Investment {self._invest_prop}"
        if self.computed_props[invest_key]:
            insert_effects(self.computed_props[invest_key].fixCosts, effects)
            insert_effects(self.computed_props[invest_key].specificCosts, effects)

    @property
    def invest_args_viable(self):
        "Checks if the computation of Investment paramns is possible"
        return self.props["Startjahr"] is not None and self.props["Lebensdauer"] is not None


class GridFee(EnergySystemObject):
    """
    Represents an object within an energy system With a Grid Connection.
    Newly introduced Concepts:
    Grid Connection:
        - Yearly Costs for Connecting to a Grid

    Attributes:
        props (dict): A dictionary holding the properties of the object, initialized from keyword arguments.
        kwargs (dict): Additional keyword arguments not directly assigned to properties.
        computed_props (dict): Computed properties derived from the object's attributes.
        flix_comps (List[cME]): A list of components associated with the object, where cME is a custom type representing a component in the energy system.

    """

    # Defining allowed properties, default values and allowed types
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Netzentgelt [€/(MW*a)]": (0, Union[int, float]),
    }

    def validate_properties(self):
        """
        Validates the properties against the definitions and types specified in _property_definitions and _allowed_kwargs.

        Raises:
            ValueError: If a mandatory property is missing or an invalid property is provided.
            TypeError: If a property has an incorrect type.
        """
        super().validate_properties()

        if not self.invest_args_viable and self.props['Netzentgelt [€/(MW*a)]'] != 0:
            raise ValueError(f"If 'Netzentgelt [€/(MW*a)]' is used, 'Startjahr' and 'Lebensdauer' must be set!")

    def compute_investment(self, years_of_model: List[int]):
        '''
        Extends the funcitonality of compute_investments to include Netzentgelte into the investment
        Parameters
        ----------
        years_of_model

        Returns
        -------

        '''
        super().compute_investment(years_of_model)

        if self.computed_props[f"Investment {self._invest_prop}"]:
            current = self.computed_props[f"Investment {self._invest_prop}"].specificCosts.get("costs", 0)
            self.computed_props[f"Investment {self._invest_prop}"].specificCosts["costs"] = (
                    current + self.grid_fee_per_invest_per_a * self.years_in_model(years_of_model))
        elif self.grid_fee_per_invest_per_a != 0:
            raise Exception("'Netzentgelt [€/(MW*a)]' couldnt get applied. No valid Investment found")

    @property
    def factor_grid_to_invest(self) -> float:
        '''
        Calculates the least advantageous efficiency between grid and investment power
        Typically:
        np.max(self.computed_props["exists"] / self.efficiency_from_grid))
        '''
        raise Exception("Not Implemented in Child class")

    @property
    def grid_fee_per_invest_per_a(self) -> float:
        '''
        Calculated the 'Netzentgelt [€/(MW_th*a)]' from given 'Netzentgelt [€/(MW*a)]' and efficiency
        '''
        if "Netzentgelt [€/(MW_th*a)]" not in self.computed_props:
            self.computed_props["Netzentgelt [€/(MW_th*a)]"] = (
                    self.props["Netzentgelt [€/(MW*a)]"] * self.factor_grid_to_invest)
        return self.computed_props["Netzentgelt [€/(MW_th*a)]"]


class Sink(EnergySystemObject):
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Nennleistung": (None, Optional[Union[int, float, str]]),
        "Flowname": (NO_DEFAULT, str),
        "Festes Profil": (None, Optional[str]),
        # Connections
        "Bus": (NO_DEFAULT, str),
    }

    _invest_prop = "Nennleistung"

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["Festes Profil"] = None
        if self.props["Festes Profil"]:
            self.computed_props["Festes Profil"] = as_time_series(self.props["Festes Profil"], time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:

        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        # Inserting effects as keys
        self.insert_effects_into_investargs(effects)

        comp = cSink(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            exists=self.computed_props["exists"],
            sink=cFlow(
                label=self.props["Flowname"],
                bus=busses[self.props["Bus"]],
                nominal_val=self.computed_props["Nennleistung"],
                val_rel=self.computed_props["Festes Profil"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class Source(EnergySystemObject):
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Nennleistung": (None, Optional[Union[int, float, str]]),
        "Flowname": (NO_DEFAULT, str),
        "Festes Profil": (None, Optional[str]),
        # Connections
        "Bus": (NO_DEFAULT, str),
    }

    _invest_prop = "Nennleistung"

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)

        # Direct links to time_series_data
        self.computed_props["Festes Profil"] = None
        if self.props["Festes Profil"]:
            self.computed_props["Festes Profil"] = as_time_series(self.props["Festes Profil"], time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cSource(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            exists=self.computed_props["exists"],
            source=cFlow(
                label=self.props["Flowname"],
                bus=busses[self.props["Bus"]],
                nominal_val=self.computed_props["Nennleistung"],
                val_rel=self.computed_props["Festes Profil"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class LinearTransformer_1_1(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Nennleistung": (None, Optional[Union[int, float, str]]),
        "Nennleistung In": (1e9, Union[int, float]),
        "Wirkungsgrad": (NO_DEFAULT, Union[int, float, str]),
        "Kosten pro MWh von Bus": (0, Union[int, float, str]),
        # Connections
        "Zu Bus": (NO_DEFAULT, str),
        "Von Bus": (NO_DEFAULT, str),
        # Labels
        "Flowname in": ("in", str),
        "Flowname out": ("in", str),
    }

    _invest_prop = "Nennleistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("max_rel", 1) * self.computed_props["exists"] / self.computed_props["Wirkungsgrad"])
        self.computed_props["Faktor für Netzentgeltumrechnung"] = value
        return value

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["Wirkungsgrad"] = as_time_series(self.props["Wirkungsgrad"], time_series_data)
        self.computed_props["Kosten pro MWh von Bus"] = as_time_series(
            self.props["Kosten pro MWh von Bus"], time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        flow_out = cFlow(label=self.props["Flowname out"],
                         bus=busses[self.props["Zu Bus"]],
                         nominal_val=self.computed_props["Nennleistung"],
                         investArgs=self.computed_props["Investment Nennleistung"],
                         **self.kwargs
                         )

        flow_in = cFlow(label=self.props["Flowname in"],
                        bus=busses[self.props["Von Bus"]],
                        costsPerFlowHour={effects["costs"]: self.computed_props["Kosten pro MWh von Bus"]}
                        )

        comp = cBaseLinearTransformer(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            exists=self.computed_props["exists"],
            inputs=[flow_in],
            outputs=[flow_out],
            factor_Sets=[{flow_in: self.computed_props["Wirkungsgrad"], flow_out: 1}]
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class Kessel(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "eta_th": (NO_DEFAULT, Union[int, float, str]),
        "Brennstoff": (NO_DEFAULT, str),
        "Zusatzkosten pro MWh Brennstoff": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("max_rel", 1) * self.computed_props["exists"] / self.computed_props["eta_th"])
        self.computed_props["Faktor für Netzentgeltumrechnung"] = value
        return value

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["eta_th"] = as_time_series(self.props["eta_th"], time_series_data)
        self.computed_props["Zusatzkosten pro MWh Brennstoff"] = as_time_series(
            self.props["Zusatzkosten pro MWh Brennstoff"], time_series_data)

        # Brennstoff
        self.computed_props["Brennstoffkosten"] = as_time_series(self.props["Brennstoff"], time_series_data)
        self.computed_props["CO2 Faktor"] = as_time_series(co2_factors.get(self.props["Brennstoff"], 0),
                                                               time_series_data)
        self.computed_props["CO2 Kosten"] = self.computed_props["CO2 Faktor"] * time_series_data["CO2"].to_numpy()

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        # Inserting effects as keys
        self.insert_effects_into_investargs(effects)

        comp = cKessel(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            eta=self.computed_props["eta_th"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            Q_fu=cFlow(
                label='Qfu',
                bus=busses[self.props["Brennstoff"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Brennstoffkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Brennstoff"] +
                        self.computed_props["CO2 Kosten"],
                    effects["CO2"]: self.computed_props["CO2 Faktor"]
                }
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class EHK(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "eta_th": (NO_DEFAULT, Union[int, float, str]),
        "Zusatzkosten pro MWh Strom": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromBezug", str)
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("max_rel", 1) * self.computed_props["exists"] / self.computed_props["eta_th"])
        self.computed_props["Faktor für Netzentgeltumrechnung"] = value
        return value


    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["eta_th"] = as_time_series(self.props["eta_th"], time_series_data)
        self.computed_props["Zusatzkosten pro MWh Strom"] = as_time_series(
            self.props["Zusatzkosten pro MWh Strom"], time_series_data)
        self.computed_props["Stromkosten"] = as_time_series("Strom", time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cEHK(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            eta=self.computed_props["eta_th"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            P_el=cFlow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"]}
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class Rueckkuehler(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "Strombedarf": (0, Union[int, float]),
        "Zusatzkosten pro MWh Strom": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromBezug", str)
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("max_rel", 1) * self.computed_props["exists"] * self.props["Strombedarf"])
        self.computed_props["Faktor für Netzentgeltumrechnung"] = value
        return value

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["Zusatzkosten pro MWh Strom"] = as_time_series(
            self.props["Zusatzkosten pro MWh Strom"], time_series_data)
        self.computed_props["Stromkosten"] = as_time_series("Strom", time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cCoolingTower(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            specificElectricityDemand=self.props["Strombedarf"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            P_el=cFlow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"]}
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class KWK(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "eta_th": (NO_DEFAULT, Union[int, float, str]),
        "eta_el": (NO_DEFAULT, Union[int, float, str]),
        "Brennstoff": (NO_DEFAULT, str),
        "Zusatzkosten pro MWh Brennstoff": (0, Union[int, float, str]),
        # Stromvergütung CO2
        "Vorlauftemperatur": ("TVL_FWN", Union[int, float, str]),
        "Rücklauftemperatur": ("TRL_FWN", Union[int, float, str]),
        "Umgebungstemperatur": ("Tamb", Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromEinspeisung", str)
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("max_rel", 1) * self.computed_props["exists"] / self.computed_props["eta_th"])
        self.computed_props["Faktor für Netzentgeltumrechnung"] = value
        return value

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["eta_th"] = as_time_series(self.props["eta_th"], time_series_data)
        self.computed_props["eta_el"] = as_time_series(self.props["eta_el"], time_series_data)
        self.computed_props["Brennstoffkosten"] = as_time_series(self.props["Brennstoff"], time_series_data)
        self.computed_props["Zusatzkosten pro MWh Brennstoff"] = as_time_series(
            self.props["Zusatzkosten pro MWh Brennstoff"], time_series_data)

        self.computed_props["CO2 Faktor"] = as_time_series(co2_factors.get(self.props["Brennstoff"], 0),
                                                               time_series_data)
        self.computed_props["CO2 Kosten"] = self.computed_props["CO2 Faktor"] * time_series_data["CO2"].to_numpy()
        self.computed_props["Stromerlöse"] = as_time_series("Strom", time_series_data)
        # CO2 Vergütung Strom (Ohne Kosten)
        try:
            self.computed_props["CO2 Reward Strom"] = fuel_factor_for_electrical_energy(
                electrical_efficiency=self.computed_props["eta_el"],
                thermal_efficiency=self.computed_props["eta_th"],
                inferior_temperature=as_time_series(self.props["Umgebungstemperatur"], time_series_data),
                forward_flow_temperature=as_time_series(self.props["Vorlauftemperatur"], time_series_data),
                reverse_flow_temperature=as_time_series(self.props["Rücklauftemperatur"], time_series_data)
            ) * self.computed_props["CO2 Faktor"]
        except KeyError:
            print(
                f"Computation of CO2 Reward did not work properly. Using default values instedOptimization itself isnot affected. "
                f"Only take care interpreting CO2 Emissions")
            self.computed_props["CO2 Reward Strom"] = fuel_factor_for_electrical_energy(
                electrical_efficiency=self.computed_props["eta_el"],
                thermal_efficiency=self.computed_props["eta_th"],
            ) * self.computed_props["CO2 Faktor"]

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cKWK(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            eta_el=self.computed_props["eta_el"],
            eta_th=self.computed_props["eta_th"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            P_el=cFlow(
                label="Pel",
                bus=busses[self.props["Strombus"]],
                costsPerFlowHour={
                    effects["costs"]: -self.computed_props["Stromerlöse"],
                    effects["CO2FW"]: -self.computed_props["CO2 Reward Strom"]},
            ),
            Q_fu=cFlow(
                label='Qfu',
                bus=busses[self.props["Brennstoff"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Brennstoffkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Brennstoff"] +
                        self.computed_props["CO2 Kosten"],
                    effects["CO2"]: self.computed_props["CO2 Faktor"]
                }
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class KWKekt(EnergySystemObject):
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Brennstoff Leistung": (NO_DEFAULT, Union[int, float]),
        "Elektrische Leistung (Stützpunkte)": (NO_DEFAULT, str),
        "Thermische Leistung (Stützpunkte)": (NO_DEFAULT, str),
        "Brennstoff": (NO_DEFAULT, str),
        "Zusatzkosten pro MWh Brennstoff": (0, Union[int, float, str]),
        "Ausschaltbar": (True, bool),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromEinspeisung", str)
    }

    _invest_prop = "Brennstoff Leistung"

    def validate_properties(self):
        super().validate_properties()
        # Validate Power Points
        electric_power_points = list_of_numbers_from_string(
            self.props["Elektrische Leistung (Stützpunkte)"], delimiter='-')
        thermal_power_points = list_of_numbers_from_string(
            self.props["Thermische Leistung (Stützpunkte)"], delimiter='-')
        if len(electric_power_points) != len(thermal_power_points):
            raise ValueError("The number of electricity and thermal power points must be equal")
        for epp, tpp in zip(electric_power_points, thermal_power_points):
            if epp / self.props["Brennstoff Leistung"] > 1:
                raise ValueError(f"The electric efficiency of {self.props['Name']} exceeds 100%.")
            if tpp / self.props["Brennstoff Leistung"] > 1:
                raise ValueError(f"The thermal efficiency of {self.props['Name']} exceeds 100%.")
            if (epp + tpp) / self.props["Brennstoff Leistung"] > 1:
                raise ValueError(f"The total efficiency of {self.props['Name']} exceeds 100%.")

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["Brennstoffkosten"] = as_time_series(self.props["Brennstoff"], time_series_data)
        self.computed_props["Zusatzkosten pro MWh Brennstoff"] = as_time_series(
            self.props["Zusatzkosten pro MWh Brennstoff"], time_series_data)

        self.computed_props["CO2 Faktor"] = as_time_series(co2_factors.get(self.props["Brennstoff"], 0),
                                                               time_series_data)

        self.computed_props["Elektrische Leistung (Stützpunkte)"] = list_of_numbers_from_string(
            self.props["Elektrische Leistung (Stützpunkte)"], delimiter='-')
        self.computed_props["Thermische Leistung (Stützpunkte)"] = list_of_numbers_from_string(
            self.props["Thermische Leistung (Stützpunkte)"], delimiter='-')

        self.computed_props["Stromerlöse"] = as_time_series("Strom", time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = KWKektB(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            exists=self.computed_props["exists"],
            BusFuel=busses[self.props["Brennstoff"]],
            BusTh=busses[self.props["Wärmebus"]],
            BusEl=busses[self.props["Strombus"]],
            nominal_val_Qfu=self.computed_props["Brennstoff Leistung"],
            segPel=self.computed_props["Elektrische Leistung (Stützpunkte)"],
            segQth=self.computed_props["Thermische Leistung (Stützpunkte)"],
            costsPerFlowHour_fuel={effects["costs"]:
                                       self.computed_props["Brennstoffkosten"] +
                                       self.computed_props["Zusatzkosten pro MWh Brennstoff"]
                                   },
            costsPerFlowHour_el={effects["costs"]: -self.computed_props["Stromerlöse"]},
            iCanSwitchOff=self.props["Ausschaltbar"],
            investArgs=self.computed_props[f"Investment {self._invest_prop}"],
            **self.kwargs
        )

        self.flix_comps.extend(comp)
        return self.flix_comps


class Waermepumpe(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "Zusatzkosten pro MWh Strom": (0, Union[int, float, str]),
        "COP": (None, Optional[Union[int, float, str]]),
        # COP computation
        "Quelltemperatur": (None, Optional[Union[int, float, str]]),
        "Zieltemperatur": ("TVL_FWN", Optional[Union[int, float, str]]),
        # BEW Operation Funding
        "SCOP für BEW": (None, Optional[Union[int, float]]),
        "Maximale Stromkostenförderung BEW": (None, Optional[float]),
        # Einsatzbeschränkung
        "Untergrenze für Einsatz": (None, Optional[Union[int, float]]),
        "Zeitreihe für Einsatzbeschränkung": (None, Optional[str]),

        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromBezug", str)
    }

    _invest_prop = "Thermische Leistung"

    def validate_properties(self):
        super().validate_properties()

        # Check for valid COP computation
        if not self.props["COP"]:
            if not self.props["Quelltemperatur"] or not self.props["Zieltemperatur"]:
                raise Exception(f"Need to specify a 'COP' for {self.props['Name']} or "
                                f"use 'Quelltemperatur' and 'Zieltemperatur' to calculate the COP internally.")
        if self.props["COP"]:
            if self.props["Quelltemperatur"]:  #or self.props["Zieltemperatur"]: # TODO: Make Zieltemperatur defualt= None
                raise Exception(f"Either specify a 'COP' for {self.props['Name']} "
                                f"OR use 'Quelltemperatur' and 'Zieltemperatur' to calculate the COP internally.")

        # BEW Operation Funding
        if self.props["SCOP für BEW"]:
            if not self.props["Startjahr"]:
                raise Exception(f"Need to specify a Year of Operation start for {self.props['Name']} to use HP "
                                f"operation funding, because its limited to 10 years.")
            if not self.props["Maximale Stromkostenförderung BEW"]:
                raise Exception(f"Need to specify 'Maximale Stromkostenförderung BEW' for {self.props['Name']} to use HP "
                                f"operation funding.")

        # Einsatzbeschränkung
        if not ((self.props["Untergrenze für Einsatz"] is None) ==
                (self.props["Zeitreihe für Einsatzbeschränkung"] is None)):
            raise Exception(f"Need to specify either both or none of 'Zeitreihe für Einsatzbeschränkung' and "
                            f"'Untergrenze für Einsatz' for {self.props['Name']}.")

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.computed_props["Einsatzbeschränkung"] * self.computed_props["exists"] /
                       self.computed_props["COP"])
        self.computed_props["Faktor für Netzentgeltumrechnung"] = value
        return value

    def compute_cop(self, time_series_data) -> Union[float, np.ndarray]:
        if self.props["COP"]:
            self.computed_props["COP"] = as_time_series(self.props["COP"], time_series_data)
        else:
            self.computed_props["COP"] = calculate_cop(
                source_temperature=as_time_series(self.props["Quelltemperatur"], time_series_data),
                target_temperature=as_time_series(self.props["Zieltemperatur"], time_series_data),
                eta=0.5)
        return self.computed_props["COP"]

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["Zusatzkosten pro MWh Strom"] = as_time_series(
            self.props["Zusatzkosten pro MWh Strom"], time_series_data)
        self.computed_props["Stromkosten"] = as_time_series("Strom", time_series_data)

        # COP berechnen
        self.compute_cop(time_series_data)

        # BEW Operation Funding
        self.computed_props["BEW Förderung Strom"] = 0
        if self.props["SCOP für BEW"]:
            fund_per_mw_el = bew_operation_funding_from_scop(self.props["SCOP für BEW"], "MWh_el")

            electricity_costs_per_flow_hour = (self.computed_props["Stromkosten"] +
                                               self.computed_props["Zusatzkosten pro MWh Strom"])
            # Begrenzung der Förderung auf x% der Stromkosten
            max_fund = self.props["Maximale Stromkostenförderung BEW"]
            fund_per_mw_el = np.where(
                fund_per_mw_el < electricity_costs_per_flow_hour * max_fund,
                fund_per_mw_el, electricity_costs_per_flow_hour * max_fund)
            # Begrenzung auf 10 Jahre
            self.computed_props["BEW Förderung Strom"] = fund_per_mw_el * exists(self.props["Startjahr"], 10,
                                                                                 years_of_model)

        # Einsatzbeschränkung
        self.computed_props["Einsatzbeschränkung"] = self.kwargs.pop("max_rel", 1)
        if self.props["Zeitreihe für Einsatzbeschränkung"]:
            self.computed_props["Einsatzbeschränkung"] = np.where(
                as_time_series(self.props["Zeitreihe für Einsatzbeschränkung"], time_series_data)
                <= self.props["Untergrenze für Einsatz"],
                0, self.computed_props["Einsatzbeschränkung"])

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cHeatPump(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            COP=self.computed_props["COP"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                max_rel=self.computed_props["Einsatzbeschränkung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            P_el=cFlow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"],
                    effects["funding"]: self.computed_props["BEW Förderung Strom"]
                }
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class AbwaermeWaermepumpe(Waermepumpe):
    _property_definitions = {
        **Waermepumpe._property_definitions,
        "Abwärmekosten": (NO_DEFAULT, Union[int, float, str]),
        # Connections
        "Abwärmebus": ("Abwaerme", str)
    }

    _invest_prop = "Thermische Leistung"

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        super().computation(years_of_model, co2_factors, time_series_data)
        self.computed_props["Abwärmekosten"] = as_time_series(self.props["Abwärmekosten"], time_series_data)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cAbwaermeHP(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            COP=self.computed_props["COP"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                max_rel=self.computed_props["Einsatzbeschränkung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            P_el=cFlow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"],
                    effects["funding"]: self.computed_props["BEW Förderung Strom"]
                },
            ),
            Q_ab=cFlow(
                label='Qab',
                bus=busses[self.props["Abwärmebus"]],
                costsPerFlowHour={effects["costs"]: self.computed_props["Abwärmekosten"]}
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class Geothermie(Waermepumpe):
    _property_definitions = {
        **Waermepumpe._property_definitions,
        "Anteil Pumpstrom pro MW_geo": (NO_DEFAULT, Union[int, float]),
        # Connections
        "Abwärmebus": ("Abwaerme", str)
    }

    _invest_prop = "Thermische Leistung"

    def compute_cop(self, time_series_data) -> Union[float, np.ndarray]:
        super().compute_cop(time_series_data)
        self.computed_props["COP ohne Pumpstrom"] = self.computed_props["COP"]
        self.computed_props["COP"] = self.computed_props["COP ohne Pumpstrom"] / (
                1 + self.props["Anteil Pumpstrom pro MW_geo"])
        return self.computed_props["COP"]

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:

        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cAbwaermeHP(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            COP=self.computed_props["COP"],
            exists=self.computed_props["exists"],
            Q_th=cFlow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                nominal_val=self.computed_props["Thermische Leistung"],
                max_rel=self.computed_props["Einsatzbeschränkung"],
                investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                **self.kwargs
            ),
            P_el=cFlow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                costsPerFlowHour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"],
                    effects["funding"]: self.computed_props["BEW Förderung Strom"]
                },
            ),
            Q_ab=cFlow(
                label='Qab',
                bus=busses[self.props["Abwärmebus"]],
            )
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class Abwaerme(EnergySystemObject):
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "Abwärmekosten": (NO_DEFAULT, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Abwärmebus": ("Abwaerme", str)
    }

    _invest_prop = "Thermische Leistung"

    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Direct links to time_series_data
        self.computed_props["Abwärmekosten"] = as_time_series(self.props["Abwärmekosten"], time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        q_th = cFlow(label='Qth',
                     bus=busses[self.props["Wärmebus"]],
                     nominal_val=self.computed_props["Thermische Leistung"],
                     investArgs=self.computed_props[f"Investment {self._invest_prop}"],
                     **self.kwargs
                     )

        q_abw = cFlow(label='Qabw',
                      bus=busses[self.props["Abwärmebus"]],
                      costsPerFlowHour={effects["costs"]: self.computed_props["Abwärmekosten"]}
                      )

        comp = cBaseLinearTransformer(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            exists=self.computed_props["exists"],
            inputs=[q_abw],
            outputs=[q_th],
            factor_Sets=[{q_abw: 1, q_th: 1}]
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class Speicher(EnergySystemObject):
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "Kapazität [MWh]": (None, Optional[Union[int, float, str]]),
        # Investment Kapazität
        "Investkosten [€/MWh]": (0, Union[int, float]),
        "Sonstige Fixkosten [€/(MWh*a)]": (0, Union[int, float]),
        "Investgruppe Kapazität": (None, Optional[str]),

        "VerlustProStunde": (0, Union[int, float]),
        "eta_load": (NO_DEFAULT, Union[int, float]),
        "eta_unload": (NO_DEFAULT, Union[int, float]),
        # Beschränkung
        "AbhängigkeitVonDT": (False, bool),
        "Untere Temperatur": ("TRL_FWN", Union[int, float, str]),
        "Obere Temperatur": ("TVL_FWN", Union[int, float, str]),
    }

    _invest_prop = "Thermische Leistung"

    def compute_investment(self, years_of_model: List[int]):
        super().compute_investment(years_of_model)
        self.computed_props[f"Investment {self._invest_prop} 2"] = None
        if self.invest_args_viable:
            investmentSize_is_fixed = isinstance(self.computed_props[self._invest_prop], (int, float))
            self.computed_props[f"Investment {self._invest_prop} 2"] = cInvestArgs(
                investmentSize_is_fixed=investmentSize_is_fixed,
                investment_is_optional=self.props["Optional"])

            # Add Effect to link installed thermal power of in and out flows
            if not investmentSize_is_fixed:
                fx = cEffectType(label=f"{self.props['Name']}_link_power", unit="",
                                 description=f"Links the in and outflow investment value of storage {self.props['Name']}",
                                 min_investSum=0, max_investSum=0)
                self.flix_comps.append(fx)
                self.computed_props[f"Investment {self._invest_prop}"].specificCosts[fx] = 1
                self.computed_props[f"Investment {self._invest_prop} 2"].specificCosts = {fx: -1}

    def compute_investment_capacity(self, years_of_model: List[int]):
        self.computed_props["Kapazität [MWh]"], min_invest, max_invest = (
            handle_invest_parameter(self.props["Kapazität [MWh]"]))

        self.computed_props[f"Investment Kapazität [MWh]"] = None
        if self.invest_args_viable:
            self.computed_props["fixed_effects_capacity"], self.computed_props[
                "specific_effects_capacity"] = costs_and_funding(
                interest_rate=self.props["Zinssatz"],
                lifetime=self.props["Lebensdauer"],
                invest_costs=0, annual_costs=0,
                specific_invest_costs=self.props["Investkosten [€/MWh]"],
                specific_annual_costs=self.props["Sonstige Fixkosten [€/(MWh*a)]"],
                funding_rate=self.props["Fördersatz"],
                years_in_model=self.years_in_model(years_of_model)
            )
            if self.props["Investgruppe Kapazität"]:
                self.computed_props["specific_effects_capacity"][self.props["Investgruppe Kapazität"]] = 1

            self.computed_props["Investment Kapazität [MWh]"] = cInvestArgs(
                fixCosts={key: value for key, value in self.computed_props["fixed_effects_capacity"].items() if value},
                specificCosts={key: value for key, value in self.computed_props["specific_effects_capacity"].items() if
                               value},
                investmentSize_is_fixed=isinstance(self.computed_props["Kapazität [MWh]"], (int, float)),
                investment_is_optional=self.props["Optional"],
                min_investmentSize=min_invest,
                max_investmentSize=max_invest)


    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Limiting capcity and Thermische Leistung
        self.computed_props["Einsatzbeschränkung"] = (
            as_time_series(self.props["Obere Temperatur"], time_series_data) -
            as_time_series(self.props["Untere Temperatur"], time_series_data)) / 65
        if np.max(self.computed_props["Einsatzbeschränkung"]) > 1:
            raise Exception(
                f"The maximal temperature spread is too high for the capacity of the storage. Maximum is 65")
        if "max_rel" in self.kwargs:
            max_rel = self.kwargs.pop("max_rel")
            self.computed_props["Einsatzbeschränkung"] = np.where(
                self.computed_props["Einsatzbeschränkung"] < max_rel,
                self.computed_props["Einsatzbeschränkung"], max_rel)

        self.compute_investment(years_of_model)
        self.compute_investment_capacity(years_of_model)

    def insert_effects_into_investargs(self, effects: Dict[str, cEffectType]) -> None:
        # Inserting effects as keys
        invest_keys = [f"Investment {self._invest_prop}",
                       f"Investment {self._invest_prop} 2",
                       f"Investment Kapazität [MWh]"]
        for invest_key in invest_keys:
            if self.computed_props[invest_key]:
                insert_effects(self.computed_props[invest_key].fixCosts, effects)
                insert_effects(self.computed_props[invest_key].specificCosts, effects)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, cEffectType],
                          busses: Dict[str, cBus]) -> List[cME]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = cStorage(
            label=self.props["Name"],
            group=self.props["Gruppe"],
            exists=self.computed_props["exists"],
            capacity_inFlowHours=self.computed_props["Kapazität [MWh]"],
            eta_load=self.props["eta_load"],
            eta_unload=self.props["eta_unload"],
            fracLossPerHour=self.props["VerlustProStunde"],
            max_rel_chargeState=self.computed_props["Einsatzbeschränkung"]
            if isinstance(self.computed_props["Einsatzbeschränkung"], (int, float))
            else np.append(self.computed_props["Einsatzbeschränkung"], self.computed_props["Einsatzbeschränkung"][-1]),
            investArgs=self.computed_props["Investment Kapazität [MWh]"],

            inFlow=cFlow(label='QthLoad',
                         bus=busses["Fernwaerme"],
                         nominal_val=self.computed_props["Thermische Leistung"],
                         max_rel=self.computed_props["Einsatzbeschränkung"],
                         investArgs=self.computed_props["Investment Thermische Leistung"],
                         **self.kwargs
                         ),
            outFlow=cFlow(label='QthUnload',
                          bus=busses["Fernwaerme"],
                          nominal_val=self.computed_props["Thermische Leistung"],
                          max_rel=self.computed_props["Einsatzbeschränkung"],
                          investArgs=self.computed_props["Investment Thermische Leistung 2"],
                          ),
            avoidInAndOutAtOnce=True,
        )

        self.flix_comps.append(comp)
        return self.flix_comps


class ComponentFactory:
    def __init__(self,
                 time_series_data: pd.DataFrame,
                 co2_factors: Dict[str, float],
                 years_of_model: List[int],
                 effects: Dict[str, cEffectType],
                 busses: Dict[str, cBus]):
        self.time_series_data = time_series_data
        self.co2_factors = co2_factors
        self.years_of_model = years_of_model
        self.effects = effects
        self.busses = busses

        self.created_comps: List[EnergySystemObject] = []

    def create_energy_object(self, obj_type, **properties):
        obj_class = self.get_class_by_type(obj_type)
        if obj_class:
            energy_obj = obj_class(**properties)
            self.created_comps.append(energy_obj)
            return energy_obj.connect_to_system(time_series_data=self.time_series_data,
                                                co2_factors=self.co2_factors,
                                                years_of_model=self.years_of_model,
                                                effects=self.effects,
                                                busses=self.busses)
        else:
            raise ValueError(f"Unknown energy object type: {obj_type}")

    def get_class_by_type(self, obj_type):
        # Map obj_type to the appropriate class
        class_map = {
            'Waermepumpe': Waermepumpe,
            'AbwaermeWP': AbwaermeWaermepumpe,
            'Geothermie': Geothermie,
            'KWK': KWK,
            'KWKekt': KWKekt,
            'Kessel': Kessel,
            'EHK': EHK,
            'Speicher': Speicher,
            'AbwaermeHT': Abwaerme,
            'Rueckkuehler': Rueckkuehler,
            'LinearTransformer_1_1': LinearTransformer_1_1,
            'Sink': Sink,
            'Source': Source,

            # More mappings as needed
        }
        return class_map.get(obj_type)

    def print_comps(self):
        rep = ""
        for comp in sorted(self.created_comps, key=lambda comp: comp.props["Name"]):
            rep += f"{comp}\n"
        return rep


##############################

def exists(first_year: int, lifetime: int, years_in_model: list[int], steps_per_year: int = 8760) -> [int, np.ndarray]:
    index_per_year = np.array(index_per_year_in_model(first_year, lifetime, years_in_model))
    if np.sum(index_per_year) == 0:
        return 0
    if np.sum(index_per_year) == len(index_per_year):
        return 1
    return np.repeat(index_per_year, steps_per_year)


def index_per_year_in_model(first_year: Optional[int], lifetime: Optional[int], years_of_model: list[int]) -> List[int]:
    if first_year is None and lifetime is None:
        return [1] * len(years_of_model)
    elif first_year is None or lifetime is None:
        raise Exception("Either both or none of 'Startjahr' and 'Endjahr' must be set per Component.")
    else:
        # Create a new list with 1s and 0s based on the conditions
        return [1 if first_year <= num < first_year + lifetime else 0 for num in years_of_model]

def as_time_series(value: Union[float, int, str], time_series_data: pd.DataFrame) -> Union[int, float, np.ndarray]:
    if isinstance(value, (int, float)):
        # return np.ones(len(time_series_data.index)) * value
        return value
    elif value in time_series_data.columns:
        return time_series_data[value].to_numpy()
    else:
        raise KeyError(f"{value} is not in TimeSeries Data of the DistrictHeatingSystem.")

def calculate_cop(source_temperature: np.ndarray, target_temperature: np.ndarray, eta: float = 0.5) -> np.ndarray:
    '''
    Calculates the COP of a heatpump per Timestep from the Temperature of Heat sink and Heat source in Kelvin
    Parameters
    ----------
    source_temperature : np.array, float, pd.Dataframe
        Temperature of the Heat Source in Degrees Celcius
    target_temperature : np.array, float, pd.Dataframe
        Temperature of the Heat Sink in Degrees Celcius
    eta : float
        Relation to the thermodynamicaly ideal COP

    Returns
    -------
    np.ndarray

    '''
    # Celsius zu Kelvin
    source_temperature = source_temperature + 273.15
    target_temperature = target_temperature + 273.15
    return (target_temperature / (target_temperature - source_temperature)) * eta

def bew_operation_funding_from_scop(scop: Union[int, float],
                                    unit: Literal["MWh_amb", "MWh_th", "MWh_el"] = "MWh_amb") -> Union[int, float]:
    '''
    Calclulated the maximum funding according to the BEW.
    Parameters
    ----------
    scop: assumed scop (seasonal coefficent of Performance) or cop

    Returns
    -------
    Funding in euro per MWh_amb

    '''
    if unit not in ["MWh_amb", "MWh_th", "MWh_el"]:
        raise Exception(f"Not a valid unit. Choose from: {['MWh_amb', 'MWh_th', 'MWh_el']}")

    value = (5.5 - (6.8 - 17 / scop) * 0.75) * (scop / (scop - 1))  # ct/kWh
    fund_amb = value * 10  # €/MWh_amb
    if fund_amb >= 92:  # Funding is limited to 92 €/MWh_amb
        fund_amb = 92

    if unit == "MWh_amb":
        return fund_amb
    elif unit == "MWh_th":
        return fund_amb * ((scop - 1) / scop)
    else:
        return fund_amb * (scop - 1)


def insert_effects(dictionary: Dict[Union[cEffectType, str], Any],
                   effects: Dict[str, cEffectType]) -> None:

    if dictionary is None or dictionary == 0 or dictionary == {}:
        return None
    for effect_name, value in list(dictionary.items()):
        if effect_name in effects.keys():
            dictionary[effects[effect_name]] = dictionary.pop(effect_name)
        elif not isinstance(effect_name, cEffectType):
            raise KeyError(f"Key '{effect_name}' is not found in effects Collection.")

def get_annuity_factor(interest_rate: float, lifetime: int) -> float:
    if interest_rate == 0:  # Preventing ZeroDicvision
        annuity_factor = 1 / lifetime
    else:
        annuity_factor = (((1 + interest_rate) ** lifetime * interest_rate) /
                          ((1 + interest_rate) ** lifetime - 1))
    return annuity_factor

def costs_and_funding(
        interest_rate: float,
        lifetime: int,
        years_in_model: int,
        invest_costs: float,
        specific_invest_costs: float,
        annual_costs: float,
        specific_annual_costs: float,
        funding_rate: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    '''
    Calculates the annual costs and funding for an investment based on various financial parameters.

    This function computes the fixed and specific costs and funding for an investment, considering the
    interest rate, lifetime of the investment, investment costs (both per MW and per year), other costs (both per MW
    and per year), funding rate, and grid fee per MW per year. The costs and funding are calculated using the annuity
    method, which spreads out the initial investment costs over the lifetime of the investment, adjusted for the
    interest rate.

    Parameters:
    - interest_rate (float): The annual interest rate used for calculating the annuity factor.
    - lifetime (int): The lifetime of the investment in years.
    - invest_costs (float): The total investment costs.
    - invest_costs_per_mw (float): The investment costs per megawatt (MW).
    - other_annual_costs (float): Other annual costs not included in the investment costs.
    - other_annual_costs_per_mw (float): Other costs per megawatt (MW) not included in the investment costs.
    - funding_rate (float): The rate at which the investment is funded.

    Returns:
    - Tuple[Dict[str, float], Dict[str, float]]: A tuple containing two dictionaries:
        1. Fixed costs and funding, with keys being strings and values being the corresponding amounts in currency units.
        2. Specific costs and funding, similar to the fixed costs but calculated per MW.
    '''

    annuity_factor = get_annuity_factor(interest_rate=interest_rate, lifetime=lifetime)

    # Calculate costs and funding
    fix_costs = {
        "costs": (invest_costs * annuity_factor + annual_costs) * years_in_model,
        "funding": invest_costs * annuity_factor * funding_rate * years_in_model
    }
    specific_costs = {
        "costs": (specific_invest_costs * annuity_factor + specific_annual_costs) *
                 years_in_model,
        "funding": specific_invest_costs * annuity_factor * funding_rate * years_in_model
    }

    return fix_costs, specific_costs

def handle_invest_parameter(invest_parameter: Union[int, float, str, type(None)]) -> Tuple[
    Optional[Union[int, float]], float, float]:
    '''
    Handles an 'invest_parameter' value by assessing its type and assigning appropriate min, max, and value variables.

    If 'invest_parameter' is string, it should be in the format 'min-max'. If it doesn't follow this format,
    an exception will be raised. In this case, min and max are parsed from the string, and value is set to None.

    If 'invest_parameter' is not string, the min is set to 0, max is set to 1e9 and value is set to the
    'invest_parameter' itself.

    Args:
        invest_parameter (Union[str, int, float]): A number (int, float) or min-max range (string)

    Returns:
        Tuple: Return a tuple containing:
            - value : Value of 'invest_parameter', if it was number. Else, None
            - min   : Minimum limit for 'invest_parameter'
            - max   : Maximum Limit for 'invest_parameter'

    Raises:
        Exception: If 'invest_parameter' is string but does not follow 'min-max' format
    '''
    min, max = 0, 1e9
    if isinstance(invest_parameter, type(None)):
        return None, min, max
    if isinstance(invest_parameter, (int, float)):
        value = invest_parameter
        return value, min, max
    if isinstance(invest_parameter, str):
        lower_bound, upper_bound = check_min_max_format(invest_parameter)
        return None, lower_bound, upper_bound

    raise Exception(f"Wrong format of string for thermal_power '{invest_parameter}'."
                    f"If thermal power is passed as a string, it must be of the format 'min-max'")

def check_min_max_format(input_string: str) -> Tuple[float, float]:
    '''
    This function checks if a string is of the format "min-max" where min and max can be integers or decimal numbers
    with . or , as decimal separators.

    Parameters
    ----------
    input_string : str
        The input string to check.

    Returns
    -------
    bool
        True if the string matches the "min-max" format, False otherwise.
    '''
    input_string = input_string.replace(',', '.').replace(' ', '')
    if not re.match(r'^\d+(.\d+)?-\d+(.\d+)?$', input_string):
        raise ValueError(f"String '{input_string}' is not of Format 'min-max'")
    lower_bound, upper_bound = input_string.split("-")
    return float(lower_bound), float(upper_bound)


def fuel_factor_for_electrical_energy(
        electrical_efficiency: Union[int, float, np.ndarray],
        thermal_efficiency: Union[int, float, np.ndarray],
        inferior_temperature: Union[int, float, np.ndarray] = 20,
        forward_flow_temperature: Union[int, float, np.ndarray] = 120,
        reverse_flow_temperature: Union[int, float, np.ndarray] = 60,
) -> np.ndarray:
    '''
    Using the carnot mehtod, the fuel factor for electrical energy in a heating network is calculated
    https://en.wikipedia.org/wiki/Carnot_method
    '''
    inferior_temperature = inferior_temperature + 273.15
    forward_flow_temperature = forward_flow_temperature + 273.15
    reverse_flow_temperature = reverse_flow_temperature + 273.15
    superior_temperature = ((forward_flow_temperature - reverse_flow_temperature) /
                            np.log((forward_flow_temperature / reverse_flow_temperature)))
    n_carnot = 1 - (inferior_temperature / superior_temperature)

    a_el = (1 * electrical_efficiency) / (electrical_efficiency + n_carnot * thermal_efficiency)
    return a_el / electrical_efficiency

def list_of_numbers_from_string(input_string: str, delimiter='-') -> List[float]:
    '''
    This function was written to extract numbers from a string
    ----------
    Returns
    -------
    List[float]
    '''
    input_string = input_string.replace(',', '.')
    return [float(i) for i in input_string.split(delimiter)]

def is_valid_format_segmentsOfFlows(input_string: str, mode: Literal['validate', 'decode']) -> Union[bool, list]:
    '''
    This function was written to check if a string is of the format "0;0 ;5;10 ; 10;30"
    In mode 'validate, returns bool. In mode 'decode', returns a list of numbers
    ----------
    Returns
    -------
    bool
    '''

    # Replace commas with dots to handle decimal separators
    input_string = input_string.replace(',', '.')

    # Split the string into a list of substrings using semicolon as the delimiter
    numbers_str = input_string.split(';')
    # Convert each substring to either int or float
    numbers = [int(num) if '.' not in num else float(num) for num in numbers_str]

    if not isinstance(numbers, list):
        pass
        # raise Exception("Conversion to segmentsOfFlows didnt work. Use numbers, seperated by ';'")
    elif not all(isinstance(element, (int, float)) for element in numbers):
        pass
        # raise Exception("Conversion to segmentsOfFlows didnt work. Use numbers, seperated by ';'")
    else:
        if mode == 'validate':
            return True
        elif mode == 'decode':
            return numbers
        else:
            raise Exception(f"{mode} is not a valid mode.")
    if mode == 'validate':
        return False
    else:
        raise Exception("Error encountered in parsing of String")

def print_dict(data: Dict[str, Union[str, int, float, np.ndarray]]) -> str:
    keys = sorted(data.keys())
    values = [data[key] for key in keys]

    representation = ""
    for key, value in zip(keys, values):
        representation += f"{key} = {value}\n"

    return representation