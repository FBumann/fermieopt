# -*- coding: utf-8 -*-
import re
import logging
import textwrap
from typing import Union, List, Dict, Optional, Literal, Any, Tuple

import pandas as pd
import numpy as np

import flixOpt as fx
import flixOpt.elements
from flixOpt.structure import Element

NO_DEFAULT = object()  # Unique object as signal that no default value exists

logger = logging.getLogger('flixOpt')

class Validator:
    """
    This class is Used to provide a general structure and functionality to reliably create Objects from kwargs.
    It provides type validation and properties to distinguish between optional and mandatory kwargs

    Attributes:
        props (dict): A dictionary holding the properties of the object, initialized from keyword arguments.
        kwargs (dict): Additional keyword arguments not directly assigned to properties.
        computed_props (dict): Computed properties derived from the object's attributes.
        flix_comps (List[Element]): A list of components associated with the object, where Element is a custom type
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
        connect_to_system(time_series_data: pd.DataFrame, co2_factors: Dict[str, float], years_of_model: List[int], effects: Dict[str, fx.Effect], busses: Dict[str, fx.Bus]) -> List[Element]: Initializes a flixOpt Component from the computed data.
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
        self.meta_data = {}
        self.flix_comps: List[Element] = []

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
                k in ["relative_minimum", "relative_maximum", "effects_per_running_hour", "effects_per_flow_hour",
                      "effects_per_switch_on"]
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        """
        Needs to be implemented by Child class
        Connects the energy system object to the overall system, creating a component representation and integrating it into the system model.

        Args:
            time_series_data (pd.DataFrame): A DataFrame containing the time series data for the computation.
            co2_factors (Dict[str, float]): A dictionary mapping CO2 factors to their values.
            years_of_model (List[int]): A list of years considered existing in the model.
            effects (Dict[str, fx.Effect]): A dictionary mapping effect type labels to effect type objects.
            busses (Dict[str, fx.Bus]): A dictionary mapping bus labels to their bus objects.

        Returns:
            List[Element]: A list of components to be added to the flixOpt Model.
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
        - Several optional attributes, which are directly passed to fx.Flow of flixOpt.flixStructure

    Attributes:
        props (dict): A dictionary holding the properties of the object, initialized from keyword arguments.
        kwargs (dict): Additional keyword arguments not directly assigned to properties.
        computed_props (dict): Computed properties derived from the object's attributes.
        flix_comps (List[Element]): A list of components associated with the object, where Element is a custom type representing a component in the energy system.

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
        "relative_minimum": Union[int, float, str],
        "relative_maximum": Union[int, float, str],
        "flow_hours_total_min": int,
        "flow_hours_total_max": int,
        "load_factor_min": Union[int, float],
        "load_factor_max": Union[int, float],
        "effects_per_flow_hour": Union[int, float, str],

        "effects_per_running_hour": Union[int, float, str],
        "effects_per_switch_on": Union[int, float, str],
        "on_hours_total_min": int,
        "on_hours_total_max": int,
        "consecutive_on_hours_min": int,
        "consecutive_on_hours_max": int,
        "switch_on_total_max": int,
    }

    _flow_kwargs = ["relative_minimum", "relative_maximum", "flow_hours_total_min", "flow_hours_total_max",
                    "load_factor_min", "load_factor_max", "effects_per_flow_hour"]

    _on_kwargs = ["effects_per_running_hour", "effects_per_switch_on", "on_hours_total_min", "on_hours_total_max",
                    "consecutive_on_hours_min", "consecutive_on_hours_max", "switch_on_total_max"]

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
            self.meta_data["fixed_effects"], self.meta_data["specific_effects"] = costs_and_funding(
                interest_rate=self.props["Zinssatz"],
                starting_year=self.props["Startjahr"],
                lifetime=self.props["Lebensdauer"],
                specific_invest_costs=self.props["Investkosten [€/MW]"],
                specific_annual_costs=self.props["Sonstige Fixkosten [€/(MW*a)]"],
                invest_costs=self.props["Investkosten [€]"],
                annual_costs=self.props["Sonstige Fixkosten [€/a]"],
                funding_rate=self.props["Fördersatz"],
                years_of_model=years_of_model
            )
            self.computed_props["fixed_effects"] = {key: sum(value) for key, value in
                                                    self.meta_data["fixed_effects"].items()}
            self.computed_props["specific_effects"] = {key: sum(value) for key, value in
                                                       self.meta_data["specific_effects"].items()}

            if self.props["Investgruppe"]:
                self.computed_props["specific_effects"][self.props["Investgruppe"]] = 1

            size = self.computed_props[self._invest_prop]
            self.computed_props[f"Investment {self._invest_prop}"] = fx.InvestParameters(
                fix_effects={key: value for key, value in self.computed_props["fixed_effects"].items() if value},
                specific_effects={key: value for key, value in self.computed_props["specific_effects"].items() if
                               value},
                fixed_size=size if isinstance(size, (int, float)) else None,
                optional=self.props["Optional"],
                minimum_size=min_invest,
                maximum_size=max_invest)

    def insert_effects_into_investargs(self, effects: Dict[str, fx.Effect]) -> None:
        # Inserting effects as keys
        invest_key = f"Investment {self._invest_prop}"
        if self.computed_props[invest_key]:
            insert_effects(self.computed_props[invest_key].fix_effects, effects)
            insert_effects(self.computed_props[invest_key].specific_effects, effects)

    @property
    def invest_args_viable(self):
        "Checks if the computation of Investment paramns is possible"
        return self.props["Startjahr"] is not None and self.props["Lebensdauer"] is not None

    def accounting_years(self, years_of_model: List[int]) -> np.ndarray:
        lifetime, start_year = self.props["Lebensdauer"], self.props["Startjahr"]
        return np.array([1 if start_year <= year < (start_year + lifetime) else 0 for year in years_of_model])

    @property
    def flow_kwargs(self) -> Dict[str, Any]:
        return {key: value for key, value in self.kwargs.items() if key in self._flow_kwargs}

    @property
    def on_kwargs(self) -> Dict[str, Any]:
        return {key: value for key, value in self.kwargs.items() if key in self._on_kwargs}

    @property
    def on_parameters(self) -> Optional[fx.OnOffParameters]:
        return fx.OnOffParameters(self.on_kwargs) if self.on_kwargs else None


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
        flix_comps (List[Element]): A list of components associated with the object, where Element is a custom type representing a component in the energy system.

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
            current = self.computed_props[f"Investment {self._invest_prop}"].specific_effects.get("costs", 0)
            self.computed_props[f"Investment {self._invest_prop}"].specific_effects["costs"] = (
                    current + self.grid_fee_per_invest_per_a * self.years_in_model(years_of_model))

            self.meta_data["specific_effects"]['costs'] = (
                    self.meta_data["specific_effects"].get('costs', np.array([0])) +
                    self.grid_fee_per_invest_per_a * self.accounting_years(years_of_model)
            )

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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:

        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        # Inserting effects as keys
        self.insert_effects_into_investargs(effects)

        comp = fx.Sink(
            label=self.props["Name"],
            sink=fx.Flow(
                meta_data=self.meta_data,
                label=self.props["Flowname"],
                bus=busses[self.props["Bus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Nennleistung"] ,
                fixed_relative_profile=self.computed_props["Festes Profil"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            )
        )

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.Source(
            label=self.props["Name"],
            source=fx.Flow(
                label=self.props["Flowname"],
                meta_data=self.meta_data,
                bus=busses[self.props["Bus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Nennleistung"],
                fixed_relative_profile=self.computed_props["Festes Profil"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            )
        )

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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
        "Flowname out": ("out", str),
    }

    _invest_prop = "Nennleistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("relative_maximum", 1) * self.computed_props["exists"] / self.computed_props["Wirkungsgrad"])
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        flow_out = fx.Flow(label=self.props["Flowname out"],
                           meta_data=self.meta_data,
                           bus=busses[self.props["Zu Bus"]],
                           size=self.computed_props["Investment Nennleistung"] or self.computed_props["Nennleistung"],
                           can_be_off=self.on_parameters,
                           **self.flow_kwargs
                         )

        flow_in = fx.Flow(label=self.props["Flowname in"],
                        bus=busses[self.props["Von Bus"]],
                        effects_per_flow_hour={effects["costs"]: self.computed_props["Kosten pro MWh von Bus"]}
                        )

        comp = fx.LinearConverter(
            label=self.props["Name"],
            inputs=[flow_in],
            outputs=[flow_out],
            conversion_factors=[{flow_in: self.computed_props["Wirkungsgrad"], flow_out: 1}]
        )

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
        self.flix_comps.append(comp)
        return self.flix_comps


class Kessel(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "eta_th": (NO_DEFAULT, Union[int, float, str]),
        "Brennstoff": (NO_DEFAULT, str),
        "Zusatzkosten pro MWh Brennstoff": (0, Union[int, float, str]),
        "Grüne Wärme": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("relative_maximum", 1) * self.computed_props["exists"] / self.computed_props["eta_th"])
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        # Inserting effects as keys
        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.Boiler(
            label=self.props["Name"],
            eta=self.computed_props["eta_th"],
            Q_th=fx.Flow(
                label='Qth',
                meta_data=self.meta_data,
                bus=busses[self.props["Wärmebus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            Q_fu=fx.Flow(
                label='Qfu',
                bus=busses[self.props["Brennstoff"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Brennstoffkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Brennstoff"] +
                        self.computed_props["CO2 Kosten"],
                    effects["CO2"]: self.computed_props["CO2 Faktor"]
                }
            )
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(comp.Q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
        self.flix_comps.append(comp)
        return self.flix_comps


class EHK(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "eta_th": (NO_DEFAULT, Union[int, float, str]),
        "Zusatzkosten pro MWh Strom": (0, Union[int, float, str]),
        "Grüne Wärme": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromBezug", str)
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("relative_maximum", 1) * self.computed_props["exists"] / self.computed_props["eta_th"])
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.Power2Heat(
            label=self.props["Name"],
            eta=self.computed_props["eta_th"],
            Q_th=fx.Flow(
                label='Qth',
                meta_data=self.meta_data,
                bus=busses[self.props["Wärmebus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"]}
            )
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(comp.Q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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
        value = np.max(self.kwargs.get("relative_maximum", 1) * self.computed_props["exists"] * self.props["Strombedarf"])
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.CoolingTower(
            label=self.props["Name"],
            specific_electricity_demand=self.props["Strombedarf"],
            Q_th=fx.Flow(
                label='Qth',
                meta_data=self.meta_data,
                bus=busses[self.props["Wärmebus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"]}
            )
        )

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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

        "Grüne Wärme": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromEinspeisung", str),
    }

    _invest_prop = "Thermische Leistung"

    @property
    def factor_grid_to_invest(self) -> float:
        value = np.max(self.kwargs.get("relative_maximum", 1) * self.computed_props["exists"] / self.computed_props["eta_th"])
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.CHP(
            label=self.props["Name"],
            eta_el=self.computed_props["eta_el"],
            eta_th=self.computed_props["eta_th"],
            Q_th=fx.Flow(
                label='Qth',
                meta_data=self.meta_data,
                bus=busses[self.props["Wärmebus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            P_el=fx.Flow(
                label="Pel",
                bus=busses[self.props["Strombus"]],
                effects_per_flow_hour={
                    effects["costs"]: -self.computed_props["Stromerlöse"],
                    effects["CO2FW"]: -self.computed_props["CO2 Reward Strom"]},
            ),
            Q_fu=fx.Flow(
                label='Qfu',
                bus=busses[self.props["Brennstoff"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Brennstoffkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Brennstoff"] +
                        self.computed_props["CO2 Kosten"],
                    effects["CO2"]: self.computed_props["CO2 Faktor"]
                }
            )
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(comp.Q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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
        "Grüne Wärme": (0, Union[int, float, str]),
        # Connections
        "Wärmebus": ("Fernwaerme", str),
        "Strombus": ("StromEinspeisung", str)
    }

    _invest_prop = "Brennstoff Leistung"

    def validate_properties(self):
        super().validate_properties()
        # Validate Power Points
        electric_power_points = tuple_of_numbers_from_str(
            self.props["Elektrische Leistung (Stützpunkte)"], delimiter='-')
        thermal_power_points = tuple_of_numbers_from_str(
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

        self.computed_props["Elektrische Leistung (Stützpunkte)"] = tuple_of_numbers_from_str(
            self.props["Elektrische Leistung (Stützpunkte)"], delimiter='-')
        self.computed_props["Thermische Leistung (Stützpunkte)"] = tuple_of_numbers_from_str(
            self.props["Thermische Leistung (Stützpunkte)"], delimiter='-')

        self.computed_props["Stromerlöse"] = as_time_series("Strom", time_series_data)

        self.compute_investment(years_of_model)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        flow_heat = fx.Flow('Qth', busses[self.props["Wärmebus"]])
        flow_fuel = fx.Flow('Qfu', busses[self.props["Brennstoff"]],
                            meta_data=self.meta_data,
                            size=self.computed_props[f"Investment {self._invest_prop}"],
                            effects_per_flow_hour={effects["costs"]:
                                                       self.computed_props["Brennstoffkosten"] +
                                                       self.computed_props["Zusatzkosten pro MWh Brennstoff"]},
                            **self.flow_kwargs)
        flow_el = fx.Flow('Pel', busses[self.props["Strombus"]],
                          effects_per_flow_hour={effects["costs"]: -self.computed_props["Stromerlöse"]})

        if self.props["Ausschaltbar"]:
            segmented_conversion_factors = {
                flow_fuel: [(0, 1e-5), (self.computed_props["Brennstoff Leistung"], self.computed_props["Brennstoff Leistung"])],
                flow_el: [(0, 1e-5), self.computed_props["Elektrische Leistung (Stützpunkte)"]],
                flow_heat: [(0, 1e-5), self.computed_props["Thermische Leistung (Stützpunkte)"]]
            }
        else:
            segmented_conversion_factors = {
                flow_fuel: [(self.computed_props["Brennstoff Leistung"], self.computed_props["Brennstoff Leistung"])],
                flow_el: [self.computed_props["Elektrische Leistung (Stützpunkte)"]],
                flow_heat: [self.computed_props["Thermische Leistung (Stützpunkte)"]]
            }


        comp = fx.LinearConverter(
            label=self.props["Name"],
            inputs=[flow_fuel],
            outputs=[flow_heat, flow_el], segmented_conversion_factors=segmented_conversion_factors)
        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(flow_heat, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
        self.flix_comps.append(comp)
        return self.flix_comps


class Waermepumpe(GridFee):
    _property_definitions = {
        **GridFee._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "Zusatzkosten pro MWh Strom": (0, Union[int, float, str]),
        "COP": (None, Optional[Union[int, float, str]]),
        # COP computation
        "Carnot Effizienz": (0.5, Union[float, str]),
        "Quelltemperatur": (None, Optional[Union[int, float, str]]),
        "Zieltemperatur": ("TVL_FWN", Optional[Union[int, float, str]]),
        # BEW Operation Funding
        "SCOP für BEW": (None, Optional[Union[int, float]]),
        "Maximale Stromkostenförderung BEW": (None, Optional[float]),
        # Einsatzbeschränkung
        "Untergrenze für Einsatz": (None, Optional[Union[int, float]]),
        "Zeitreihe für Einsatzbeschränkung": (None, Optional[str]),

        "Grüne Wärme": (0, Union[int, float, str]),

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
                eta=as_time_series(self.props["Carnot Effizienz"], time_series_data))
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
        self.computed_props["Einsatzbeschränkung"] = self.kwargs.pop("relative_maximum", 1)
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.HeatPump(
            label=self.props["Name"],
            COP=self.computed_props["COP"],
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.props["Wärmebus"]],
                meta_data=self.meta_data,
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                relative_maximum=self.computed_props["Einsatzbeschränkung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"],
                    effects["funding"]: self.computed_props["BEW Förderung Strom"]
                }
            )
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(comp.Q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.HeatPumpWithSource(
            label=self.props["Name"],
            COP=self.computed_props["COP"],
            Q_th=fx.Flow(
                label='Qth',
                meta_data=self.meta_data,
                bus=busses[self.props["Wärmebus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                relative_maximum=self.computed_props["Einsatzbeschränkung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"],
                    effects["funding"]: self.computed_props["BEW Förderung Strom"]
                },
            ),
            Q_ab=fx.Flow(
                label='Qab',
                bus=busses[self.props["Abwärmebus"]],
                effects_per_flow_hour={effects["costs"]: self.computed_props["Abwärmekosten"]}
            )
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(comp.Q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:

        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.linear_converters.HeatPumpWithSource(
            label=self.props["Name"],
            COP=self.computed_props["COP"],
            Q_th=fx.Flow(
                label='Qth',
                meta_data=self.meta_data,
                bus=busses[self.props["Wärmebus"]],
                size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                relative_maximum=self.computed_props["Einsatzbeschränkung"],
                can_be_off=self.on_parameters,
                **self.flow_kwargs
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.props["Strombus"]],
                effects_per_flow_hour={
                    effects["costs"]:
                        self.computed_props["Stromkosten"] +
                        self.computed_props["Zusatzkosten pro MWh Strom"],
                    effects["funding"]: self.computed_props["BEW Förderung Strom"]
                },
            ),
            Q_ab=fx.Flow(
                label='Qab',
                bus=busses[self.props["Abwärmebus"]],
            )
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(comp.Q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
        self.flix_comps.append(comp)
        return self.flix_comps


class Abwaerme(EnergySystemObject):
    _property_definitions = {
        **EnergySystemObject._property_definitions,
        "Thermische Leistung": (None, Optional[Union[int, float, str]]),
        "Abwärmekosten": (NO_DEFAULT, Union[int, float, str]),

        "Grüne Wärme": (0, Union[int, float, str]),
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
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        q_th = fx.Flow(label='Qth',
                       meta_data=self.meta_data,
                     bus=busses[self.props["Wärmebus"]],
                     size=self.computed_props[f"Investment {self._invest_prop}"] or self.computed_props["Thermische Leistung"],
                     can_be_off=self.on_parameters,
                **self.flow_kwargs
                     )

        q_abw = fx.Flow(label='Qabw',
                      bus=busses[self.props["Abwärmebus"]],
                      effects_per_flow_hour={effects["costs"]: self.computed_props["Abwärmekosten"]}
                      )

        comp = fx.LinearConverter(
            label=self.props["Name"],
            inputs=[q_abw],
            outputs=[q_th],
            conversion_factors=[{q_abw: 1, q_th: 1}]
        )

        # Allocate Green Heat
        if self.props['Grüne Wärme'] != 0:
            self.computed_props['Grüne Wärme'] = as_time_series(self.props['Grüne Wärme'], time_series_data)
            add_effect_per_flow_hour(q_th, effects["Gruene_Waerme"], effects['costs'], self.computed_props['Grüne Wärme'])

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
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

    def __init__(self, **props):
        super().__init__(**props)
        self.meta_data_storage = {}

    def compute_investment(self, years_of_model: List[int]):
        super().compute_investment(years_of_model)
        self.computed_props[f"Investment {self._invest_prop} 2"] = None
        if self.invest_args_viable:
            size = self.computed_props[self._invest_prop]
            self.computed_props[f"Investment {self._invest_prop} 2"] = fx.InvestParameters(
                fixed_size=size,
                maximum_size=self.computed_props[f"Investment {self._invest_prop}"].maximum_size,
                optional=self.props["Optional"])

            # Add Effect to link installed thermal power of in and out flows
            if not isinstance(size, (int, float)):
                effect = fx.Effect(label=f"{self.props['Name']}_link_power", unit="",
                                 description=f"Links the in and outflow investment value of storage {self.props['Name']}",
                                 minimum_invest=0, maximum_invest=0)
                self.flix_comps.append(effect)
                self.computed_props[f"Investment {self._invest_prop}"].specific_effects[effect] = 1
                self.computed_props[f"Investment {self._invest_prop} 2"].specific_effects = {effect: -1}

    def compute_investment_capacity(self, years_of_model: List[int]):
        self.computed_props["Kapazität [MWh]"], min_invest, max_invest = (
            handle_invest_parameter(self.props["Kapazität [MWh]"]))

        self.computed_props[f"Investment Kapazität [MWh]"] = None
        if self.invest_args_viable:
            self.meta_data_storage["fixed_effects"], self.meta_data_storage["specific_effects"] = costs_and_funding(
                interest_rate=self.props["Zinssatz"],
                starting_year=self.props["Startjahr"],
                lifetime=self.props["Lebensdauer"],
                invest_costs=0, annual_costs=0,
                specific_invest_costs=self.props["Investkosten [€/MWh]"],
                specific_annual_costs=self.props["Sonstige Fixkosten [€/(MWh*a)]"],
                funding_rate=self.props["Fördersatz"],
                years_of_model=years_of_model
            )
            self.computed_props["fixed_effects_capacity"] = {key: sum(value) for key, value in
                                                             self.meta_data_storage["fixed_effects"].items()}
            self.computed_props["specific_effects_capacity"] = {key: sum(value) for key, value in
                                                                self.meta_data_storage["specific_effects"].items()}
            if self.props["Investgruppe Kapazität"]:
                self.computed_props["specific_effects_capacity"][self.props["Investgruppe Kapazität"]] = 1

            size = self.computed_props["Kapazität [MWh]"]
            self.computed_props["Investment Kapazität [MWh]"] = fx.InvestParameters(
                fix_effects={key: value for key, value in self.computed_props["fixed_effects_capacity"].items() if value},
                specific_effects={key: value for key, value in self.computed_props["specific_effects_capacity"].items() if
                               value},
                fixed_size=size if isinstance(size, (int, float)) else None,
                optional=self.props["Optional"],
                minimum_size=min_invest,
                maximum_size=max_invest)


    def computation(self,
                    years_of_model: List[int],
                    co2_factors: Dict[str, float],
                    time_series_data: pd.DataFrame) -> None:
        self.finalize_kwargs(time_series_data)
        # Limiting capcity and Thermische Leistung
        self.computed_props["Einsatzbeschränkung"] = 1
        if self.props["AbhängigkeitVonDT"]:
            self.computed_props["Einsatzbeschränkung"] = (
                as_time_series(self.props["Obere Temperatur"], time_series_data) -
                as_time_series(self.props["Untere Temperatur"], time_series_data)) / 65
        if "relative_maximum" in self.kwargs:
            relative_maximum = self.kwargs.pop("relative_maximum")
            self.computed_props["Einsatzbeschränkung"] = np.where(
                self.computed_props["Einsatzbeschränkung"] < relative_maximum,
                self.computed_props["Einsatzbeschränkung"], relative_maximum)

        self.compute_investment(years_of_model)
        self.compute_investment_capacity(years_of_model)

    def insert_effects_into_investargs(self, effects: Dict[str, fx.Effect]) -> None:
        # Inserting effects as keys
        invest_keys = [f"Investment {self._invest_prop}",
                       f"Investment {self._invest_prop} 2",
                       f"Investment Kapazität [MWh]"]
        for invest_key in invest_keys:
            if self.computed_props[invest_key]:
                insert_effects(self.computed_props[invest_key].fix_effects, effects)
                insert_effects(self.computed_props[invest_key].specific_effects, effects)

    def connect_to_system(self,
                          time_series_data: pd.DataFrame,
                          co2_factors: Dict[str, float],
                          years_of_model: List[int],
                          effects: Dict[str, fx.Effect],
                          busses: Dict[str, fx.Bus]) -> List[Element]:
        self.computation(years_of_model, co2_factors, time_series_data)
        if np.max(self.computed_props["exists"]) == 0:  # Dont add Components which dont exists anyway
            return []

        self.insert_effects_into_investargs(effects)

        comp = fx.Storage(
            label=self.props["Name"],
            meta_data=self.meta_data_storage,
            capacity_in_flow_hours=self.computed_props["Investment Kapazität [MWh]"] or self.computed_props["Kapazität [MWh]"],
            eta_charge=self.props["eta_load"],
            eta_discharge=self.props["eta_unload"],
            relative_loss_per_hour=self.props["VerlustProStunde"],
            relative_maximum_charge_state=self.computed_props["Einsatzbeschränkung"]
            if isinstance(self.computed_props["Einsatzbeschränkung"], (int, float))
            else np.append(self.computed_props["Einsatzbeschränkung"], self.computed_props["Einsatzbeschränkung"][-1]),

            charging=fx.Flow(label='QthLoad',
                             bus=busses["Fernwaerme"],
                             meta_data=self.meta_data,
                             size=self.computed_props["Investment Thermische Leistung"] or self.computed_props["Thermische Leistung"],
                             relative_maximum=self.computed_props["Einsatzbeschränkung"]
                             ),
            discharging=fx.Flow(label='QthUnload',
                                meta_data={'fixed_effects': {}, 'specific_effects': {}},
                                bus=busses["Fernwaerme"],
                                size=self.computed_props["Investment Thermische Leistung 2"] or self.computed_props["Thermische Leistung"],
                                relative_maximum=self.computed_props["Einsatzbeschränkung"],
                                can_be_off=self.on_parameters,
                                **self.flow_kwargs
                          ),
            prevent_simultaneous_charge_and_discharge=True,
        )

        restrict_availlability(comp, self.computed_props['exists'])
        update_meta_data(comp, {'Gruppe': self.props["Gruppe"],
                                'Startjahr': self.props['Startjahr'],
                                'Lebensdauer': self.props['Lebensdauer']})
        validate_invest_meta_data(comp)
        self.flix_comps.append(comp)
        return self.flix_comps


class ComponentFactory:
    def __init__(self,
                 time_series_data: pd.DataFrame,
                 co2_factors: Dict[str, float],
                 years_of_model: List[int],
                 effects: Dict[str, fx.Effect],
                 busses: Dict[str, fx.Bus]):
        self.time_series_data = time_series_data
        self.co2_factors = co2_factors
        self.years_of_model = years_of_model
        self.effects = effects
        self.busses = busses

        self.created_comps: List[EnergySystemObject] = []

    def create_energy_object(self, obj_type: str, object_properties: Dict):
        obj_class = self.get_class_by_type(obj_type)
        if obj_class:
            energy_obj = obj_class(**object_properties)
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


def insert_effects(dictionary: Dict[Union[fx.Effect, str], Any],
                   effects: Dict[str, fx.Effect]) -> None:

    if dictionary is None or dictionary == 0 or dictionary == {}:
        return None
    for effect_name, value in list(dictionary.items()):
        if effect_name in effects.keys():
            dictionary[effects[effect_name]] = dictionary.pop(effect_name)
        elif not isinstance(effect_name, fx.Effect):
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
        starting_year: int,
        lifetime: int,
        years_of_model: List[int],
        invest_costs: float,
        specific_invest_costs: float,
        annual_costs: float,
        specific_annual_costs: float,
        funding_rate: float) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    '''
    Calculates the annual costs and funding for an investment based on various financial parameters.

    This function computes the fixed and specific costs and funding for an investment, considering the
    interest rate, lifetime of the investment, investment costs (both per MW and per year), other costs (both per MW
    and per year), funding rate, and grid fee per MW per year. The costs and funding are calculated using the annuity
    method, which spreads out the initial investment costs over the lifetime of the investment, adjusted for the
    interest rate.

    Parameters:
    - interest_rate (float): The annual interest rate used for calculating the annuity factor.
    - starting_year: first year of operation
    - lifetime (int): lifetime for calculating the investment
    - years_of_model (List[int]): The years used in the model
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
    accounting_years = np.array([1 if starting_year <= year < (starting_year + lifetime) else 0 for year in years_of_model])

    # Calculate costs and funding
    fix_costs = {
        "costs": ((invest_costs * annuity_factor + annual_costs) * accounting_years).tolist(),
        "funding": (invest_costs * annuity_factor * funding_rate * accounting_years).tolist()
    }
    specific_costs = {
        "costs": ((specific_invest_costs * annuity_factor + specific_annual_costs) * accounting_years).tolist(),
        "funding": ((specific_invest_costs * annuity_factor * funding_rate) * accounting_years).tolist()
    }

    def clean_dict(d):
        # Remove keys with lists that are empty or contain only zeros
        keys_to_remove = [key for key, values in d.items() if not values or all(value == 0 for value in values)]
        for key in keys_to_remove:
            del d[key]
        """
        # TODO: Maybe do this later on
        # Check if the dictionary is now empty or all remaining lists are empty or contain only zeros
        if not d or all(not values or all(value == 0 for value in values) for values in d.values()):
            return None
        """
        return d

    return clean_dict(fix_costs), clean_dict(specific_costs)

def handle_invest_parameter(invest_parameter: Union[int, float, str, type(None)]
                            ) -> Tuple[Optional[Union[int, float]], float, float]:
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


def tuple_of_numbers_from_str(input_string: str, delimiter='-') -> Tuple[float, ...]:
    '''
    This function was written to extract numbers from a string
    ----------
    Returns
    -------
    Tuple[float, ...]
    '''
    input_string = input_string.replace(',', '.')
    return tuple([float(i) for i in input_string.split(delimiter)])


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


def add_effect_per_flow_hour(flow: fx.Flow, effect: fx.Effect, standard_effect: fx.Effect, factor: Union[float, np.ndarray]):
    if isinstance(flow.effects_per_flow_hour, dict):
        flow.effects_per_flow_hour.update({effect: factor})
    elif flow.effects_per_flow_hour is None:
        flow.effects_per_flow_hour = {effect: factor}
    else:
        flow.effects_per_flow_hour = {effect: factor, standard_effect: flow.effects_per_flow_hour}

def restrict_availlability(component: flixOpt.elements.Component, exists: Union[int, float, np.ndarray]) -> flixOpt.elements.Component:
    for flow in component.inputs + component.outputs:
        flow.relative_maximum = flow.relative_maximum * exists
        flow.relative_minimum = flow.relative_minimum * exists
    if isinstance(component, fx.Storage):
        storage_exists_exists = exists if isinstance(exists, (int, float)) else np.append(exists, exists[-1])
        component.relative_maximum_charge_state = component.relative_maximum_charge_state * storage_exists_exists
        component.relative_minimum_charge_state = component.relative_minimum_charge_state * storage_exists_exists
    return component

def update_meta_data(component: flixOpt.elements.Component, meta_data: Dict):
    if component.meta_data is None:
        component.meta_data = {}
    component.meta_data.update(meta_data)
    for flow in component.inputs + component.outputs:
        if flow.meta_data is None:
            flow.meta_data = {}
        flow.meta_data.update(meta_data)
    return component

def validate_invest_meta_data(component: flixOpt.elements.Component):
    for flow in component.inputs + component.outputs:
        if isinstance(flow.size, fx.InvestParameters):
            fix_effects = flow.size.fix_effects or {}  # Making sure its a dict
            for effect, value in fix_effects.items():
                if effect.label not in flow.meta_data['fixed_effects']:
                    logger.warning('Effect {effect.label} not found in fixed_effects meta_data')
                elif abs(abs(sum(flow.meta_data['fixed_effects'][effect.label])) - abs(value)) >= 1e-5:
                    logger.critical(
                        f'The meta_data for {flow.label_full=} is not correct for the investment effects per period.'
                        f'The total {effect.label=} passed to the InvestParameters is {value}. '
                        f'The meta_data is {flow.meta_data["fixed_effects"][effect.label]}, '
                        f'which totals to {sum(flow.meta_data["fixed_effects"][effect.label])}')

            specific_effects = flow.size.specific_effects or {}  # Making sure its a dict
            for effect, value in specific_effects.items():
                if effect.label not in flow.meta_data['specific_effects']:
                    logger.warning(f'Effect {effect.label} not found in specific_effects meta_data. {value=}')
                elif abs(abs(sum(flow.meta_data['specific_effects'][effect.label])) - abs(value)) >= 1e-5:
                    logger.critical(
                        f'The meta_data for {flow.label_full=} is not correct for the investment effects per period.'
                        f'The total {effect.label=} passed to the InvestParameters is {value}. '
                        f'The meta_data is {flow.meta_data["specific_effects"][effect.label]}, '
                        f'which totals to {sum(flow.meta_data["specific_effects"][effect.label])}')