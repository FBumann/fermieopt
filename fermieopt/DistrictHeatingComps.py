import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import flixOpt as fx
import flixOpt.elements
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator, ValidationError

from fermieopt.meta_data import MetaData, MetaDataFactory

logger = logging.getLogger('flixOpt')


class Element(
    BaseModel,
    populate_by_name=True,  # Enables using both field names and aliases
    extra='forbid',
):  # Forbids unexpected keys in input data
    name: str = Field(alias='Name')
    group: Optional[str] = Field(alias='Gruppe', default=None)
    _meta_data: MetaData = PrivateAttr(default_factory=MetaDataFactory.create)

    def add_to_flow_system(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float] = None,
        years_of_model: List[int] = None,
    ):
        self._insert_data(time_series_data)
        flow_system.add_elements(
            self._convert_to_flixopt(flow_system, busses, time_series_data, co2_factors, years_of_model)
        )

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        raise NotImplementedError

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        raise NotImplementedError


class InvestElement(Element):
    start_year: Optional[int] = Field(alias='Startjahr', default=None, ge=1800)
    amortization_time: Optional[int] = Field(alias='Abschreibungsdauer', default=None, ge=1)
    lifetime: Optional[int] = Field(alias='Lebensdauer', default=None, ge=1)
    optional: bool = Field(alias='Optional', default=False)
    invest_costs_fixed: Union[int, float] = Field(alias='Investkosten (fix) [€]', default=0)
    invest_costs_specific: Union[int, float] = Field(alias='Investkosten (spezifisch) [€/MW]', default=0)
    annual_costs_fixed: Union[int, float] = Field(alias='Sonstige Fixkosten (fix) [€/a]', default=0)
    annual_costs_specific: Union[int, float] = Field(alias='Sonstige Fixkosten (spezifisch) [€/(MW*a)]', default=0)
    interest_rate: Union[int, float] = Field(alias='Zinssatz', default=0)
    funding_rate: Union[int, float] = Field(alias='Fördersatz', default=0)
    invest_group: Optional[str] = Field(alias='Investgruppe', default=None)

    @property
    def needs_investment(self) -> bool:
        return self.invest_costs_fixed != 0 or self.invest_costs_specific != 0 or self.optional is True

    @field_validator('invest_group', mode='before')
    @classmethod
    def validate_invest_group(cls, value) -> Optional[str]:
        if value is None:
            return value
        elif isinstance(value, str):
            value = value.replace(',', '.')  # allow , as decimal separator
            values = value.split(':')
            if len(values) != 2:
                raise ValidationError(f'Invalid invest group: {value}. Must be None or of form: "xyz:0.5"')
            name, _ = values
            return name
        raise ValidationError(f'Invalid invest group: {value}. Must be None or of form: "xyz:0.5"')

    @model_validator(mode='after')
    def validate_years(self):
        """Validates the start and lifetime of the element"""
        if not (self.start_year is None) == (self.lifetime is None):
            raise ValueError("Either set BOTH or NONE of 'Startjahr' and 'Lebensdauer'!")
        if self.lifetime is not None and self.amortization_time is None:
            self.amortization_time = self.lifetime
            logger.debug(f'Amortization time of {self.name} was set to {self.lifetime} years, as no amortization time was given')
        return self

    @staticmethod
    def annuity_factor(interest_rate: float, duration_in_years: int) -> float:
        """Get the annuity factor for a given interest rate and lifetime"""
        if interest_rate == 0:  # Preventing ZeroDivision
            annuity_factor = 1 / duration_in_years
        else:
            annuity_factor = ((1 + interest_rate) ** duration_in_years * interest_rate) / ((1 + interest_rate) ** duration_in_years - 1)
        return annuity_factor

    @staticmethod
    def costs_and_funding(
        interest_rate: float,
        starting_year: int,
        amortization_time: int,
        lifetime: int,
        years_of_model: List[int],
        invest_costs: float,
        specific_invest_costs: float,
        annual_costs: float,
        specific_annual_costs: float,
        funding_rate: float,
    ) -> Tuple[Dict[str, np.ndarray[float]], Dict[str, np.ndarray[float]]]:
        """
        Calculates the annual costs and funding for an investment based on various financial parameters.

        This function computes the fixed and specific costs and funding for an investment, considering the
        interest rate, lifetime of the investment, investment costs (both per MW and per year), other costs (both per MW
        and per year), funding rate, and grid fee per MW per year. The costs and funding are calculated using the annuity
        method, which spreads out the initial investment costs over the lifetime of the investment, adjusted for the
        interest rate.

        Parameters:
        - interest_rate (float): The annual interest rate used for calculating the annuity factor.
        - starting_year (int): first year of operation
        - amortization_time (int): amortization time for calculating the investment
        - lifetime (int): lifetime for calculating the fixed yearly costs
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
        """
        annuity_factor = InvestElement.annuity_factor(interest_rate=interest_rate, duration_in_years=amortization_time)

        operation_years = np.array(
            [1 if starting_year <= year < (starting_year + lifetime) else 0 for year in years_of_model]
        )
        amortization_years = np.array(
            [1 if starting_year <= year < (starting_year + amortization_time) else 0 for year in years_of_model]
        )

        # Calculate costs and funding
        fix_costs = {
            'costs': invest_costs * annuity_factor * amortization_years +
                     annual_costs * operation_years,
            'funding': invest_costs * annuity_factor * amortization_years * funding_rate,
        }
        specific_costs = {
            'costs': specific_invest_costs * annuity_factor * amortization_years +
                     specific_annual_costs* operation_years,
            'funding': specific_invest_costs * annuity_factor * amortization_years * funding_rate,
        }

        def clean_dict(d):
            # Remove keys with lists or arrays that are empty or contain only zeros
            keys_to_remove = [key for key, values in d.items()
                              if values is None or np.all(values == 0)]
            for key in keys_to_remove:
                del d[key]

            return d

        return clean_dict(fix_costs), clean_dict(specific_costs)

    def insert_size(
        self,
        flow: fx.Flow,
        size: Union[int, float, Tuple[Union[int, float], Union[int, float]]],
        effects: Dict[str, fx.Effect],
        years_of_model: List[int],
    ) -> None:
        if not self.needs_investment:
            flow.size = size
        else:
            fixed_effects_per_period, specific_effects_per_period = self.costs_and_funding(
                interest_rate=self.interest_rate,
                starting_year=self.start_year,
                amortization_time=self.amortization_time,
                lifetime=self.lifetime,
                years_of_model=years_of_model,
                invest_costs=self.invest_costs_fixed,
                specific_invest_costs=self.invest_costs_specific,
                annual_costs=self.annual_costs_fixed,
                specific_annual_costs=self.annual_costs_specific,
                funding_rate=self.funding_rate,
            )

            fixed_effects_total = {
                effects[effect]: np.sum(values) for effect, values in fixed_effects_per_period.items()
            }
            specific_effects_total = {
                effects[effect]: np.sum(values) for effect, values in specific_effects_per_period.items()
            }
            if self.invest_group is not None:
                specific_effects_total[effects[self.invest_group]] = 1

            flow.size = fx.InvestParameters(
                optional=self.optional,
                fixed_size=size if isinstance(size, (int, float)) else None,
                minimum_size=0 if isinstance(size, (int, float)) else size[0],
                maximum_size=None if isinstance(size, (int, float)) else size[1],
                fix_effects=fixed_effects_total,
                specific_effects=specific_effects_total,
            )
            if not flow.meta_data:
                flow.meta_data = MetaDataFactory.create()

            flow.meta_data['invest']['costs']['fixed_effects'] += fixed_effects_per_period.get('costs', 0)
            flow.meta_data['invest']['costs']['specific_effects'] += specific_effects_per_period.get('costs', 0)
            flow.meta_data['invest']['funding']['fixed_effects'] += fixed_effects_per_period.get('funding', 0)
            flow.meta_data['invest']['funding']['specific_effects'] += specific_effects_per_period.get('funding', 0)

    def restrict_availlability(self, component: flixOpt.elements.Component, years_in_model: List[int]) -> None:
        existance = exists(self.start_year, self.lifetime, years_in_model)
        restrict_availlability(component, existance)


class PowerInvestElement(InvestElement):
    power: Union[int, float, Tuple[Union[int, float], Union[int, float]]] = Field(alias='Nennleistung [MW]')
    fixed_profile: Optional[str] = Field(alias='Festes Profil', default=None)

    def _insert_data(self, data: pd.DataFrame):
        self.fixed_profile = extract_data(self.fixed_profile, data)

    @field_validator('power', mode='before')
    @classmethod
    def validate_power(cls, value) -> Union[int, float, Tuple[Union[int, float], Union[int, float]]]:
        return validate_invest_range(value, label='Nennleistung [MW]')


class ThermalInvestElement(InvestElement):
    thermal_power: Union[int, float, str] = Field(alias='Thermische Leistung [MW]')
    grid_fee_per_year: Union[float, str] = Field(alias='Netzentgelt [€/(MW*a)]', default=0)
    bus_heat: str = Field(alias='Wärmebus', default='Fernwärme')

    costs_per_mwh_heat_extra: Union[int, float, str] = Field(
        alias='Zusätzliche Wärmeerzeugungskosten [€/MWh]', default=0
    )
    relative_maximum: Union[int, float, str] = Field(alias='Relative thermische Leistungsobergrenze', default=1)
    relative_minimum: Union[int, float, str] = Field(alias='Relative thermische Leistungsuntergrenze', default=0)
    green_heat_factor: Union[int, float, str] = Field(alias='Grüne Wärme', default=0)

    def _insert_data(self, data: pd.DataFrame):
        self.costs_per_mwh_heat_extra = extract_data(self.costs_per_mwh_heat_extra, data)
        self.relative_maximum = extract_data(self.relative_maximum, data)
        self.relative_minimum = extract_data(self.relative_minimum, data)
        self.green_heat_factor = extract_data(self.green_heat_factor, data)

    def thermal_effects_per_flow_hour(self,
                                      effects: Dict[str, fx.Effect],
                                      ) -> Dict[fx.Effect, Union[int, float, np.ndarray]]:
        """Calculates the thermal_effects per flow_hour."""
        data = {effects['Gruene_Waerme']: self.green_heat_factor,
                effects['costs']: self.costs_per_mwh_heat_extra}
        return {effect: value for effect, value in data.items() if np.sum(value) not in [0, None]}

    @staticmethod
    def insert_grid_fee(
        grid_fee: Union[int, float], invest_flow: fx.Flow, efficiency: Union[int, float, np.ndarray], effect: fx.Effect
    ) -> None:
        """Adds the grid fee to the investment parameters of the invest_flow and it's meta_data."""
        if grid_fee == 0:
            return None
        if grid_fee != 0 and not isinstance(invest_flow.size, fx.InvestParameters):
            raise Exception('There are no InvestParameters to add the grid_fee to')
        else:
            highest_possible_grid_draw = np.max(invest_flow.relative_maximum / efficiency)
            yearly_grid_fee = grid_fee * highest_possible_grid_draw
            if invest_flow.size.specific_effects is None:
                invest_flow.size.specific_effects = {effect: yearly_grid_fee}
            else:
                invest_flow.size.specific_effects[effect] = yearly_grid_fee + invest_flow.size.specific_effects.get(
                    effect, 0
                )

            assert effect.label == 'costs', f"Effect {effect.label} is not 'costs', which is expected in this function"
            if not invest_flow.meta_data:
                invest_flow.meta_data = MetaDataFactory.create()
            invest_flow.meta_data['invest']['costs']['specific_effects'] += yearly_grid_fee
            invest_flow.meta_data['yearly_grid_fee_per_thermal_power'] = yearly_grid_fee
            invest_flow.meta_data['highest_possible_grid_draw'] = highest_possible_grid_draw

    @property
    def needs_investment(self) -> bool:
        return super().needs_investment or self.grid_fee_per_year != 0

    @field_validator('thermal_power', mode='before')
    @classmethod
    def validate_thermal_power(cls, value) -> Union[int, float, Tuple[Union[int, float], Union[int, float]]]:
        return validate_invest_range(value, label='Thermische Leistung [MW]')


class Sink(PowerInvestElement):
    bus: str = Field(alias='Bus')
    flow_label: str = Field(alias='Flowname', default='sink')

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        comp = fx.Sink(
            label=self.name,
            sink=fx.Flow(
                label=self.flow_label, size=1, bus=busses[self.bus], fixed_relative_profile=self.fixed_profile
            ),
        )
        self.insert_size(
            comp.sink,
            self.power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(comp, years_of_model)
        return comp


class Source(PowerInvestElement):
    bus: str = Field(alias='Bus')
    flow_label: str = Field(alias='Flowname', default='source')

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects
        comp = fx.Source(
            label=self.name,
            source=fx.Flow(label=self.flow_label, bus=busses[self.bus], fixed_relative_profile=self.fixed_profile),
        )

        self.insert_size(
            comp.source,
            self.power if self.power is not None else (self.minimum_power, self.maximum_power),
            effects,
            years_of_model,
        )
        self.restrict_availlability(comp, years_of_model)
        return comp


class LinearTransformer(PowerInvestElement):
    efficiency: Union[int, float, str] = Field(alias='Wirkungsgrad')
    bus_in: str = Field(alias='Von Bus')
    bus_out: str = Field(alias='Zu Bus')
    flow_label_in: str = Field(alias='Flowname in', default='in')
    flow_label_out: str = Field(alias='Flowname out', default='out')
    cost_per_mwh_in: Union[int, float, str] = Field(alias='Kosten pro MWh von Bus', default=0)

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.efficiency = extract_data(self.efficiency, data)
        self.cost_per_mwh_in = extract_data(self.cost_per_mwh_in, data)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects
        flow_out = fx.Flow(label=self.flow_label_out, bus=busses[self.bus_out], fixed_relative_profile=self.fixed_profile)

        flow_in = fx.Flow(
            label=self.flow_label_in,
            bus=busses[self.bus_in],
            effects_per_flow_hour={effects['costs']: self.cost_per_mwh_in},
        )

        comp = fx.LinearConverter(
            label=self.name,
            inputs=[flow_in],
            outputs=[flow_out],
            conversion_factors=[{flow_in: self.efficiency, flow_out: 1}],
        )
        self.insert_size(
            flow_out,
            self.power if self.power is not None else (self.minimum_power, self.maximum_power),
            flow_system.effect_collection.effects,
            years_of_model,
        )
        self.restrict_availlability(comp, years_of_model)
        return comp


class FuelThermalInvestElement(ThermalInvestElement):
    eta_thermal: Union[float, str] = Field(alias='Thermischer Wirkungsgrad')
    fuel_type: str = Field(alias='Brennstoff')
    fuel_cost_extra: Union[float, str] = Field(alias='Brennstoffkosten Zusatz [€/MWh_hu]', default=0)
    _fuel_costs: Union[float, np.ndarray] = 0

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.eta_thermal = extract_data(self.eta_thermal, data)
        self.fuel_cost_extra = extract_data(self.fuel_cost_extra, data)
        self._fuel_costs = extract_data(self.fuel_type, data)

    def co2_factor(self, time_series_data: pd.DataFrame, co2_factors: Dict[str, float]) -> float:
        return extract_data(co2_factors.get(self.fuel_type, 0), time_series_data)

    def fuel_effects_per_flow_hour(self,
                                      effects: Dict[str, fx.Effect],
                                      time_series_data: pd.DataFrame,
                                      co2_factors: Dict[str, float]
                                      ) -> Dict[fx.Effect, Union[int, float, np.ndarray]]:
        """Calculates the thermal_effects per flow_hour."""
        data = {effects['costs']: (
                        self._fuel_costs
                        + self.fuel_cost_extra
                        + (self.co2_factor(time_series_data, co2_factors) * extract_data('CO2', time_series_data))
                ),
                effects['CO2']: self.co2_factor(time_series_data, co2_factors),
                effects['Gruene_Waerme']: self.green_heat_factor
        }

        return {effect: value for effect, value in data.items() if np.sum(value) not in [0, None]}


class Kessel(FuelThermalInvestElement):
    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        boiler = fx.linear_converters.Boiler(
            label=self.name,
            eta=self.eta_thermal,
            Q_fu=fx.Flow(
                label='Qfu',
                bus=busses[self.fuel_type],
                effects_per_flow_hour=self.fuel_effects_per_flow_hour(effects, time_series_data, co2_factors),
            ),
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
        )
        self.insert_size(
            boiler.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(boiler, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, boiler.Q_th, boiler.eta, effects['costs'])
        return boiler


class KWK(FuelThermalInvestElement):
    eta_el: Union[int, float, str] = Field(alias='Elektrischer Wirkungsgrad')
    forward_flow_temperature: Union[int, float, str] = Field(
        alias='Vorlauftemperatur', default='Vorlauftemperatur Fernwärmenetz [°C]'
    )
    reverse_flow_temperature: Union[int, float, str] = Field(
        alias='Rücklauftemperatur', default='Rücklauftemperatur Fernwärmenetz [°C]'
    )
    ambient_temperature: Union[int, float, str] = Field(alias='Umgebungstemperatur', default='Tamb')

    bus_elec: str = Field(alias='Strombus', default='StromEinspeisung')

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        chp = fx.linear_converters.CHP(
            label=self.name,
            eta_th=self.eta_thermal,
            eta_el=self.eta_el,
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                relative_minimum=self.relative_minimum,
                relative_maximum=self.relative_maximum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.bus_elec],
                effects_per_flow_hour={
                    effects['costs']: -1 * extract_data('Strom', time_series_data),
                    effects['CO2FW']: -1 * self.co2_emissions_electricity(time_series_data, co2_factors),
                },
            ),
            Q_fu=fx.Flow(
                label='Qfu',
                bus=busses[self.fuel_type],
                effects_per_flow_hour=self.fuel_effects_per_flow_hour(effects, time_series_data, co2_factors),
            ),
        )
        self.insert_size(
            chp.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(chp, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, chp.Q_th, chp.eta_th, effects['costs'])
        return chp

    def _insert_data(self, time_series_data: pd.DataFrame):
        super()._insert_data(time_series_data)
        self.eta_el = extract_data(self.eta_el, time_series_data)
        self.forward_flow_temperature = extract_data(self.forward_flow_temperature, time_series_data)
        self.reverse_flow_temperature = extract_data(self.reverse_flow_temperature, time_series_data)
        self.ambient_temperature = extract_data(self.ambient_temperature, time_series_data)

    def co2_emissions_electricity(self, time_series_data: pd.DataFrame, co2_factors: Dict[str, float]) -> np.ndarray:
        try:
            fuel_factor_electricity = self.fuel_factor_for_electrical_energy(
                electrical_efficiency=self.eta_el,
                thermal_efficiency=self.eta_thermal,
                inferior_temperature=self.ambient_temperature,
                forward_flow_temperature=self.forward_flow_temperature,
                reverse_flow_temperature=self.reverse_flow_temperature,
            )
        except KeyError:
            logger.warning(
                'Computation of CO2 Reward did not work properly. Using default values instead. '
                'Optimization itself is not affected. Only take care interpreting CO2 Emissions'
            )
            fuel_factor_electricity = self.fuel_factor_for_electrical_energy(
                electrical_efficiency=self.eta_el,
                thermal_efficiency=self.eta_thermal,
            )
        return fuel_factor_electricity * self.co2_factor(time_series_data, co2_factors)

    @staticmethod
    def fuel_factor_for_electrical_energy(
        electrical_efficiency: Union[int, float, np.ndarray],
        thermal_efficiency: Union[int, float, np.ndarray],
        inferior_temperature: Union[int, float, np.ndarray] = 20,
        forward_flow_temperature: Union[int, float, np.ndarray] = 120,
        reverse_flow_temperature: Union[int, float, np.ndarray] = 60,
    ) -> np.ndarray:
        """
        Using the carnot mehtod, the fuel factor for electrical energy in a heating network is calculated
        https://en.wikipedia.org/wiki/Carnot_method
        """
        inferior_temperature = inferior_temperature + 273.15
        forward_flow_temperature = forward_flow_temperature + 273.15
        reverse_flow_temperature = reverse_flow_temperature + 273.15
        superior_temperature = (forward_flow_temperature - reverse_flow_temperature) / np.log(
            (forward_flow_temperature / reverse_flow_temperature)
        )
        n_carnot = 1 - (inferior_temperature / superior_temperature)

        a_el = (1 * electrical_efficiency) / (electrical_efficiency + n_carnot * thermal_efficiency)
        return a_el / electrical_efficiency


class Waermepumpe(ThermalInvestElement):
    cop: Optional[Union[int, float, str]] = Field(alias='COP', default=None)
    carnot_efficiency: Optional[Union[int, float, str]] = Field(alias='Carnot Effizienz', default=0.5)
    source_temperature: Union[int, float, str] = Field(alias='Quelltemperatur', default=None)
    sink_temperature: Union[int, float, str] = Field(
        alias='Zieltemperatur', default='Vorlauftemperatur Fernwärmenetz [°C]'
    )

    extra_costs_per_mwh_elec: Union[int, float, str] = Field(alias='Stromkosten Zusatz [€/MWh]', default=0)

    scop_bew: Optional[Union[int, float]] = Field(alias='SCOP für BEW', default=None)
    max_bew_elec_funding: Optional[Union[int, float]] = Field(alias='Maximale Stromkostenförderung BEW', default=None)

    bus_elec: str = Field(alias='Strombus', default='StromBezug')

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        heat_pump = fx.linear_converters.HeatPump(
            label=self.name,
            COP=self._get_cop(time_series_data),
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.bus_elec],
                effects_per_flow_hour=self._electricity_effects_per_flow_hour(effects, time_series_data, years_of_model),
            ),
        )
        self.insert_size(
            heat_pump.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(heat_pump, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, heat_pump.Q_th, heat_pump.COP, effects['costs'])
        return heat_pump

    def _get_cop(self, time_series_data: pd.DataFrame) -> Union[float, np.ndarray]:
        if self.cop is not None:
            return extract_data(self.cop, time_series_data)
        else:
            return self.calculate_cop(
                source_temperature=extract_data(self.source_temperature, time_series_data),
                target_temperature=extract_data(self.sink_temperature, time_series_data),
                eta=self.carnot_efficiency,
            )

    def _get_electricity_costs_per_mwh(self, time_series_data: pd.DataFrame) -> Union[float, np.ndarray]:
        return extract_data('Strom', time_series_data) + extract_data(self.extra_costs_per_mwh_elec, time_series_data)

    def _get_operation_funding_bew(
        self, time_series_data: pd.DataFrame, years_of_model: List[int]
    ) -> Union[float, np.ndarray]:
        if self.scop_bew is None:
            return 0
        fund_per_mw_el = self.bew_operation_funding_from_scop(self.scop_bew, 'MWh_el')

        electricity_costs_per_flow_hour = self._get_electricity_costs_per_mwh(time_series_data)
        # Begrenzung der Förderung auf x% der Stromkosten
        fund_per_mw_el = np.where(
            fund_per_mw_el < electricity_costs_per_flow_hour * self.max_bew_elec_funding,
            fund_per_mw_el,
            electricity_costs_per_flow_hour * self.max_bew_elec_funding,
        )
        # Begrenzung auf 10 Jahre
        return fund_per_mw_el * exists(self.start_year, 10, years_of_model)

    def _electricity_effects_per_flow_hour(self,
                                      effects: Dict[str, fx.Effect],
                                      time_series_data: pd.DataFrame,
                                      years_of_model: List[int]
                                      ) -> Dict[fx.Effect, Union[int, float, np.ndarray]]:
        """Calculates the electricity_effects per flow_hour."""

        data = {effects['costs']: self._get_electricity_costs_per_mwh(time_series_data),
                effects['funding']: self._get_operation_funding_bew(time_series_data, years_of_model),
                }
        return {effect: value for effect, value in data.items() if np.sum(value) not in [0, None]}

    @staticmethod
    def bew_operation_funding_from_scop(
        scop: Union[int, float], unit: Literal['MWh_amb', 'MWh_th', 'MWh_el'] = 'MWh_amb'
    ) -> Union[int, float]:
        """
        Calclulated the maximum funding according to the BEW.
        Parameters
        ----------
        scop: assumed scop (seasonal coefficent of Performance) or cop

        Returns
        -------
        Funding in euro per MWh_amb

        """
        if unit not in ['MWh_amb', 'MWh_th', 'MWh_el']:
            raise Exception(f'Not a valid unit. Choose from: {["MWh_amb", "MWh_th", "MWh_el"]}')

        value = (5.5 - (6.8 - 17 / scop) * 0.75) * (scop / (scop - 1))  # ct/kWh
        fund_amb = value * 10  # €/MWh_amb
        if fund_amb >= 92:  # Funding is limited to 92 €/MWh_amb
            fund_amb = 92

        if unit == 'MWh_amb':
            return fund_amb
        elif unit == 'MWh_th':
            return fund_amb * ((scop - 1) / scop)
        else:
            return fund_amb * (scop - 1)

    @staticmethod
    def calculate_cop(source_temperature: np.ndarray, target_temperature: np.ndarray, eta: float = 0.5) -> np.ndarray:
        """
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

        """
        # Celsius zu Kelvin
        source_temperature = source_temperature + 273.15
        target_temperature = target_temperature + 273.15
        return (target_temperature / (target_temperature - source_temperature)) * eta

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.cop = extract_data(self.cop, data)
        self.source_temperature = extract_data(self.source_temperature, data)
        self.sink_temperature = extract_data(self.sink_temperature, data)

        self.extra_costs_per_mwh_elec = extract_data(self.extra_costs_per_mwh_elec, data)

    @model_validator(mode='after')
    def validate_cop(self):
        if not self.cop and (not self.source_temperature or not self.sink_temperature):
            raise Exception(
                f"Need to specify a 'COP' for {self.name} or "
                f"use 'Quelltemperatur' and 'Zieltemperatur' to calculate the COP internally."
            )
        if self.cop and (self.source_temperature and self.sink_temperature):
            raise Exception(
                f"Either specify a 'COP' for {self.name} "
                f"OR use 'Quelltemperatur' and 'Zieltemperatur' to calculate the COP internally."
            )
        return self

    @model_validator(mode='after')
    def validate_bew(self):
        if self.scop_bew:
            if not self.start_year:
                raise Exception(
                    f'Need to specify a Year of Operation start for {self.name} to use HeatPump '
                    f'operation funding, because its limited to 10 years.'
                )
            if not self.max_bew_elec_funding:
                raise Exception(
                    f"Need to specify 'Maximale Stromkostenförderung BEW' for {self.name} to use HeatPump "
                    f'operation funding.'
                )
        return self


class Speicher(ThermalInvestElement):
    capacity: Union[int, float, str] = Field(alias='Kapazität [MWh]')
    invest_costs_capacity_specific: Union[int, float] = Field(alias='Investkosten [€/MWh]', default=0)
    annual_costs_capacity_specific: Union[int, float] = Field(alias='Sonstige Fixkosten (fix) [€/(MWh*a)]', default=0)

    eta_load: Union[int, float, str] = Field(alias='eta_load')
    eta_unload: Union[int, float, str] = Field(alias='eta_unload')
    loss_per_hour: Union[int, float, str] = Field(alias='VerlustProStunde', default=0)

    depends_on_temperature: bool = Field(alias='AbhängigkeitVonDT', default=False)
    temperature_lower: Union[int, float, str] = Field(
        alias='Untere Temperatur', default='Rücklauftemperatur Fernwärmenetz [°C]'
    )
    temperature_upper: Union[int, float, str] = Field(
        alias='Obere Temperatur', default='Vorlauftemperatur Fernwärmenetz [°C]'
    )

    default_temperature_spread: Union[int, float] = Field(alias='Nenn-Temperaturspreizung', default=65)

    def _insert_data(self, time_series_data: pd.DataFrame):
        super()._insert_data(time_series_data)
        self.eta_load = extract_data(self.eta_load, time_series_data)
        self.eta_unload = extract_data(self.eta_unload, time_series_data)
        self.loss_per_hour = extract_data(self.loss_per_hour, time_series_data)
        self.temperature_lower = extract_data(self.temperature_lower, time_series_data)
        self.temperature_upper = extract_data(self.temperature_upper, time_series_data)

        self.relative_maximum = self.relative_maximum * self._get_normalized_temperature_spread()

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        storage = fx.Storage(
            label=self.name,
            capacity_in_flow_hours=0,
            eta_charge=self.eta_load,
            eta_discharge=self.eta_unload,
            relative_loss_per_hour=self.loss_per_hour,
            relative_maximum_charge_state=self.relative_maximum_capacity,
            charging=fx.Flow(
                label='charging',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
            discharging=fx.Flow(
                label='discharging',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
            ),
            prevent_simultaneous_charge_and_discharge=True,
        )
        self.insert_size(
            storage.charging,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(storage, years_of_model)
        self.link_second_flow_size(storage.charging, storage.discharging, flow_system)
        self.insert_capacity(storage, effects, years_of_model)
        return storage

    def link_second_flow_size(self, flow_with_size: fx.Flow, flow_to_link: fx.Flow, flow_system: fx.FlowSystem) -> None:
        """
        Links the size of the second flow to the size of the first flow. If needed, a new Effect is added to the FlowSystem
        """
        if isinstance(flow_with_size.size, (int, float)):
            flow_to_link.size = flow_with_size.size
            return None
        elif isinstance(flow_with_size.size, fx.InvestParameters):
            flow_to_link.size = fx.InvestParameters(
                optional=flow_with_size.size.optional,
                fixed_size=flow_with_size.size.fixed_size,
                minimum_size=flow_with_size.size.minimum_size,
                maximum_size=flow_with_size.size.maximum_size,
            )

            if flow_with_size.size.fixed_size is None:
                effect = fx.Effect(
                    label=f'{self.name}_link_thermal_power',
                    unit='',
                    description=f'Links the charge and discharge investment value of storage {self.name}',
                    minimum_invest=0,
                    maximum_invest=0,
                )
                flow_system.add_effects(effect)

                flow_with_size.size.specific_effects[effect] = 1
                flow_to_link.size.specific_effects = {effect: -1}

    def insert_capacity(self, storage: fx.Storage, effects: [str, fx.Effect], years_of_model: List[int]) -> None:
        if not self.needs_investment_capacity:
            storage.capacity_in_flow_hours = self.capacity
        else:
            _, specific_effects_per_period = self.costs_and_funding(
                interest_rate=self.interest_rate,
                starting_year=self.start_year,
                amortization_time=self.amortization_time,
                lifetime=self.lifetime,
                years_of_model=years_of_model,
                invest_costs=0,
                annual_costs=0,
                specific_invest_costs=self.invest_costs_capacity_specific,
                specific_annual_costs=self.annual_costs_capacity_specific,
                funding_rate=self.funding_rate,
            )

            specific_effects_total = {
                effects[effect]: np.sum(values) for effect, values in specific_effects_per_period.items()
            }

            storage.capacity_in_flow_hours = fx.InvestParameters(
                optional=self.optional,
                fixed_size=self.capacity if isinstance(self.capacity, (int, float)) else None,
                minimum_size=self.minimum_capacity,
                maximum_size=self.maximum_capacity,
                specific_effects=specific_effects_total,
            )
            if not storage.meta_data:
                storage.meta_data = MetaDataFactory.create()

            storage.meta_data['invest']['costs']['specific_effects'] += specific_effects_per_period.get('costs', 0)
            storage.meta_data['invest']['funding']['specific_effects'] += specific_effects_per_period.get('funding', 0)

    def _get_normalized_temperature_spread(self) -> Union[float, np.ndarray]:
        return (self.temperature_upper - self.temperature_lower) / self.default_temperature_spread

    @field_validator('grid_fee_per_year')
    @classmethod
    def validate_grid_fee(cls, value):
        if value is not None:
            raise ValueError(f"Netzentgelt is not supported for '{cls.__name__}")
        return value

    @property
    def minimum_capacity(self) -> float:
        return float(self.capacity.split('-')[0]) if isinstance(self.capacity, str) else 0

    @property
    def maximum_capacity(self):
        return float(self.capacity.split('-')[1]) if isinstance(self.capacity, str) else None

    @property
    def relative_maximum_capacity(self) -> Union[float, np.ndarray]:
        if isinstance(self.relative_maximum, (int, float)):
            return self.relative_maximum
        else:  # Append the last value to the array
            return np.concatenate((self.relative_maximum, np.array([self.relative_maximum[-1]])))

    @property
    def needs_investment_capacity(self) -> bool:
        return (
            self.invest_costs_capacity_specific != 0
            or self.annual_costs_capacity_specific != 0
            or self.optional is True
        )


class EHK(ThermalInvestElement):
    eta_thermal: Union[float, str] = Field(alias='Thermischer Wirkungsgrad')
    extra_costs_per_mwh_elec: Union[int, float, str] = Field(alias='Stromkosten Zusatz [€/MWh]', default=0)
    bus_elec: str = Field(alias='Strombus', default='StromBezug')

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.eta_thermal = extract_data(self.eta_thermal, data)
        self.extra_costs_per_mwh_elec = extract_data(self.extra_costs_per_mwh_elec, data)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        ehk = fx.linear_converters.Power2Heat(
            label=self.name,
            eta=self.eta_thermal,
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.bus_elec],
                effects_per_flow_hour={
                    effects['costs']: (extract_data('Strom', time_series_data) + self.extra_costs_per_mwh_elec)
                },
            ),
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
        )
        self.insert_size(
            ehk.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(ehk, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, ehk.Q_th, ehk.eta, effects['costs'])
        return ehk


class Rueckkuehler(ThermalInvestElement):
    specific_electricity_demand: Union[int, float, str] = Field(alias='Strombedarf', default=0)
    extra_costs_per_mwh_elec: Union[int, float, str] = Field(alias='Stromkosten Zusatz [€/MWh]', default=0)

    bus_elec: str = Field(alias='Strombus', default='StromBezug')

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.specific_electricity_demand = extract_data(self.specific_electricity_demand, data)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        cool = fx.linear_converters.CoolingTower(
            label=self.name,
            specific_electricity_demand=self.specific_electricity_demand,
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.bus_elec],
                effects_per_flow_hour={
                    effects['costs']: extract_data('Strom', time_series_data) + self.extra_costs_per_mwh_elec
                },
            ),
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
        )
        self.insert_size(
            cool.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(cool, years_of_model)
        if cool.specific_electricity_demand != 0:
            self.insert_grid_fee(
                self.grid_fee_per_year, cool.Q_th, 1 / cool.specific_electricity_demand, effects['costs']
            )
        return cool


class AbwaermeWaermepumpe(Waermepumpe):
    heat_source_costs: Union[int, float, str] = Field(alias='Abwärmekosten', default=0)
    bus_waste_heat: str = Field(alias='Abwärmebus', default='Abwärme')

    def _insert_data(self, data: pd.DataFrame):
        self.heat_source_costs = extract_data(self.heat_source_costs, data)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        heat_pump = fx.linear_converters.HeatPumpWithSource(
            label=self.name,
            COP=self._get_cop(time_series_data),
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.bus_elec],
                effects_per_flow_hour=self._electricity_effects_per_flow_hour(effects, time_series_data, years_of_model),
            ),
            Q_ab=fx.Flow(
                label='Qab',
                bus=busses[self.bus_waste_heat],
                effects_per_flow_hour={effects['costs']: self.heat_source_costs},
            ),
        )
        self.insert_size(
            heat_pump.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(heat_pump, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, heat_pump.Q_th, heat_pump.COP, effects['costs'])
        return heat_pump


class Geothermie(Waermepumpe):
    amount_of_pump_electricity: Union[int, float, str] = Field(alias='Anteil Pumpstrom pro MW_geo')
    bus_waste_heat: str = Field(alias='Abwärmebus', default='Abwärme')

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.amount_of_pump_electricity = extract_data(self.amount_of_pump_electricity, data)

    def _get_cop(self, time_series_data: pd.DataFrame) -> Union[float, np.ndarray]:
        if self.cop:
            return extract_data(self.cop, time_series_data)
        else:
            cop_wo_pump = self.calculate_cop(
                source_temperature=extract_data(self.source_temperature, time_series_data),
                target_temperature=extract_data(self.sink_temperature, time_series_data),
                eta=0.5,
            )

            return cop_wo_pump / (1 + self.amount_of_pump_electricity)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        heat_pump = fx.linear_converters.HeatPumpWithSource(
            label=self.name,
            COP=self._get_cop(time_series_data),
            Q_th=fx.Flow(
                label='Qth',
                bus=busses[self.bus_heat],
                relative_maximum=self.relative_maximum,
                relative_minimum=self.relative_minimum,
                effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
            ),
            P_el=fx.Flow(
                label='Pel',
                bus=busses[self.bus_elec],
                effects_per_flow_hour=self._electricity_effects_per_flow_hour(effects, time_series_data, years_of_model),
            ),
            Q_ab=fx.Flow(label='Qab', bus=busses[self.bus_waste_heat]),
        )
        self.insert_size(
            heat_pump.Q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(heat_pump, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, heat_pump.Q_th, heat_pump.COP, effects['costs'])
        return heat_pump


class Abwaerme(ThermalInvestElement):
    waste_heat_costs: Union[int, float, str] = Field(alias='Abwärmekosten')
    bus_waste_heat: str = Field(alias='Abwärmebus', default='Abwärme')

    def _insert_data(self, data: pd.DataFrame):
        super()._insert_data(data)
        self.waste_heat_costs = extract_data(self.waste_heat_costs, data)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        q_th = fx.Flow(
            label='Qth',
            bus=busses[self.bus_heat],
            relative_minimum=self.relative_minimum,
            relative_maximum=self.relative_maximum,
            effects_per_flow_hour=self.thermal_effects_per_flow_hour(effects),
        )

        q_abw = fx.Flow(
            label='Qabw',
            bus=busses[self.bus_waste_heat],
            effects_per_flow_hour={effects['costs']: self.waste_heat_costs},
        )

        comp = fx.LinearConverter(
            label=self.name, inputs=[q_abw], outputs=[q_th], conversion_factors=[{q_abw: 1, q_th: 1}]
        )

        self.insert_size(
            q_th,
            self.thermal_power,
            effects,
            years_of_model,
        )
        self.restrict_availlability(comp, years_of_model)
        # No Grid Connection!
        return comp


class KWKekt(InvestElement):
    fuel_power: Union[int, float] = Field(alias='Brennstoff Leistung')
    electrical_power: Tuple[float, float] = Field(alias='Elektrische Leistung (Stützpunkte)')
    thermal_power: Tuple[float, float] = Field(alias='Thermische Leistung (Stützpunkte)')
    fuel_type: str = Field(alias='Brennstoff')
    fuel_costs: Union[float, str] = Field(alias='Brennstoffkosten [€/MWh_hu]', default=0)
    can_be_off: bool = Field(alias='Ausschaltbar', default=True)

    bus_elec: str = Field(alias='Strombus', default='StromEinspeisung')
    bus_heat: str = Field(alias='Wärmebus', default='Fernwärme')

    relative_maximum: Union[int, float, str] = Field(alias='Relative Brennstoff Leistungsobergrenze', default=1)
    relative_minimum: Union[int, float, str] = Field(alias='Relative Brennstoff Leistungsuntergrenze', default=0)
    green_heat_factor: Union[int, float, str] = Field(alias='Grüne Wärme', default=0)

    def _insert_data(self, data: pd.DataFrame):
        self.fuel_costs = extract_data(self.fuel_costs, data)
        self.relative_maximum = extract_data(self.relative_maximum, data)
        self.relative_minimum = extract_data(self.relative_minimum, data)
        self.green_heat_factor = extract_data(self.green_heat_factor, data)

    def _convert_to_flixopt(
        self,
        flow_system: fx.FlowSystem,
        busses: Dict[str, fx.Bus],
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
    ):
        effects = flow_system.effect_collection.effects

        flow_heat = fx.Flow('Qth', busses[self.bus_heat],
                            effects_per_flow_hour={effects['Gruene_Waerme']: self.green_heat_factor})
        flow_fuel = fx.Flow('Qfu', busses[self.fuel_type], effects_per_flow_hour={effects['costs']: self.fuel_costs},
                            relative_minimum=self.relative_maximum,
                            relative_maximum=self.relative_maximum)
        flow_el = fx.Flow(
            'Pel',
            busses[self.bus_elec],
            effects_per_flow_hour={effects['costs']: -1 * extract_data('Strom', time_series_data)},
        )

        if self.can_be_off:
            segmented_conversion_factors = {
                flow_fuel: [(0, 1e-5), (self.fuel_power, self.fuel_power)],
                flow_el: [(0, 1e-5), self.electrical_power],
                flow_heat: [(0, 1e-5), self.thermal_power],
            }
        else:
            segmented_conversion_factors = {
                flow_fuel: [(self.fuel_power, self.fuel_power)],
                flow_el: [self.electrical_power],
                flow_heat: [self.thermal_power],
            }

        comp = fx.LinearConverter(
            label=self.name,
            inputs=[flow_fuel],
            outputs=[flow_heat, flow_el],
            segmented_conversion_factors=segmented_conversion_factors,
        )
        self.insert_size(flow_fuel, self.fuel_power, effects, years_of_model)
        self.restrict_availlability(comp, years_of_model)
        return comp

    @field_validator('electrical_power', mode='before')
    @classmethod
    def validate_electrical_power(cls, value):
        start_end = numbers_from_str(value)
        if len(start_end) != 2:
            raise ValueError('The electrical power points must be exactly 2 numbers (start and end).')
        else:
            return start_end

    @field_validator('thermal_power', mode='before')
    @classmethod
    def validate_thermal_power(cls, value):
        return numbers_from_str(value)

    @model_validator(mode='after')
    def validate_power_points(self):
        if len(self.electrical_power) != len(self.thermal_power):
            raise ValueError(
                f'The number of electrical power points ({len(self.electrical_power)}) must match the '
                f'number of thermal power points ({len(self.thermal_power)}).'
            )

        for epp, tpp in zip(self.electrical_power, self.thermal_power, strict=False):
            if epp / self.fuel_power > 1:
                raise ValueError(f'The electric efficiency of {self.props["Name"]} exceeds 100%.')
            if tpp / self.fuel_power > 1:
                raise ValueError(f'The thermal efficiency of {self.props["Name"]} exceeds 100%.')
            if (epp + tpp) / self.fuel_power > 1:
                raise ValueError(f'The total efficiency of {self.props["Name"]} exceeds 100%.')


class ElementFactory:
    def __init__(
        self,
        flow_system: fx.FlowSystem,
        time_series_data: pd.DataFrame,
        co2_factors: Dict[str, float],
        years_of_model: List[int],
        busses: Dict[str, fx.Bus],
    ):
        self.time_series_data = time_series_data
        self.co2_factors = co2_factors
        self.years_of_model = years_of_model
        self.flow_system = flow_system
        self.busses = busses

        self.created_comps: List[Element] = []

    def create_energy_object(self, obj_type: str, properties: Dict) -> None:
        obj_class = self.get_class_by_type(obj_type)
        if obj_class:
            energy_obj: Element = obj_class(**properties)
            self.created_comps.append(energy_obj)
            energy_obj.add_to_flow_system(
                flow_system=self.flow_system,
                busses=self.busses,
                time_series_data=self.time_series_data,
                co2_factors=self.co2_factors,
                years_of_model=self.years_of_model,
            )
            logger.info(f'Created {obj_type} "{energy_obj.name}"')
        else:
            raise ValueError(f'Unknown energy object type: {obj_type}')

    def get_class_by_type(self, obj_type):
        # Map obj_type to the appropriate class
        class_map = {
            'Waermepumpe': Waermepumpe,
            'KWK': KWK,
            'Kessel': Kessel,
            'Speicher': Speicher,
            'LinearTransformer_1_1': LinearTransformer,
            'Sink': Sink,
            'Source': Source,
            'AbwaermeWP': AbwaermeWaermepumpe,
            'Geothermie': Geothermie,
            'KWKekt': KWKekt,
            'EHK': EHK,
            'AbwaermeHT': Abwaerme,
            'Rueckkuehler': Rueckkuehler,
            # More mappings as needed
        }
        return class_map.get(obj_type)

    def print_comps(self):
        rep = ''
        for comp in sorted(self.created_comps, key=lambda comp: comp.name):
            rep += f'{comp}\n'
        return rep


def extract_data(value: Union[str, Any], data: pd.DataFrame) -> Union[np.ndarray, Any]:
    """
    Extracts data from a DataFrame based on the provided value. If the value is a string, it is assumed to be a column name
    and the corresponding data is returned. If the value is not a string, it is assumed to be the actual data and is simply
    returned.
    """

    if isinstance(value, str):
        if value not in data.columns:
            raise KeyError(
                f"Column '{value}' not found in the time series data provided. "
                f'Only the following columns where found: {list(data.columns)}'
            )
        return data[value].to_numpy()
    else:
        return value


def insert_effects(dictionary: Dict[Union[fx.Effect, str], Any], effects: Dict[str, fx.Effect]) -> None:
    if dictionary is None or dictionary == 0 or dictionary == {}:
        return None
    for effect_name in dictionary:
        if effect_name in effects.keys():
            dictionary[effects[effect_name]] = dictionary.pop(effect_name)
        elif not isinstance(effect_name, fx.Effect):
            raise KeyError(f"Key '{effect_name}' is not found in effects Collection.")


# Limit availability of Elements
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


def restrict_availlability(
    component: flixOpt.elements.Component, exists: Union[int, float, np.ndarray]
) -> flixOpt.elements.Component:
    for flow in component.inputs + component.outputs:
        flow.relative_maximum = flow.relative_maximum * exists
        flow.relative_minimum = flow.relative_minimum * exists
    if isinstance(component, fx.Storage):
        storage_exists_exists = exists if isinstance(exists, (int, float)) else np.append(exists, exists[-1])
        component.relative_maximum_charge_state = component.relative_maximum_charge_state * storage_exists_exists
        component.relative_minimum_charge_state = component.relative_minimum_charge_state * storage_exists_exists
    return component


def numbers_from_str(input_string: str, delimiter: str = '-', check_ascending: bool = False) -> Tuple[float, ...]:
    """
    Extract numbers from a delimited string and return them as a tuple of floats.

    Parameters
    ----------
    input_string : str
        The input string to extract numbers from.
    delimiter : str, optional
        The delimiter to use for splitting the input string. Default is '-'.
    check_ascending : bool, optional
        Whether to check if the numbers are in ascending order. Default is False.

    Returns:
    -------
    Tuple[float, ...]
        A tuple of floats extracted from the input string.

    Raises:
    ------
    ValueError
        If the input string contains invalid or non-numeric values.
    """
    input_string = input_string.replace(',', '.')  # Replace commas with dots (for robust decimal support)

    parts = [
        part.strip() for part in input_string.split(delimiter)
    ]  # Split the string by the delimiter and strip whitespace

    # Convert to floats and validate
    try:
        numbers = tuple(float(part) for part in parts if part)
    except ValueError as e:
        raise ValueError(f"Invalid input: '{input_string}'. All parts must be numeric.") from e

    if not numbers:  # Check for empty results
        raise ValueError(f"Invalid input: '{input_string}'. No valid numbers found.")
    if check_ascending and not all(x <= y for x, y in zip(numbers, numbers[1:], strict=False)):
        raise ValueError(
            f"Invalid input: '{input_string}'. Numbers must be in ascending order. "
            f'Found a violation in the sequence: {numbers}.'
        )
    return numbers


def add_effect_per_flow_hour(
    flow: fx.Flow, effect: fx.Effect, standard_effect: fx.Effect, factor: Union[float, np.ndarray]
):
    if isinstance(flow.effects_per_flow_hour, dict):
        flow.effects_per_flow_hour.update({effect: factor})
    elif flow.effects_per_flow_hour is None:
        flow.effects_per_flow_hour = {effect: factor}
    else:
        flow.effects_per_flow_hour = {effect: factor, standard_effect: flow.effects_per_flow_hour}


# validation functions


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
                        f'which totals to {sum(flow.meta_data["fixed_effects"][effect.label])}'
                    )

            specific_effects = flow.size.specific_effects or {}  # Making sure its a dict
            for effect, value in specific_effects.items():
                if effect.label not in flow.meta_data['specific_costs']:
                    logger.warning(f'Effect {effect.label} not found in specific_costs meta_data. {value=}')
                elif abs(abs(sum(flow.meta_data['specific_costs'][effect.label])) - abs(value)) >= 1e-5:
                    logger.critical(
                        f'The meta_data for {flow.label_full=} is not correct for the investment effects per period.'
                        f'The total {effect.label=} passed to the InvestParameters is {value}. '
                        f'The meta_data is {flow.meta_data["specific_costs"][effect.label]}, '
                        f'which totals to {sum(flow.meta_data["specific_costs"][effect.label])}'
                    )


def validate_invest_range(
        value: Union[int, float, str],
        label: str
) -> Union[int, float, Tuple[Union[int, float], Union[int, float]]]:
    """
    This function was written to validate the investment range of a component.
    It checks if the value is a number or a string in the format 'X-Y' and if it is positive.
    If the value is a string, it is split into two parts and validated as a range.
    If the value is a number, it is validated as positive.
    """
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError(f"'{label}' must be positive.")
        return value
    elif isinstance(value, str):
        parts = value.split('-')
        if len(parts) != 2:
            raise ValueError("Invalid range format. Expected 'X-Y'.")
        try:
            start, end = float(parts[0]), float(parts[1])
        except ValueError as e:
            raise ValueError("Invalid range format. Expected 'X-Y'.") from e
        if start >= end:
            raise ValueError('Range start must be less than range end.')
        if start < 0:
            raise ValueError('Range start must be positive.')
        return start, end
    else:
        raise ValueError(f"'{label}' must be a number or a string in the format 'X-Y'.")
