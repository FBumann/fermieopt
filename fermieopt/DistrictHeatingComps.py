from typing import Optional, Union, Any, Dict, Literal, List, Tuple
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from rich import print

import flixOpt as fx
import flixOpt.elements

logger = logging.getLogger('flixOpt')


class Element(BaseModel,
              populate_by_name = True,  # Enables using both field names and aliases
              extra = 'forbid'):  # Forbids unexpected keys in input data
    name: str = Field(alias="Name")
    group: Optional[str] = Field(alias='Gruppe', default=None)

    def add_to_flow_system(self,
                           flow_system: fx.FlowSystem,
                           effects: Dict[str, fx.Effect],
                           busses: Dict[str, fx.Bus],
                           time_series_data: pd.DataFrame,
                           co2_factors: Dict[str, float] = None,
                           years_of_model: List[int] = None):
        flow_system.add_elements(
            self._convert_to_flixopt(flow_system, effects, busses, time_series_data, co2_factors, years_of_model)
        )

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        raise NotImplementedError

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        raise NotImplementedError


class InvestElement(Element):
    start_year: Optional[int] = Field(alias='Startjahr', default=None, ge=1800)
    lifetime: Optional[int] = Field(alias='Lebensdauer', default=None, ge=1)
    optional: bool = Field(alias='Optional', default=False)
    invest_costs_fixed: Union[int, float] = Field(alias='Investkosten (fix) [€]', default=0)
    invest_costs_specific: Union[int, float] = Field(alias='Investkosten (spezifisch) [€/MW]', default=0)
    annual_costs_fixed: Union[int, float] = Field(alias='Sonstige Fixkosten (fix) [€/a]', default=0)
    annual_costs_specific: Union[int, float] = Field(alias='Sonstige Fixkosten (spezifisch) [€/(MW*a)]', default=0)
    interest_rate: Union[int, float] = Field(alias='Zinssatz', default=0)
    funding_rate: Union[int, float] = Field(alias='Fördersatz', default=0)

    @property
    def needs_investment(self) -> bool:
        return (self.invest_costs_fixed is not None
                or self.invest_costs_specific is not None
                or self.optional)

    @model_validator(mode='after')
    def validate_years(self):
        """Validates the start and lifetime of the element"""
        if not (self.start_year is None) == (self.lifetime is None):
            raise ValueError(f"Either set BOTH or NONE of 'Startjahr' and 'Lebensdauer'!")
        return self

    @staticmethod
    def annuity_factor(interest_rate: float, lifetime: int) -> float:
        """ Get the annuity factor for a given interest rate and lifetime """
        if interest_rate == 0:  # Preventing ZeroDivision
            annuity_factor = 1 / lifetime
        else:
            annuity_factor = (((1 + interest_rate) ** lifetime * interest_rate) /
                              ((1 + interest_rate) ** lifetime - 1))
        return annuity_factor

    @staticmethod
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
        """
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
        """
        annuity_factor = InvestElement.annuity_factor(interest_rate=interest_rate, lifetime=lifetime)
        accounting_years = np.array(
            [1 if starting_year <= year < (starting_year + lifetime) else 0 for year in years_of_model])

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


class PowerInvestElement(InvestElement):
    power: Union[int, float] = Field(alias='Nennleistung [MW]')
    flow_label: str = Field(alias="Flowname")
    fixed_profile: Optional[str] = Field(alias="Festes Profil", default=None)
    bus: str = Field(alias="Bus")

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        self.fixed_profile = extract_data(self.fixed_profile, data)

    def insert_size(self, 
                                  flow: fx.Flow, 
                                  effects: Dict[str, fx.Effect], 
                                  years_of_model: List[int]) -> None:
        if not self.needs_investment:
            flow.size = self.power
        else:
            fixed_effects_per_period, specific_effects_per_period = self.costs_and_funding(
                interest_rate=self.interest_rate,
                starting_year=self.start_year,
                lifetime=self.lifetime,
                years_of_model=years_of_model,
                invest_costs=self.invest_costs_fixed,
                specific_invest_costs=self.invest_costs_specific,
                annual_costs=self.annual_costs_fixed,
                specific_annual_costs=self.annual_costs_specific,
                funding_rate=self.funding_rate)

            fixed_effects_total = {effect: np.sum(values) for effect, values in fixed_effects_per_period.items()}
            specific_effects_total = {effect: np.sum(values) for effect, values in specific_effects_per_period.items()}

            flow.size = fx.InvestParameters(
                optional=self.optional,
                fixed_size=self.power if isinstance(self.power, (int, float)) else None,
                minimum_size=self.minimum_power,
                maximum_size=self.maximum_power,
                fix_effects=insert_effects(fixed_effects_total, effects),
                specific_effects=insert_effects(specific_effects_total, effects),
            )
            update_meta_data(flow,
                             {'invest_effects': fixed_effects_per_period,
                              'specific_effects': specific_effects_per_period
                              })

    @property
    def minimum_power(self) -> Optional[float]:
        return float(self.power.split("-")[0]) if isinstance(self.power, str) else None

    @property
    def maximum_power(self):
        return float(self.power.split("-")[1]) if isinstance(self.power, str) else None

    @field_validator("power", mode="after")
    def validate_power(cls, value):
        return validate_invest_range(value, label="Nennleistung [MW]")


class ThermalInvestElement(InvestElement):
    thermal_power: Union[int, float, str] = Field(alias='Thermische Leistung [MW]')
    grid_fee_per_year: Union[float, str] = Field(alias='Netzentgelt [€/(MW*a)]', default=0)

    def insert_size(self,
                    flow: fx.Flow,
                    effects: Dict[str, fx.Effect],
                    years_of_model: List[int]):
        if not self.needs_investment:
            flow.size = self.thermal_power
        else:
            fixed_effects_per_period, specific_effects_per_period = self.costs_and_funding(
                interest_rate=self.interest_rate,
                starting_year=self.start_year,
                lifetime=self.lifetime,
                years_of_model=years_of_model,
                invest_costs=self.invest_costs_fixed,
                specific_invest_costs=self.invest_costs_specific,
                annual_costs=self.annual_costs_fixed,
                specific_annual_costs=self.annual_costs_specific,
                funding_rate=self.funding_rate)

            fixed_effects_total = {effect: np.sum(values) for effect, values in fixed_effects_per_period.items()}
            specific_effects_total = {effect: np.sum(values) for effect, values in specific_effects_per_period.items()}

            flow.size = fx.InvestParameters(
                optional=self.optional,
                fixed_size=self.power if isinstance(self.power, (int, float)) else None,
                minimum_size=self.minimum_power,
                maximum_size=self.maximum_power,
                fix_effects=insert_effects(fixed_effects_total, effects),
                specific_effects=insert_effects(specific_effects_total, effects),
            )

            update_meta_data(flow, {
                'invest_effects': fixed_effects_per_period,
                'specific_effects': specific_effects_per_period
            })

    @staticmethod
    def insert_grid_fee(grid_fee: Union[int, float],
                        grid_flow: fx.Flow,
                        invest_flow: fx.Flow,
                        efficiency: Union[int, float, np.ndarray],
                        effect: fx.Effect) -> None:
        """ Adds the grid fee to the investment parameters of the invest_flow and it's meta_data. """
        if not isinstance(invest_flow.size, fx.InvestParameters) and not grid_fee == 0:
            raise Exception("There are no InvestParameters to add the grid_fee to")
        else:
            highest_possible_grid_draw = np.max(grid_flow.relative_maximum / efficiency)
            yearly_grid_fee = grid_fee * highest_possible_grid_draw
            invest_flow.size.specific_effects[effect] = (yearly_grid_fee
                                                         + invest_flow.size.specific_effects.get(effect, 0))
            update_meta_data(invest_flow, {'yearly_grid_fee_per_thermal_power': yearly_grid_fee,
                                           'highest_possible_grid_draw': highest_possible_grid_draw})
            update_meta_data(invest_flow, {'specific_effects': {effect: yearly_grid_fee}}, mode='add')

    @property
    def minimum_thermal_power(self) -> Optional[float]:
        return float(self.thermal_power.split("-")[0]) if isinstance(self.thermal_power, str) else None

    @property
    def maximum_thermal_power(self):
        return float(self.thermal_power.split("-")[1]) if isinstance(self.thermal_power, str) else None

    @field_validator("thermal_power", mode="after")
    def validate_thermal_power(cls, value):
        return validate_invest_range(value, label="Thermische Leistung [MW]")


class Sink(PowerInvestElement):
    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        comp = fx.Sink(
            label=self.name,
            sink=fx.Flow(label=self.flow_label,
                         bus=busses[self.bus],
                         fixed_relative_profile=self.fixed_profile))
        self.insert_size(comp.sink, effects, years_of_model)
        return comp


class Source(PowerInvestElement):
    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        comp = fx.Source(
            label=self.name,
            source=fx.Flow(label=self.flow_label,
                         bus=busses[self.bus],
                         fixed_relative_profile=self.fixed_profile)
        )

        self.insert_size(comp.source, effects, years_of_model)
        return comp


class LinearTransformer(PowerInvestElement):
    efficiency: Union[int, float, str] = Field(alias="Wirkungsgrad")
    bus_in: str = Field(alias="Von Bus")
    bus_out: str = Field(alias="Zu Bus")
    flow_label_in: str = Field(alias="Flowname in")
    cost_per_mwh_in: Union[int, float, str] = Field(alias="Kosten pro MWh von Bus", default=0)

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        super()._insert_data(data)
        self.efficiency = extract_data(self.efficiency, data)
        self.cost_per_mwh_in = extract_data(self.cost_per_mwh_in, data)

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):

        flow_out = fx.Flow(label=self.flow_label,
                           bus=busses[self.bus_out],
                           fixed_relative_profile=self.fixed_profile
                           )

        flow_in = fx.Flow(label=self.flow_label_in,
                           bus=busses[self.bus_in],
                           effects_per_flow_hour={effects['costs']: self.cost_per_mwh_in}
                           )

        comp = fx.LinearConverter(label=self.name,
                                  inputs=[flow_in],
                                  outputs=[flow_out],
                                  conversion_factors=[{flow_in: self.efficiency, flow_out: 1}])
        self.insert_size(flow_out, effects, years_of_model)
        return comp


class Kessel(ThermalInvestElement):
    eta_thermal: Union[float, str] = Field(alias="Thermischer Wirkungsgrad")
    fuel_type: str = Field(alias='Brennstoff')

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        self.eta_thermal = extract_data(self.eta_thermal, data)

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        boiler = fx.linear_converters.Boiler(
            label=self.name,
            eta=self.eta_thermal,
            Q_fu=fx.Flow(label="Q_fu", bus=busses["bus_fuel"],
                         effects_per_flow_hour={effects['costs']: extract_data(self.fuel_type, time_series_data)}),
            Q_th=fx.Flow(label="Q_th", bus=busses["bus_heat"])
        )
        self.insert_size(boiler.Q_th, effects, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, boiler.Q_fu, boiler.Q_th, boiler.eta, effects['costs'])
        return boiler


class KWK(ThermalInvestElement):
    eta_th: Union[int, float, str] = Field(alias='eta_th')
    eta_el: Union[int, float, str] = Field(alias='eta_el')
    fuel_type: str = Field(alias='Brennstoff')
    extra_costs_per_mwh_fuel: Union[int, float, str] = Field(alias='Zusatzkosten pro MWh Brennstoff', default=0)
    forward_flow_temperature: Union[int, float, str] = Field(alias='Vorlauftemperatur')
    reverse_flow_temperature: Union[int, float, str] = Field(alias='Rücklauftemperatur')
    ambient_temperature: Union[int, float, str] = Field(alias='Umgebungstemperatur')

    bus_heat: str = Field(alias="Wärmebus", default='Fernwärme')
    bus_elec: str = Field(alias="Strombus", default='StromBezug')

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        self._insert_data(time_series_data)

        chp = fx.linear_converters.CHP(
            label=self.name,
            eta_th=self.eta_th,
            eta_el=self.eta_el,
            Q_th=fx.Flow(label='Qth', bus=busses[self.bus_heat]),
            P_el=fx.Flow(label='Pel', bus=busses[self.bus_elec],
                         effects_per_flow_hour={
                             effects['costs']: -1 * extract_data('Strom', time_series_data),
                             effects['CO2FW']: -1 * self.co2_emissions_electricity(time_series_data, co2_factors)
                         }),
            Q_fu=fx.Flow(label='Qfu', bus=busses[self.bus_fuel],
                         effects_per_flow_hour={
                             effects['costs']: (extract_data(self.fuel_type, time_series_data) +
                                                self.extra_costs_per_mwh_fuel +
                                                extract_data('CO2', time_series_data)),
                             effects['CO2']: (self.co2_factor(time_series_data, co2_factors) *
                                              extract_data('CO2', time_series_data))
                         }),
        )
        self.insert_size(chp.Q_th, effects, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, chp.P_el, chp.Q_th, chp.eta_el, effects['costs'])
        return chp

    def _insert_data(self, time_series_data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        super()._insert_data(time_series_data)
        self.eta_th = extract_data(self.eta_th, time_series_data)
        self.eta_el = extract_data(self.eta_el, time_series_data)
        self.extra_costs_per_mwh_fuel = extract_data(self.extra_costs_per_mwh_fuel, time_series_data)
        self.forward_flow_temperature = extract_data(self.forward_flow_temperature, time_series_data)
        self.reverse_flow_temperature = extract_data(self.reverse_flow_temperature, time_series_data)
        self.ambient_temperature = extract_data(self.ambient_temperature, time_series_data)

    def co2_factor(self, time_series_data: pd.DataFrame, co2_factors: Dict[str, float]) -> float:
        return extract_data(co2_factors.get(self.fuel_type, 0), time_series_data)

    def co2_emissions_electricity(self, time_series_data: pd.DataFrame, co2_factors: Dict[str, float]) -> np.ndarray:
        try:
            fuel_factor_electricity = self.fuel_factor_for_electrical_energy(
                electrical_efficiency=self.eta_el,
                thermal_efficiency=self.eta_th,
                inferior_temperature=self.ambient_temperature,
                forward_flow_temperature=self.forward_flow_temperature,
                reverse_flow_temperature=self.reverse_flow_temperature
            )
        except KeyError:
            logger.warning(
                f"Computation of CO2 Reward did not work properly. Using default values instead. "
                f"Optimization itself is not affected. Only take care interpreting CO2 Emissions")
            fuel_factor_electricity = self.fuel_factor_for_electrical_energy(
                electrical_efficiency=self.eta_el,
                thermal_efficiency=self.eta_th,
            )
        return fuel_factor_electricity * self.co2_factor(time_series_data, co2_factors)

    @staticmethod
    def fuel_factor_for_electrical_energy(electrical_efficiency: Union[int, float, np.ndarray],
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
        superior_temperature = ((forward_flow_temperature - reverse_flow_temperature) /
                                np.log((forward_flow_temperature / reverse_flow_temperature)))
        n_carnot = 1 - (inferior_temperature / superior_temperature)

        a_el = (1 * electrical_efficiency) / (electrical_efficiency + n_carnot * thermal_efficiency)
        return a_el / electrical_efficiency


class Waermepumpe(ThermalInvestElement):
    cop: Optional[Union[int, float, str]] = Field(alias="COP")
    source_temperature: Union[int, float, str] = Field(alias="Quelltemperatur")
    sink_temperature: Union[int, float, str] = Field(alias="Zieltemperatur")

    extra_costs_per_mwh_elec: Union[int, float, str] = Field(alias="Zusatzkosten pro MWh Strom", default=0)
    minimum_source_temperature: Optional[Union[int, float]] = Field(alias="Untergrenze für Einsatz", default=None)
    operation_restriction: Optional[str] = Field(alias="Zeitreihe für Einsatzbeschränkung", default=None)

    scop_bew: Optional[Union[int, float]] = Field(alias="SCOP für BEW", default=None)
    max_bew_elec_funding: Optional[Union[int, float]] = Field(alias="Maximale Stromkostenförderung BEW", default=None)

    bus_heat: str = Field(alias="Wärmebus", default='Fernwärme')
    bus_elec: str = Field(alias="Strombus", default='StromBezug')

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        self._insert_data(time_series_data)

        heat_pump = fx.linear_converters.HeatPump(
            label=self.name,
            COP = self._get_cop(time_series_data),
            Q_th=fx.Flow(label='Qth', bus=busses[self.bus_heat], relative_maximum=self.relative_maximum),
            P_el=fx.Flow(label='Pel', bus=busses[self.bus_elec],
                         effects_per_flow_hour={
                             effects['costs']: self._get_electricity_costs_per_mwh(time_series_data),
                             effects['funding']: self._get_operation_funding_bew(time_series_data, years_of_model)
                         })
        )
        self.insert_size(heat_pump.Q_th, effects, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, heat_pump.P_el, heat_pump.Q_th, heat_pump.COP, effects['costs'])
        return heat_pump

    def _get_cop(self, time_series_data: pd.DataFrame) -> Union[float, np.ndarray]:
        if self.cop:
            return extract_data(self.cop, time_series_data)
        else:
            return self.calculate_cop(
                source_temperature=extract_data(self.source_temperature, time_series_data),
                target_temperature=extract_data(self.sink_temperature, time_series_data),
                eta=0.5)

    def _get_electricity_costs_per_mwh(self, time_series_data: pd.DataFrame) -> Union[float, np.ndarray]:
        return extract_data('Strom', time_series_data) + extract_data(self.extra_costs_per_mwh_elec, time_series_data)

    def _get_operation_funding_bew(self, time_series_data: pd.DataFrame, years_of_model: List[int]) -> Union[float, np.ndarray]:
        fund_per_mw_el = self.bew_operation_funding_from_scop(self.scop_bew, "MWh_el")

        electricity_costs_per_flow_hour = self._get_electricity_costs_per_mwh(time_series_data)
        # Begrenzung der Förderung auf x% der Stromkosten
        fund_per_mw_el = np.where(
            fund_per_mw_el < electricity_costs_per_flow_hour * self.max_bew_elec_funding,
            fund_per_mw_el, electricity_costs_per_flow_hour * self.max_bew_elec_funding)
        # Begrenzung auf 10 Jahre
        return fund_per_mw_el * exists(self.start_year, 10, years_of_model)

    @staticmethod
    def bew_operation_funding_from_scop(scop: Union[int, float],
                                        unit: Literal["MWh_amb", "MWh_th", "MWh_el"] = "MWh_amb"
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
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        super()._insert_data(data)
        self.cop = extract_data(self.cop, data)
        self.source_temperature = extract_data(self.source_temperature, data)
        self.sink_temperature = extract_data(self.sink_temperature, data)

        self.extra_costs_per_mwh_elec = extract_data(self.extra_costs_per_mwh_elec, data)

    @model_validator(mode='after')
    def validate_cop(self):
        if not self.cop:
            if not self.source_temperature or not self.sink_temperature:
                raise Exception(f"Need to specify a 'COP' for {self.name} or "
                                f"use 'Quelltemperatur' and 'Zieltemperatur' to calculate the COP internally.")
        if self.cop:
            if self.source_temperature or self.sink_temperature:
                raise Exception(f"Either specify a 'COP' for {self.name} "
                                f"OR use 'Quelltemperatur' and 'Zieltemperatur' to calculate the COP internally.")
        return self

    @model_validator(mode='after')
    def validate_bew(self):
        if self.scop_bew:
            if not self.start_year:
                raise Exception(f"Need to specify a Year of Operation start for {self.name} to use HeatPump "
                                f"operation funding, because its limited to 10 years.")
            if not self.max_bew_elec_funding:
                raise Exception(f"Need to specify 'Maximale Stromkostenförderung BEW' for {self.name} to use HeatPump "
                                f"operation funding.")
        return self

    @model_validator(mode='after')
    def validate_operation_restriction(self):
        if not (self.time_series_for_operation_restriction is None) == (self.minimum_source_temperature is None):
            raise Exception(f"Need to specify either both or none of 'Zeitreihe für Einsatzbeschränkung' and "
                            f"'ntergrenze für Einsatz' for {self.name}.")
        return self


class Speicher(ThermalInvestElement):
    capacity: Union[int, float, str] = Field(alias='Kapazität [MWh]')
    invest_costs_capacity_specific: Union[int, float] = Field(alias='Investkosten [€/MWh]', default=0)
    annual_costs_capacity_specific: Union[int, float] = Field(alias='Sonstige Fixkosten [€/(MWh*a)]', default=0)

    eta_load: Union[int, float, str] = Field(alias='eta_load')
    eta_unload: Union[int, float, str] = Field(alias='eta_unload')
    loss_per_hour: Union[int, float, str] = Field(alias='VerlustProStunde', default=0)

    depends_on_temperature: bool = Field(alias='AbhängigkeitVonDT', default=False)
    temperature_lower: Union[int, float, str] = Field(alias='Untere Temperatur')
    temperature_upper: Union[int, float, str] = Field(alias='Obere Temperatur')

    default_temperature_spread: Union[int, float] = Field(alias='TemperaturSpread', default=65)

    def _insert_data(self, time_series_data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        super()._insert_data(time_series_data)
        self.eta_load = extract_data(self.eta_load, time_series_data)
        self.eta_unload = extract_data(self.eta_unload, time_series_data)
        self.loss_per_hour = extract_data(self.loss_per_hour, time_series_data)
        self.temperature_lower = extract_data(self.temperature_lower, time_series_data)
        self.temperature_upper = extract_data(self.temperature_upper, time_series_data)

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        self._insert_data(time_series_data)

        storage = fx.Storage(
            label=self.name,
            capacity_in_flow_hours=self._get_capacity(effects),
            eta_charge=self.eta_load,
            eta_discharge=self.eta_unload,
            relative_loss_per_hour=self.loss_per_hour,
            relative_maximum_charge_state=self.normalized_temperature_spread(time_series_data),
            charging=fx.Flow(label='QthLoad',
                             bus=busses["Fernwärme"],
                             relative_maximum=self.normalized_temperature_spread(time_series_data)),
            discharging=fx.Flow(label='QthUnload',
                                bus=busses["Fernwärme"],
                                relative_maximum=self.normalized_temperature_spread(time_series_data)),
            prevent_simultaneous_charge_and_discharge=True
        )
        self.insert_size(storage.charging, effects, years_of_model)
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
                maximum_size=flow_with_size.size.maximum_size)
    
            if flow_with_size.size.fixed_size is None:
                effect = fx.Effect(label=f"{self.name}_link_thermal_power", unit="",
                                 description=f"Links the charge and discharge investment value of storage {self.name}",
                                 minimum_invest=0, maximum_invest=0)
                flow_system.add_effects(effect)
    
                flow_with_size.size.specific_effects[effect] = 1
                flow_to_link.size.specific_effects = {effect: -1}

    def insert_capacity(self, storage: fx.Storage, effects: [str, fx.Effect], years_of_model: List[int]) -> None:
        if not self.needs_investment:
            storage.capacity_in_flow_hours = self.capacity
        else:
            _, specific_effects_per_period = self.costs_and_funding(
                interest_rate=self.interest_rate,
                starting_year=self.start_year,
                lifetime=self.lifetime,
                years_of_model=years_of_model,
                invest_costs=0,
                annual_costs=0,
                specific_invest_costs=self.invest_costs_capacity_specific,
                specific_annual_costs=self.annual_costs_capacity_specific,
                funding_rate=self.funding_rate)

            specific_effects_total = {effect: np.sum(values) for effect, values in specific_effects_per_period.items()}

            storage.size = fx.InvestParameters(
                optional=self.optional,
                fixed_size=self.capacity if isinstance(self.capacity, (int, float)) else None,
                minimum_size=self.minimum_capacity,
                maximum_size=self.maximum_capacity,
                specific_effects=insert_effects(specific_effects_total, effects)
            )

            update_meta_data(storage, {'specific_effects': specific_effects_per_period})

    def normalized_temperature_spread(self, time_series_data: pd.DataFrame) -> np.ndarray:
        return ((extract_data(self.temperature_upper, time_series_data)
                - extract_data(self.temperature_lower, time_series_data)
                 )
                / self.default_temperature_spread)

    @field_validator('grid_fee_per_year')
    def validate_grid_fee(cls, value):
        if value is not None:
            raise ValueError(f"Netzentgelt is not supported for '{cls.__name__}")
        return value


class EHK(ThermalInvestElement):
    eta_thermal: Union[float, str] = Field(alias="Thermischer Wirkungsgrad")

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        self.eta_thermal = extract_data(self.eta_thermal, data)

    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        ehk = fx.linear_converters.Power2Heat(
            label=self.name,
            eta=self.eta_thermal,
            P_el=fx.Flow(label="P_el", bus=busses["bus_elec"],
                         effects_per_flow_hour={effects['costs']: extract_data('Strom', time_series_data)}),
            Q_th=fx.Flow(label="Q_th", bus=busses["bus_heat"])
        )
        self.insert_size(ehk.Q_th, effects, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, ehk.P_el, ehk.Q_th, ehk.eta, effects['costs'])
        return ehk


class Rueckkuehler(ThermalInvestElement):
    specific_electricity_demand: Union[int, float, str] = Field(alias="Strombedarf", default=0)
    extra_costs_per_mwh_elec: Union[int, float, str] = Field(alias="Zusatzkosten pro MWh Strom", default=0)

    def _insert_data(self, data: pd.DataFrame):
        """Inserts data into the model. This method is supposed to be called right after creating an instance."""
        self.specific_electricity_demand = extract_data(self.specific_electricity_demand, data)


    def _convert_to_flixopt(self,
                            flow_system: fx.FlowSystem,
                            effects: Dict[str, fx.Effect],
                            busses: Dict[str, fx.Bus],
                            time_series_data: pd.DataFrame,
                            co2_factors: Dict[str, float],
                            years_of_model: List[int]):
        cool = fx.linear_converters.CoolingTower(
            label=self.name,
            specific_electricity_demand=self.specific_electricity_demand,
            P_el=fx.Flow(label="P_el", bus=busses["bus_elec"],
                         effects_per_flow_hour={effects["costs"]: time_series_data['Strom'] + self.extra_costs_per_mwh_elec}),
            Q_th=fx.Flow(label="Q_th", bus=busses["bus_heat"])
        )
        self.insert_size(cool.Q_th, effects, years_of_model)
        self.insert_grid_fee(self.grid_fee_per_year, cool.P_el, cool.Q_th, 1/cool.specificElectricityDemand, effects['costs'])
        return cool


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


class ElementFactory:
    def __init__(self,
                 flow_system: fx.FlowSystem,
                 time_series_data: pd.DataFrame,
                 co2_factors: Dict[str, float],
                 years_of_model: List[int]):
        self.time_series_data = time_series_data
        self.co2_factors = co2_factors
        self.years_of_model = years_of_model
        self.flow_system = flow_system
        self.effects = {effect.label: effect for effect in flow_system.effect_collection.effects}
        self.busses = {bus.label: bus for bus in flow_system.all_buses}

        self.created_comps: List[Element] = []

    def create_energy_object(self, obj_type: str, **properties):
        obj_class = self.get_class_by_type(obj_type)
        if obj_class:
            energy_obj: Element = obj_class(**properties)
            self.created_comps.append(energy_obj)
            return energy_obj.add_to_flow_system(flow_system=self.flow_system,
                                                 effects=self.effects,
                                                 busses=self.busses,
                                                 time_series_data=self.time_series_data,
                                                 co2_factors=self.co2_factors,
                                                 years_of_model=self.years_of_model)
        else:
            raise ValueError(f"Unknown energy object type: {obj_type}")

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

            #'AbwaermeWP': AbwaermeWaermepumpe,
            #'Geothermie': Geothermie,
            #'KWKekt': KWKekt,

            #'EHK': EHK,

            #'AbwaermeHT': Abwaerme,
            #'Rueckkuehler': Rueckkuehler,


            # More mappings as needed
        }
        return class_map.get(obj_type)

    def print_comps(self):
        rep = ""
        for comp in sorted(self.created_comps, key=lambda comp: comp.props["Name"]):
            rep += f"{comp}\n"
        return rep


def extract_data(value: Union[str, Any], data: pd.DataFrame) -> Union[np.ndarray, Any]:
    """
    Extracts data from a DataFrame based on the provided value. If the value is a string, it is assumed to be a column name
    and the corresponding data is returned. If the value is not a string, it is assumed to be the actual data and is simply
    returned.
    """

    if isinstance(value, str):
        if value not in data.columns:
            raise KeyError(f"Column '{value}' not found in the time series data provided. "
                           f"Only the following columns where found: {list(data.columns)}")
        return data[value].to_numpy()
    else:
        return value


def insert_effects(dictionary: Dict[Union[fx.Effect, str], Any],
                   effects: Dict[str, fx.Effect]) -> None:

    if dictionary is None or dictionary == 0 or dictionary == {}:
        return None
    for effect_name, value in list(dictionary.items()):
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


def restrict_availlability(component: flixOpt.elements.Component, exists: Union[int, float, np.ndarray]) -> flixOpt.elements.Component:
    for flow in component.inputs + component.outputs:
        flow.relative_maximum = flow.relative_maximum * exists
        flow.relative_minimum = flow.relative_minimum * exists
    if isinstance(component, fx.Storage):
        storage_exists_exists = exists if isinstance(exists, (int, float)) else np.append(exists, exists[-1])
        component.relative_maximum_charge_state = component.relative_maximum_charge_state * storage_exists_exists
        component.relative_minimum_charge_state = component.relative_minimum_charge_state * storage_exists_exists
    return component


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


def add_effect_per_flow_hour(flow: fx.Flow, effect: fx.Effect, standard_effect: fx.Effect, factor: Union[float, np.ndarray]):
    if isinstance(flow.effects_per_flow_hour, dict):
        flow.effects_per_flow_hour.update({effect: factor})
    elif flow.effects_per_flow_hour is None:
        flow.effects_per_flow_hour = {effect: factor}
    else:
        flow.effects_per_flow_hour = {effect: factor, standard_effect: flow.effects_per_flow_hour}


def update_meta_data(element: flixOpt.elements.Element,
                     meta_data: Dict[str, Any],
                     mode: Literal['replace', 'add'] = 'replace'
                     ):
    if element.meta_data is None:
        element.meta_data = {}

    if mode == 'replace':
        element.meta_data.update(meta_data)
    elif mode == 'add':
        for key, value in meta_data.items():
            if isinstance(value, list):
                if (isinstance(item, (int, float)) for item in value):
                    value = np.array(value)
                else:
                    raise ValueError(f"Value for key '{key}' must be a list of numeric values (int or float) with {mode=}.")
            if isinstance(value, (int, float, np.ndarray)):
                # Add to existing value if the key exists and is numeric
                if key in element.meta_data and isinstance(element.meta_data[key], (int, float, np.ndarray)):
                    element.meta_data[key] += value
                else:
                    # Add new key-value pair if the key doesn't exist
                    element.meta_data[key] = value
            else:
                raise ValueError(f"Value for key '{key}' must be numeric (int or float).")


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


def validate_invest_range(value: Union[int, float, str], label: str) -> Union[int, float, str]:
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
        try:  # Handle range strings of the format "X-Y"
            parts = value.split("-")
            if len(parts) != 2:
                raise ValueError("Invalid range format. Expected 'X-Y'.")
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                raise ValueError("Invalid range format. Expected 'X-Y'.")
            if start >= end:
                raise ValueError("Range start must be less than range end.")
            if start < 0:
                raise ValueError("Range start must be positive.")
            return value
        except ValueError as e:
            raise ValueError(f"Invalid thermal power format: {e}")
    else:
        raise ValueError(f"'{label}' must be a number or a string in the format 'X-Y'.")

