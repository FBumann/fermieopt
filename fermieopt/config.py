from typing import List


class SingletonMeta(type):
    """Metaclass to ensure a single instance per class."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
    
    
class LabelBase(metaclass=SingletonMeta):
    
    @classmethod
    def all_values(cls) -> List:
        return [
            value for key, value in cls.__dict__.items()
            if not key.startswith('__') and not callable(value) and not isinstance(value, (staticmethod, classmethod))
        ]


class BusLabels(LabelBase):
    HEAT = "Fernwärme"
    ELECTRICITY_IN = "Strom In"
    ELECTRICITY_OUT = "Strom Out"
    GAS = "Erdgas"
    HYDROGEN = "Wasserstoff"
    WASTE_HEAT = "Abwärme"


class SinkLabels(LabelBase):
    HEAT_DEMAND = "Wärmebedarf"
    HEAT_LOSSES = "Netzverluste"
    ELECTRICITY_PRODUCTION = "Strom Verkauf"


class SourceLabels(LabelBase):
    ELECTRICITY_USAGE = "Strombedarf"
    GAS_USAGE = "Gas Bedarf"
    HYDROGEN_USAGE = "Wasserstoff Bedarf"
    WASTE_HEAT_USAGE = "Abwärme Bedarf"


class OtherLabels(LabelBase):
    ENERGY_PRICES = "Energiepreise"


class EffectLabels(LabelBase):
    COSTS = 'costs'
    CO2 = 'CO2'
    CO2_HEAT = 'CO2FW'
    OBJECTIVE = 'target'
    FUNDING = 'funding'
    GREEN_HEAT = 'Gruene_Waerme'


class TemperatureLabels(LabelBase):
    NETWORK_FORWARD = 'Vorlauftemperatur Fernwärmenetz [°C]'
    NETWORK_RETURN = 'Rücklauftemperatur Fernwärmenetz [°C]'
    AMBIENT_AIR = 'Umgebungstemperatur [°C]'
    DEFAULT_SPREAD = 65


class EnergyPriceLabels(LabelBase):
    GAS = 'Erdgas-Preis [€/MWh_hu]'
    HYDROGEN = 'H2-Preis [€/MWh_hu]'
    ELECTRICITY = 'Strom-Preis [€/MWh]'
    CO2 = 'CO2-Preis [€/t]'


class EnergyTypeLabels(LabelBase):
    GAS = 'Erdgas'
    HYDROGEN = 'Wasserstoff'
    ELECTRICITY = 'Strom'


class FuelTypeToPriceMapping:
    """Dynamically maps fuel types to energy price labels."""

    @staticmethod
    def get_price_label(fuel_type: str) -> str:
        """Retrieve the price label corresponding to a fuel type."""
        mapping ={
            EnergyTypeLabels.GAS: EnergyPriceLabels.GAS,
            EnergyTypeLabels.HYDROGEN: EnergyPriceLabels.HYDROGEN,
        }
        return mapping.get(fuel_type, fuel_type)
