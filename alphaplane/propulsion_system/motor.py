from __future__ import annotations

import pandas as pd
import os

from typing import Self

import numpy as np


class ElectricMotor:
    def __init__(self, kv: float,
                 resistance: float,
                 no_load_amps: float = 0.0,
                 max_amps: float | None = None,
                 max_volts: float | None = None,
                 max_watts: float | None = None,
                 weight: float | None = None,
                 gear_ratio: float = 1.0,
                 name: str = 'Unnamed Motor',
                 brand: str = 'No Brand'
                 ) -> None:
        self.kv: float = kv
        self.resistance: float = resistance
        self.no_load_amps: float = no_load_amps
        self.max_amps: float | None = max_amps
        self.max_volts: float | None = max_volts
        self.max_watts: float | None = max_watts
        self.weight: float | None = weight
        self.gear_ratio: float = gear_ratio
        self.name: str = name
        self.brand: str = brand

    def run(self, volts: float, amps: float) -> dict[str, float]:
        power_in = volts * amps

        motor_output = {
            'volts_out': volts - amps * self.resistance,
            'amps_out': amps - self.no_load_amps
        }

        motor_output['power_out'] = motor_output['volts_out'] * motor_output['amps_out']

        motor_output['rpm'] = motor_output['volts_out'] * self.kv / self.gear_ratio
        motor_output['torque'] = motor_output['power_out'] / (2*np.pi*motor_output['rpm'] / 60)
        motor_output['efficiency'] = motor_output['power_out']/power_in

        return motor_output

    def to_qprop(self, file_path: str) -> None:
        with open(file_path, 'w') as file:
            file.write(f'{self.name}    ! name\n')
            file.write(f'\n')
            file.write(f' 1     ! motor type\n')
            file.write(f'\n')
            file.write(f'{self.resistance:.4f}    ! R\n')
            file.write(f'{self.no_load_amps:.4f}    ! I0\n')
            file.write(f'{self.kv:.4f}    ! Kv\n')

    @classmethod
    def from_database(cls, motor_name: str, motor_brand: str | None = None) -> Self:
        script_dir = os.path.dirname(__file__)
        database_path = os.path.join(script_dir, 'motors.json')
        df = pd.read_json(database_path)
        
        norm_search_name = motor_name.strip().lower()
        df['norm_name'] = df['Motor Name'].apply(lambda x: x.strip().lower())
        
        if motor_brand:
            norm_search_brand = motor_brand.strip().lower()
            df['norm_brand'] = df['Motor Brand'].apply(lambda x: x.strip().lower() if isinstance(x, str) else '')
            matches = df[(df['norm_name'].apply(lambda x: norm_search_name in x)) & 
                        (df['norm_brand'].apply(lambda x: norm_search_brand in x))]
        else:
            matches = df[df['norm_name'].apply(lambda x: norm_search_name in x)]
        
        if matches.empty:
            brand_info = f' and brand "{motor_brand}"' if motor_brand else ''
            raise ValueError(f'No motor matching name "{motor_name}"{brand_info} found in database.')
        
        shortest_match = matches.loc[matches['Motor Name'].str.len().idxmin()]
        assert isinstance(shortest_match, pd.Series)
        
        motor = cls(
            kv=shortest_match['Motor Kv'],
            resistance=shortest_match['Internal Resistance'] / 1000,
            no_load_amps=shortest_match['No Load Current'],
            max_amps=shortest_match.get('Max Continuous Current'),
            max_watts=shortest_match.get('Max Continuous Power'),
            max_volts=shortest_match.get('Max Lipo Cells', 0) * 4.2,
            weight=shortest_match['Weight'] / 1000,
            gear_ratio=shortest_match['Motor Gear Ratio'],
            name=shortest_match['Motor Name'],
            brand=shortest_match['Motor Brand']
        )
        return motor

    @staticmethod
    def database_read() -> pd.DataFrame:
        script_dir = os.path.dirname(__file__)
        database_path = os.path.join(script_dir, 'motors.json')
        df = pd.read_json(database_path)
        return df
