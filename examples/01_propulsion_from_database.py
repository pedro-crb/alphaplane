import alphaplane as ap

motor = ap.ElectricMotor.from_database('4025-330', 'Scorpion')
prop = ap.Propeller.from_APC_database('20x8E')
battery = ap.Battery.from_lipo(6)
gmp = ap.PropulsionSystem(prop, motor, battery)
# prop.show()
thrust = gmp.run(0, 'electrical_power', 700, propeller_method='qprop', density=1.06)[0]['thrust']
print(thrust)