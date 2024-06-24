from pathlib import Path

from fermieopt.flixPostXL import flixPostXL
from fermieopt.excel_output import visualize_results, create_report_grouped, cExcelFcts

# This expression can be found in every calculation folder as '[Name of calculation]_calc_info.txt'
r"""
calc_results = flixPostXL(
    nameOfCalc='2024-06-12_2024_03_25_Vergleichsrechnung_BoFiT',
    results_folder=r'C:\Users\FELIBUMA\Desktop\Modellergebnisse\2024-06-12_2024_03_25_Vergleichsrechnung_BoFiT\SolveResults',
    outputYears=[2030])
"""
calc_results.folder = r'C:/New\Path/for/evaluation/'  # Optionally new results folder.







excel = cExcelFcts(calc_results)
excel.run_excel_graphics_main()
excel.run_excel_graphics_years()

visualize_results(calc_results=calc_results,
                  effect_shares=True,
                  comps_yearly=True, buses_yearly=True, effects_yearly=True,
                  comps_daily=True, buses_daily=True, effects_daily=True,
                  comps_hourly=True, buses_hourly=True, effects_hourly=True)
for bus in calc_results.buses:
    create_report_grouped(calc_results,
                          path=Path(calc_results.folder).resolve() / f"{calc_results.label}-report_{bus}.pdf",
                          connected_to=bus)
