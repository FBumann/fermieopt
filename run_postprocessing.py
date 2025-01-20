from pathlib import Path

from fermieopt.excel_output import cExcelFcts, create_report_grouped, visualize_results
from fermieopt.flixPostprocessingXL import flixPostXL

calc_results = flixPostXL(nameOfCalc='2025-01-06_Basis+TAB-2030',
            results_folder=r'C:\Users\FELIBUMA\Downloads\Test_new_model\2025-01-06_Basis+TAB-2030\SolveResults',
            outputYears=[2030])



#excel = cExcelFcts(calc_results)
#excel.run_excel_graphics_main()
#excel.run_excel_graphics_years()

visualize_results(calc_results=calc_results,
                  comps_yearly=True, buses_yearly=True, effects_yearly=True,
                  comps_daily=True, buses_daily=True, effects_daily=True,
                  comps_hourly=True, buses_hourly=True, effects_hourly=True)
for bus in calc_results.bus_results:
    create_report_grouped(calc_results,
                          path=Path(calc_results.folder).resolve() / f"{calc_results.label}-report_{bus}.pdf",
                          connected_to=bus)
