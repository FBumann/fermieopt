from pathlib import Path

from fermieopt.excel_output import ExcelFcts, create_report_grouped, visualize_results
from fermieopt.flixPostprocessingXL import FlixPostXL

calc_results = FlixPostXL(
    calculation_name='2025-01-28-16H-30M_Test',
    results_folder=r'/Users/felix/Documents/Dokumente-eigene/Code/FermIE/tests/2025-01-28-16H-30M_Test/SolveResults',
    output_years=[2030,2045],
)


# excel = ExcelFcts(calc_results)
# excel.run_excel_graphics_main()
# excel.run_excel_graphics_years()

visualize_results(
    calc_results=calc_results,
)
for bus in calc_results.bus_results:
    create_report_grouped(
        calc_results,
        path=calc_results.folder / f'{calc_results.name}-report_{bus}.pdf',
        connected_to=bus,
    )
