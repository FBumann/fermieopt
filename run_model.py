import logging
from pathlib import Path

from fermieopt.DistrictHeating import ExcelModel
from fermieopt.excel_output import cExcelFcts, create_report_grouped, visualize_results

logger = logging.getLogger('flixOpt')

# Specify paths and solver_name
excel_file_path = r'Template_Input.xlsx'  # path of excel input file

solver_name = 'highs'  # Choose open source solver highs
# solver_name = "gurobi"  # Choose commercial solver (Free academic licences). Much faster for large Models and storages


def main(solver_name: str, excel_file_path: str):
    excel_model = ExcelModel(excel_file_path=excel_file_path)
    excel_model.district_heating_system.final_model.visualize_network(False, controls=['physics'])
    excel_model.solve_model(solver_name=solver_name, gap_frac=0.0005, timelimit=2 * 3600)
    excel_model.district_heating_system.final_model.visualize_network(
        f'{excel_model.final_directory}/{excel_model.calc_name}_network.html', controls=['physics']
    )

    calc_results = excel_model.load_results()
    logger.info('START: EXPORT OF RESULTS TO EXCEL...')
    excel = cExcelFcts(calc_results)
    excel.run_excel_graphics_main()
    excel.run_excel_graphics_years(short_version=False)
    visualize_results(calc_results=calc_results)
    for bus in calc_results.bus_results:
        create_report_grouped(
            calc_results,
            path=Path(calc_results.folder).resolve() / f'{calc_results.name}__Report_{bus}.pdf',
            connected_to=bus,
        )
    # calculation_results_for_further_inspection = excel_model.load_results()


if __name__ == '__main__':
    main(solver_name, excel_file_path)

# optional: change values for gap_frac and timelimit
"""
:param gap_frac:
    0...1 ; gap to relaxed solution. Higher values for faster solving. 0...1
:param timelimit:
    timelimit in seconds. After this time limit is exceeded, the solution process is stopped and the best yet found result is used.
    If no result is found yet, the Process is aborted
"""
