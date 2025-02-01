import logging

from fermieopt.DistrictHeating import ExcelModel
from fermieopt.excel_output import ExcelEvaluation, create_report_grouped, visualize_results

logger = logging.getLogger('flixOpt')

########################################################################################################################

excel_file_path = r'Template_Input.xlsx'  # path to the excel input file

solver_name = 'highs'                       # Preinstalled open source solver (default). Might take a while...
# solver_name = "gurobi"                    # Commercial solver (Free academic licences). Much faster...


def main(excel_file_path: str, solver_name: str = 'highs'):
    excel_model = ExcelModel(excel_file_path=excel_file_path)
    excel_model.final_model.visualize_network(False, controls=['physics'])
    excel_model.solve_model(solver_name=solver_name, gap_frac=0.0005, timelimit=2 * 3600)
    excel_model.final_model.visualize_network(
        excel_model.final_directory / f'{excel_model.calc_name}_network.html', controls=['physics']
    )

    calc_results = excel_model.load_results()
    logger.info('START: EXPORT OF RESULTS TO EXCEL...')
    excel = ExcelEvaluation(calc_results)
    excel.run_excel_graphics_main()
    excel.run_excel_graphics_years(short_version=False)
    visualize_results(calc_results=calc_results)
    for bus in calc_results.bus_results:
        create_report_grouped(
            calc_results,
            path=calc_results.folder / f'{calc_results.name}__Report_{bus}.pdf',
            connected_to=bus,
        )
    # calculation_results_for_further_inspection = excel_model.load_results()


if __name__ == '__main__':
    main(excel_file_path, solver_name)

# optional: change values for gap_frac and timelimit
"""
:param gap_frac:
    0...1 ; gap to relaxed solution. Higher values for faster solving. 0...1
:param timelimit:
    timelimit in seconds. After this time limit is exceeded, the solution process is stopped and the best yet found result is used.
    If no result is found yet, the Process is aborted
"""
