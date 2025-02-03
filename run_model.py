import logging

from fermieopt.DistrictHeating import ExcelModel
from fermieopt.excel_output import Auswertung, create_report_grouped, visualize_results

logger = logging.getLogger('flixOpt')

########################################################################################################################

excel_file_path = r'Template_Input.xlsx'  # Pfad zur Excel-Eingabedatei

solver_name = 'highs'  # Vorinstallierter Open-Source-Solver (Standard). Kann länger dauern...
# solver_name = 'gurobi'  # Kommerzieller Solver (kostenlose akademische Lizenzen verfügbar). Deutlich schneller...


def main(excel_file_path: str, solver_name: str = 'highs'):
    excel_model = ExcelModel(excel_file_path=excel_file_path)
    excel_model.final_model.visualize_network(False, controls=['physics'])
    excel_model.solve_model(solver_name=solver_name, gap_frac=0.0005, timelimit=2 * 3600)
    excel_model.final_model.visualize_network(
        excel_model.final_directory / f'{excel_model.calc_name}_network.html', controls=['physics']
    )

    calc_results = excel_model.load_results()
    logger.info('EXPORT DER ERGEBNISSE NACH EXCEL...')
    excel = Auswertung(calc_results)
    excel.exportiere_ergebnisuebersicht()
    excel.exportiere_ergebnisse_je_jahr(short_version=False)
    visualize_results(calc_results=calc_results)
    for bus in calc_results.bus_results:
        create_report_grouped(
            calc_results,
            path=calc_results.folder / f'{calc_results.name}__Report_{bus}.pdf',
            connected_to=bus,
        )
    # Berechnungsergebnisse für weitere Analysen laden
    # calculation_results_for_further_inspection = excel_model.load_results()


if __name__ == '__main__':
    main(excel_file_path, solver_name)
