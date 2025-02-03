from pathlib import Path

from fermieopt.excel_output import Auswertung, erstelle_gruppierten_pdf_report, exportiere_ergebnissdetails
from fermieopt.flixPostprocessingXL import FlixPostXL

calc_results = FlixPostXL(
    calculation_name='2025-02-01-16H-45M_Template_Input',
    results_folder=r'Ergebnisse/2025-02-01-16H-45M_Template_Input/SolveResults',
    output_years=[2030, 2045],
)


auswertung = Auswertung(calc_results)
auswertung.effekte_pro_jahr()
auswertung.vollbenutzungsstunden_pro_jahr()
auswertung.invest_entscheidungen()
