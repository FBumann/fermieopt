from fermieopt.DistrictHeating import ExcelModel
from fermieopt.excel_output import visualize_results, create_report_grouped, cExcelFcts
from pathlib import Path

# Specify paths and solver_name
excel_file_path = r'Template_Input.xlsx'  # path of excel input file

solver_name = "highs"    # Choose open source solver highs
#solver_name = "gurobi"  # Choose commercial solver (Free academic licences). Much faster for large Models and storages


def main():
    excel_model = ExcelModel(excel_file_path=excel_file_path)
    excel_model.visual_representation.show()
    excel_model.solve_model(solver_name=solver_name, gap_frac=0.01, timelimit=36000)

    calc_results = excel_model.load_results()
    print("START: EXPORT OF RESULTS TO EXCEL...")
    excel = cExcelFcts(calc_results)
    excel.run_excel_graphics_main()
    excel.run_excel_graphics_years()
    visualize_results(calc_results=calc_results,
                      effect_shares= True,
                      comps_yearly=True, buses_yearly=True, effects_yearly=True,
                      comps_daily=True, buses_daily=True, effects_daily= True,
                      comps_hourly=True, buses_hourly=True, effects_hourly=True)
    for bus in calc_results.buses:
        create_report_grouped(calc_results,
                              path= Path(calc_results.folder).resolve() / f"{calc_results.label}__Report_{bus}.pdf",
                              connected_to=bus)
    # calculation_results_for_further_inspection = excel_model.load_results()









if __name__ == '__main__':
    main()

# optional: change values for gap_frac and timelimit
'''
:param gap_frac:
    0...1 ; gap to relaxed solution. Higher values for faster solving. 0...1
:param timelimit:
    timelimit in seconds. After this time limit is exceeded, the solution process is stopped and the best yet found result is used. 
    If no result is found yet, the Process is aborted
'''