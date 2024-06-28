# README
## Purpose of this Package
THis package serves as an accessible interface for Optimizing district heating systems with flixOpt. 
MS Excel is used to define all Parameters of the model, which makes changing and reviewing parameters easy. 
The Evaluation process is also greatly simplified by this module, as many grafics are created (mostly in excel). 
Furthermore, relevant data gets written to excel as well, and can therefore be analyzed easily

This package is an Extension of flixOpt. It's to be used by running the prepared skripts 'run_model.py' or 'run_postprocessing.py'.

## Useage
1. Create a new Python project in your IDE (PyCharm, Spyder, ...) (If possible with a new .venv)
2. Install this package via pip in to your environment: `pip install https://github.com/FBumann/fermieopt.git`
3. Copy the skripts __run_model.py__ and __run_postprocessing.py__ file from this package into your project.
4. Make a local copy of the __Template_Input_Free.xlsx__ and save it somewhere on your Computer (for. ex. Desktop)
5. Copy the path of the new file into the __run_model.py__ file
6. **Edit the Excel-file to initialize your Model.**
   1. Specify the path, where the results should be saved to.
   2. Specify CO2-Limits
   3. Specify Costs for Electricity, Gas, H2, ...
   4. Specify the Heat Demand
   5. If needed, specify the Temperature of the Heating Network, the surrounding Air and other Heat sources for Heat Pumps
   6. Specify all existing and optional Heat-Generators
      1. CHP
      2. Boiler
      3. P2H 
      4. HeatPumps
      5. Other Heat Sources
      6. ...
7. Run the __run_model.py__ file
   8. The model gets created and solved
   9. The Results are saved in excel-files with visualizations
   10. Further Vizualisations are made (HeatMaps, ...)
8. __Finally: Analyse the results of your Model. It's saved under the path you specified in the input-excel-file__
