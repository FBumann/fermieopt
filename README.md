# README

## Zweck dieses Pakets
Dieses Paket bietet eine benutzerfreundliche Schnittstelle zur Optimierung von Fernwärmesystemen mit flixOpt.  
MS Excel wird verwendet, um alle Modellparameter zu definieren, was das Anpassen und Überprüfen der Parameter vereinfacht.  
Die Ergebnisse werden umfassend aufbereitet und sowohl in Excel-Dateien als auch in PDFs gespeichert. Dadurch können die Ergebnisse mit minimalem Aufwand und ohne Programmierkenntnisse analysiert werden.  

Die Modellberechnungen werden über **`run_model.py`** ausgeführt. Zusätzliche Auswertungen können nachträglich mit **`run_postprocessing.py`** ergänzt werden.  

## Erste Schritte
1. Erstellen Sie ein neues Python-Projekt in Ihrer IDE (PyCharm, Spyder, ...), idealerweise mit einer neuen virtuellen Umgebung (`.venv`).  
2. Installieren Sie dieses Paket mit:  
   ```bash
   pip install https://github.com/FBumann/fermieopt.git
   ```  
3. Laden Sie dieses Repository herunter. Kopieren Sie die Skripte **`run_model.py`**, **`run_postprocessing.py`** sowie die Datei **`Template_Input.xlsx`** in Ihr Projekt.  
4. Führen Sie **`run_model.py`** aus. Die Datei **`Template_Input.xlsx`** wird geladen und das darin definierte Modell erstellt.  
5. Definieren Sie den Speicherort der Eingabedatei in **`run_model.py`**.  
6. Das Modell wird erstellt und gelöst. Werfen Sie einen Blick in die erstellten Auswertungsdateien.  

## Nächste Schritte
1. Erstellen Sie eine Kopie des Templates und passen Sie die Parameter an Ihre Bedürfnisse an:
   - Definieren Sie den Speicherort der Ergebnisse.
   - Setzen Sie CO₂-Grenzwerte.
   - Geben Sie die Energiekosten für Strom, Gas, Wasserstoff etc. an.
   - Definieren Sie die Wärmelast.
   - Legen Sie die Vor- und Rücklauftemperaturen des Wärmenetzes sowie die Umgebungstemperatur fest. Falls keine genauen Werte vorliegen, können Schätzungen verwendet werden.
   - Definieren Sie vorhandene und optionale Wärmeerzeuger:  
     - KWK (Blockheizkraftwerk)  
     - Kessel  
     - Power-to-Heat  
     - Wärmepumpen  
     - Abwärme  
     - Speicher  
     - ...  
2. Passen Sie den Dateipfad und ggf. die gewünschten Auswertungen in **`run_model.py`** an unf führen Sie die Datei aus  
   - Das Modell wird erstellt und gelöst.  
   - Die Ergebnisse werden in Excel-Dateien mit Visualisierungen gespeichert.  
   - Zusätzliche Visualisierungen wie Heatmaps werden generiert.  
3. **Analysieren Sie die Ergebnisse Ihres Modells – die Dateien sind unter dem in der Eingabe-Excel-Datei angegebenen Pfad gespeichert.**  

## Hilfestellung
### Eingabe-Excel-Datei
Sie können automatisch Templates für alle verfügbaren Erzeugertypen generieren lassen:

```python
from fermieopt.DistrictHeatingComps import ElementFactory
ElementFactory.model_templates(optional_fields=True)
```