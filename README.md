# README

## Zweck dieses Pakets
Dieses Paket dient als benutzerfreundliche Schnittstelle zur Optimierung von Fernwärmesystemen mit flixOpt.  
MS Excel wird verwendet, um alle Parameter des Modells zu definieren, was das Ändern und Überprüfen der Parameter erleichtert.  
Die Ergebnisse werden umfassend aufbereitet und in Excel-Dateien und PDF's gespeichert. Somit können Ergbisse mit geringem Aufwand und ohne Programmierkenntnisse analysiert werden.
Dieses Paket ist eine Erweiterung von flixOpt. 
Es wird durch die vorbereiteten Skripte **`run_model.py`** ausgefährt.
Auswertungen können zusätzlich auch im nachinein über **`run_postprocessing.py`** erweitert werden.  

## Erste Schritte
1. Erstellen Sie ein neues Python-Projekt in Ihrer IDE (PyCharm, Spyder, ...) (idealerweise mit einer neuen virtuellen Umgebung `.venv`).  
2. Installieren Sie dieses Paket mit in Ihre Umgebung:  
   ```bash
   pip install https://github.com/FBumann/fermieopt.git
   ```  
3. Laden Sie dieses Repository herunter. Kopieren Sie die Skripte **`run_model.py`** und **`run_postprocessing.py`** sowie die Datei **`Template_Input.xlsx`** in Ihr Projekt.
4. Führen Sie die Datei **`run_model.py`** aus. Die Datei **`Template_Input.xlsx`** wird geladen und das darin definierte Modell erstellt.
5. Fügen Sie den Pfad der neuen Datei in die Datei **`run_model.py`** ein.
6. Das Modell wird erstellt und gelöst. Werfen Sie einen Blick in die erstellten Auswertungsdateien.

## Nächste Schritte:
1. Erstellen Sie eine Kopie des Templates. Fügen Sie ihre eigenen Parameter udn Annahmen ein.
   1. Legen Sie fest, wo die Ergebnisse gespeichert werden sollen.  
   2. Definieren Sie CO₂-Grenzwerte.  
   3. Geben Sie die Kosten für verschiedene Energieträger an (Strom, Gas, Wasserstoff, ...).  
   4. Definieren Sie die Wärmelast.  
   5. Geben sie die Vor-und Rücklauftemperaturen des Wärmenetzes sowie die Umgebungstemperatur an. Dabei kann es sich zunächst auch um geschätzte Werte handlen.  
   6. Definieren Sie alle vorhandenen und optionalen Wärmeerzeuger:  
      - KWK (Blockheizkraftwerk)  
      - Kessel  
      - Power-to-Heat
      - Wärmepumpen 
      - Abwärme  
      - Speicher
      - ...  
2. Führen Sie die Datei **`run_model.py`** aus.  
   1. Das Modell wird erstellt und gelöst.  
   2. Die Ergebnisse werden in Excel-Dateien mit Visualisierungen gespeichert.  
   3. Weitere Visualisierungen (z. B. Heatmaps) werden erstellt.  
3. **Abschließend: Analysieren Sie die Ergebnisse Ihres Modells. Die Dateien sind unter dem in der Eingabe-Excel-Datei angegebenen Pfad gespeichert.**


## Hilfestellung
### Eingabe-Excel-Datei
Die Eingabe-Excel-Datei ist eine Excel-Datei, die die Parameter des Modells definiert. Sie können für alle verfügbaren Erzeugertypen ein Template erstellen lassen.
Führen sie dafür folgenden Befehl aus:
```python
from fermieopt.DistrictHeatingComps import ElementFactory
ElementFactory.model_templates()
```
