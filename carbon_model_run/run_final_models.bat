@echo off
setlocal
echo Installing required packages...
py -m pip install -r requirements_final_models.txt
if errorlevel 1 (
  echo Package installation failed.
  pause
  exit /b 1
)
echo.
echo Running the final learned-model experiment...
py carbon_forecast_models_final.py
if errorlevel 1 (
  echo The experiment failed. Read the error above.
  pause
  exit /b 1
)
echo.
echo Finished. Check results_all_models_summary.csv and results_all_models_summary.txt.
pause
