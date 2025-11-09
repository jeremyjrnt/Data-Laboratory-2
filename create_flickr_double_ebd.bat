@echo off
REM Script to create Flickr Double Embedding VectorDB
REM Activates virtual environment and runs the indexer

echo ================================================================================
echo FLICKR DOUBLE EMBEDDING VECTORDB CREATION
echo ================================================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at .venv
    echo Please create virtual environment first with: python -m venv .venv
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated
echo.

REM Run the indexer script
echo Starting double embedding indexer...
echo.
python src\indexer\create_flickr_double_ebd.py

REM Check if script ran successfully
if errorlevel 1 (
    echo.
    echo ERROR: Indexer script failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo DONE!
echo ================================================================================
echo.
pause
