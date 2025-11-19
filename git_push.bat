@echo off
cd /d "c:\Users\binbi\Desktop\DataLab2Project"
echo Adding files to Git...
git add .
echo.
echo Making initial commit...
git commit -m "Initial commit: Vector database inspector and retrieval system"
echo.
echo Renaming branch to main...
git branch -M main
echo.
echo Pushing to GitHub...
git push -u origin main
echo.
echo Done!
pause