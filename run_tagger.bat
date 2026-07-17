@echo off
REM Danbooru Tagger launcher
REM A window will pop up asking you to choose the folder of images to tag.
REM You can change the model or threshold below if you like, but the defaults
REM work well for most people.

SET THRESHOLD=0.35
SET MODEL=wd-swinv2-v3

python tagger.py --model %MODEL% --threshold %THRESHOLD%
pause
