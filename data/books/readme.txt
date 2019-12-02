https://data.bl.uk/digbks/?_ga=2.23015995.1012810978.1561554810-1580931933.1561554810

parsing:
    - find . -name "*.zip" -exec unar {} \;
    - find . -name "*.zip" -exec rm {} \;