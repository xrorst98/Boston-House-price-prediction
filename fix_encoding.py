with open('boston_house_price_prediction.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('\u2714', '[OK]')
content = content.replace('\u2192', '->')
content = content.replace('\u2605', '*')

# Remove the sys.stdout wrapper if already added (idempotent)
content = content.replace(
    'import sys, io\nsys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")\nimport warnings',
    'import warnings'
)

# Add proper PYTHONUTF8 env hint at the very top
with open('boston_house_price_prediction.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Fixed encoding in boston_house_price_prediction.py')
