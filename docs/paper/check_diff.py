import sys

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

text_access = read_file('access.tex')
text_hl = read_file('Highlighted PDF.tex')

# Strip added preamble from HL
preamble_to_remove = r'''\usepackage{soul}
\usepackage{mdframed}
\newmdenv[hidealllines=true,backgroundcolor=yellow,innerleftmargin=2pt,innerrightmargin=2pt,innertopmargin=2pt,innerbottommargin=2pt]{highlight}
'''
text_hl = text_hl.replace(preamble_to_remove, '')

# Strip begin/end highlight
text_hl = text_hl.replace(r'\begin{highlight}' + '\n', '')
text_hl = text_hl.replace('\n' + r'\end{highlight}', '')
text_hl = text_hl.replace(r'\begin{highlight}', '')
text_hl = text_hl.replace(r'\end{highlight}', '')

if text_access == text_hl:
    print('100% MATCH')
else:
    print('DIFFERENCES FOUND!')
    # Find all differences
    for i in range(min(len(text_access), len(text_hl))):
        if text_access[i] != text_hl[i]:
            print(f'First difference at index {i}')
            print(f'access.tex: {repr(text_access[max(0, i-40):i+40])}')
            print(f'Highlighted: {repr(text_hl[max(0, i-40):i+40])}')
            break
    if len(text_access) != len(text_hl):
        print(f'Length difference! access.tex: {len(text_access)}, Highlighted: {len(text_hl)}')
