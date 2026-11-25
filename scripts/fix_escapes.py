"""
Fix invalid escape sequences in Python string literals across UI pages.
Replaces single backslash followed by letters (not valid escapes) with double backslash.
"""
import re
from pathlib import Path

# Valid Python escape sequences to leave alone
VALID_ESCAPES = set('nrtfvab0\\\'\"xuU\n')

def fix_escapes_in_file(filepath: Path) -> int:
    """Returns number of fixes made."""
    content = filepath.read_text(encoding='utf-8')
    lines = content.splitlines()
    fixed_lines = []
    fixes = 0
    
    for lineno, line in enumerate(lines, 1):
        new_line = line
        # Find all backslash sequences: \X where X is a letter
        # We need to be careful not to double-fix already-fixed \\X
        result = []
        i = 0
        s = line
        while i < len(s):
            if s[i] == '\\':
                if i + 1 < len(s):
                    next_char = s[i + 1]
                    if next_char == '\\':
                        # Already doubled, skip both
                        result.append('\\\\')
                        i += 2
                        continue
                    elif next_char in VALID_ESCAPES:
                        # Valid escape, keep as-is
                        result.append('\\')
                        result.append(next_char)
                        i += 2
                        continue
                    elif next_char.isalpha() or next_char in '{}[]()^_':
                        # Likely invalid escape in a markdown/LaTeX string
                        result.append('\\\\')
                        result.append(next_char)
                        i += 2
                        fixes += 1
                        continue
                result.append(s[i])
                i += 1
            else:
                result.append(s[i])
                i += 1
        new_line = ''.join(result)
        fixed_lines.append(new_line)
    
    new_content = '\n'.join(fixed_lines)
    if content.endswith('\n'):
        new_content += '\n'
    
    if new_content != content:
        filepath.write_text(new_content, encoding='utf-8')
    
    return fixes


ui_dir = Path('src/ui')
total_fixes = 0
for pyfile in sorted(ui_dir.glob('*.py')):
    n = fix_escapes_in_file(pyfile)
    if n > 0:
        print(f"  Fixed {n} escapes in {pyfile.name}")
    total_fixes += n

print(f"\nTotal escape fixes: {total_fixes}")
