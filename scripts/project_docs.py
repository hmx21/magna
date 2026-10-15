#!/usr/bin/env python3
"""Inspect maintained documentation and validate local Markdown links and repository paths."""
import argparse
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
ROOT = Path(__file__).resolve().parents[1]
# Inline-code tokens that look like repository paths, with an optional :line or :line-line suffix.
PATH_TOKEN = re.compile(r'(?:\./)?((?:core|docs|scripts|tests|examples|benchmarks|data)/[^\s*?\[\]{}<>$\u2026]*?)(?::\d+(?:-\d+)?)?')
# Intentionally historical path references, as (document, path); keep this list small.
HISTORICAL_PATHS = {
    # The generated benchmark ledger names the retired folder its RP2 inputs were packaged from.
    (Path('docs/development/BENCHMARKS.md'), 'examples/rp2'),
}
def documents():
    paths = [ROOT/'README.md', ROOT/'AGENTS.md', ROOT/'core/README.md', ROOT/'core/magna/README.md', ROOT/'Literature/README.md', ROOT/'Literature/WORKFLOW.md', ROOT/'Literature/RESTORE.md', ROOT/'paper/README.md']
    # Optional entries are absent from the public export.
    paths = [path for path in paths if path.exists()]
    return sorted(set(paths + list((ROOT/'docs').rglob('*.md')) + list((ROOT/'examples').rglob('*.md')) + list((ROOT/'benchmarks').rglob('*.md')) + list((ROOT/'tests/fixtures').rglob('*.md'))))
def missing_paths(path, text):
    """Repository paths written in inline code must exist, relative to the root or the document."""
    rel = path.relative_to(ROOT)
    if rel.parts[:3] == ('docs', 'development', 'benchmark-results'): return []  # recorded evidence
    text = re.sub(r'```.*?```', '', text, flags=re.S)
    text = re.sub(r'\[`[^`\n]*`\]\([^)]*\)', '', text)  # linked code is checked as a link
    errors = []
    for token in re.findall(r'`([^`\n]+)`', text):
        match = PATH_TOKEN.fullmatch(token.strip())
        if not match or '...' in token or (rel, match.group(1)) in HISTORICAL_PATHS: continue
        if not ((ROOT/match.group(1)).exists() or (path.parent/match.group(1)).exists()):
            errors.append(f'{rel}: missing path `{token}`')
    return errors
def check():
    errors = []
    for path in documents():
        text = path.read_text()
        if re.search(r'[\u3400-\u9fff]', text): errors.append(f'{path.relative_to(ROOT)}: documentation must be English')
        errors += missing_paths(path, text)
        if re.search(r'^\s*=($|\s)',text,re.M): errors.append(f'{path.relative_to(ROOT)}: equation begins with =')
        # Ignore examples and inline code when validating authored links.
        prose = re.sub(r'```.*?```', '', text, flags=re.S)
        prose = re.sub(r'`[^`\n]*`', '', prose)
        for target in re.findall(r'\]\(([^)]+)\)',prose):
            target=target.split(' "')[0].strip('<>')
            if urlsplit(target).scheme or target.startswith('#'): continue
            local=unquote(target.split('#')[0])
            if local and not (path.parent/local).exists(): errors.append(f'{path.relative_to(ROOT)}: missing link {target}')
    print('\n'.join(errors) if errors else f'{len(documents())} documents: language policy, local links, repository paths and equation formatting checked')
    return bool(errors)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['check','list','show','read']);p.add_argument('path',nargs='?');p.add_argument('--topic');p.add_argument('--section');args=p.parse_args()
    if args.command=='check':return int(check())
    if args.command=='list':
        for f in documents():
            if not args.topic or args.topic.lower() in str(f.relative_to(ROOT)).lower():print(f.relative_to(ROOT))
    else:
        if not args.path:p.error('a document path is required')
        f=(ROOT/args.path).resolve()
        if f not in documents():p.error('not a maintained documentation path')
        text=f.read_text()
        if args.command=='show':text='\n'.join(text.splitlines()[:20])
        elif args.section:
            lines=text.splitlines();start=next((i for i,l in enumerate(lines) if l.lstrip('# ').strip()==args.section),None)
            if start is None:p.error('section not found')
            depth=len(lines[start])-len(lines[start].lstrip('#'));end=next((i for i in range(start+1,len(lines)) if re.match(r'#{1,'+str(depth)+r'} ',lines[i])),len(lines));text='\n'.join(lines[start:end])
        print(text)
    return 0
if __name__=='__main__':raise SystemExit(main())
