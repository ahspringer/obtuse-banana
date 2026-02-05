from pathlib import Path
from typing import Optional

import pandas as pd


def convert_to_csv(input_path: str, output_path: Optional[str] = None, overwrite: bool = False) -> str:
    """Convert a table-like file to a comma-separated CSV.

    Supported input formats (auto-detected by extension or content):
    - Excel: .xls, .xlsx
    - JSON: .json (also tries line-delimited JSON)
    - Parquet: .parquet
    - Delimited text: comma, tab, semicolon, or whitespace-separated (.txt/.csv/.tsv)

    Returns the written CSV path.
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        outp = inp.with_suffix('.csv')
    else:
        outp = Path(output_path)

    if outp.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {outp}")

    suffix = inp.suffix.lower()
    df = None

    try:
        if suffix in ('.xls', '.xlsx'):
            df = pd.read_excel(inp)
        elif suffix == '.parquet':
            df = pd.read_parquet(inp)
        elif suffix == '.json':
            # try line-delimited first, then standard json
            try:
                df = pd.read_json(inp, lines=True)
            except ValueError:
                df = pd.read_json(inp)
        else:
            # Let pandas try to infer separator, then fall back to common ones.
            try:
                df = pd.read_csv(inp, sep=None, engine='python')
            except Exception:
                # try common explicit separators
                tried = False
                for sep in [',', '\t', ';']:
                    try:
                        df = pd.read_csv(inp, sep=sep)
                        tried = True
                        break
                    except Exception:
                        continue
                if not tried:
                    # final attempt using delim_whitespace
                    df = pd.read_csv(inp, delim_whitespace=True, engine='python')

        # Ensure we have a DataFrame
        if df is None:
            raise ValueError(f"Could not parse input file: {input_path}")

        # If pandas produced many 'Unnamed' columns (common with messy whitespace),
        # do a robust manual whitespace split fallback that uses Python's str.split().
        unnamed_count = sum(1 for c in df.columns if str(c).startswith('Unnamed'))
        if unnamed_count > 0 and unnamed_count >= (len(df.columns) // 3):
            # Manual parse: split each non-empty line by any whitespace
            with open(inp, 'r', encoding='utf-8') as fh:
                raw_lines = [l.rstrip('\n') for l in fh if l.strip()]

            parsed = [line.split() for line in raw_lines]

            # If first row looks like header (contains any non-numeric token), use it
            def _is_header_row(tokens):
                for t in tokens:
                    # consider it header if token contains alphabetic chars or underscores
                    if any(c.isalpha() for c in t) or '_' in t:
                        return True
                return False

            if parsed and _is_header_row(parsed[0]):
                header = parsed[0]
                rows = parsed[1:]
            else:
                # no header present: create generic column names based on max row length
                maxlen = max(len(r) for r in parsed)
                header = [f'col{i}' for i in range(maxlen)]
                rows = parsed

            # Normalize rows to header length (pad with empty strings)
            norm_rows = [r + [''] * (len(header) - len(r)) if len(r) < len(header) else r[:len(header)] for r in rows]

            df = pd.DataFrame(norm_rows, columns=header)
            # try converting numeric-like columns
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')

        # Drop columns that are entirely empty (all-NaN or whitespace-only strings).
        def _col_has_value(s: pd.Series) -> bool:
            non_na = s.dropna()
            if non_na.empty:
                return False
            return non_na.astype(str).str.strip().astype(bool).any()

        cols_to_drop = [c for c in df.columns if not _col_has_value(df[c])]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Write out as CSV
        df.to_csv(outp, index=False)
        return str(outp)

    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to convert {input_path} to CSV: {e}") from e


def _cli():
    import argparse

    parser = argparse.ArgumentParser(description="Convert various table formats to CSV")
    parser.add_argument('input', help='Input file path')
    parser.add_argument('-o', '--out', help='Output CSV path (defaults to same name with .csv)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output file')
    args = parser.parse_args()

    out = convert_to_csv(args.input, args.out, args.overwrite)
    print(f"Wrote CSV: {out}")


if __name__ == '__main__':
    _cli()
