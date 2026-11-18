# Makes tests/ a REGULAR package so `python -m tests.<module>` always resolves to THIS
# directory. Without it, tests/ is a namespace package — and a third-party dependency that
# (badly) ships a top-level `tests` package in site-packages would win the import instead.
