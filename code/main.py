import sys



# Ensure not imported
# --------------------

if __name__ != "__main__":
    print('"main" module must not be imported.', file = sys.stderr)
    sys.exit(1)
