"""Main

driver for running CLAuDE
"""

from Settings import Settings

from Claude import Claude

def main():
    # IF YOU WANT TO CHANGE SETTINGS, MODIFY `Settings.py`
    settings = Settings()
    claude = Claude(settings)
    claude.run()

if __name__ == '__main__':
    main()