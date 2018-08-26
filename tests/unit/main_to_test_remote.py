import os
import sys
if __name__ == '__main__':
    print('Hello from python in docker!!!')
    print(os.listdir('.'))
    print(os.path.abspath(os.curdir))
    os.system('python -m pytest '
                        '/emp_priv/tests/unit/test_framework_units.py')