import subprocess
import sys
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print('Необходимые библиотеки установлены, для следующих запусков используйте main.py')
    subprocess.check_call([sys.executable, "main.py"])
except Exception as e:
    print(f"An error occurred: {e}")
    sys.exit(1)
