import sys
import uvicorn
from runner import app

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run("server:app", host='0.0.0.0', port=5000, reload=True, log_level="info")
