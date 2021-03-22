import os
import sys
import uvicorn
from runner import app

if __name__ == '__main__':
    if 'serve' in sys.argv:
        if os.getenv('ON_RENDER') is not None:
            uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
        else:
            uvicorn.run("server:app", host='0.0.0.0', port=5000, reload=True, log_level="info")
