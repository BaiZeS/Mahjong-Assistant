from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Optional

class BackendManager:
    def __init__(self, port: int = 8765) -> None:
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger("mahjong.backend_manager")

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
            
        # Use sys.executable to ensure we use the same python interpreter
        self.process = subprocess.Popen(
            [sys.executable, "-m", "src.mahjong.backend", "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd()
        )
        self.logger.info("Backend started with PID %s", self.process.pid)

    def stop(self) -> None:
        if self.process:
            pid = self.process.pid
            self.logger.info("Stopping backend PID %s", pid)
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                
                if os.name == 'nt':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process = None
            self.logger.info("Backend stopped")

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None
