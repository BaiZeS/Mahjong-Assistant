import sys
import os
import logging
import faulthandler
from PyQt6 import QtWidgets
from src.mahjong.gui import MainWindow

def main() -> None:
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "app.log")
    crash_path = os.path.join(log_dir, "crash.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    
    crash_file = open(crash_path, "a", encoding="utf-8")
    faulthandler.enable(file=crash_file, all_threads=True)
    
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        logging.getLogger("mahjong.app").exception(
            "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
        )
    sys.excepthook = handle_exception
    
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
